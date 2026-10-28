"""Lightning adapters; optimization and checkpoint state stay with Lightning."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader, WeightedRandomSampler, Sampler
from lightning.pytorch import LightningModule, LightningDataModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from ..training.config import ModelTypes, Checkpoints, TrainingConfig
from ..training.selection import Selection
from .data import index_of
from .model import (backbone_name, build_backbone, augmentation_pipeline, preprocess,
                    load_checkpoint, validate_preprocessing)


class _IdentityModule(LightningModule):
    def __init__(self, identity, *, pretrained=True, instance=None, resolution=None):
        super().__init__()
        self.identity = dict(identity)
        for key, value in dict(loss="subcenter_arcface", sub_centers=3, freeze_backbone=False,
                               lr=1e-3, weight_decay=1e-4).items():
            self.identity.setdefault(key, value)
        self.encoder, features, prep = build_backbone(identity["backbone"], pretrained=pretrained, instance=instance,
            resolution=resolution or identity.get("preprocessing", {}).get("resolution"))
        self.identity["preprocessing"] = validate_preprocessing(identity.get("preprocessing", prep))
        dim = identity.get("embedding_dim", 256)
        self.projection = torch.nn.Linear(features, dim) if dim else torch.nn.Identity()
        self.identity["embedding_dim"] = dim
        dim = dim or features
        if identity.get("freeze_backbone", False):
            self.encoder.requires_grad_(False)
        loss = identity.get("loss", "subcenter_arcface")
        if loss == "cross_entropy":
            self.head = torch.nn.Linear(dim, len(identity["identities"]))
            self.criterion = torch.nn.CrossEntropyLoss()
        elif loss == "subcenter_arcface":
            from pytorch_metric_learning.losses import SubCenterArcFaceLoss
            self.head = SubCenterArcFaceLoss(num_classes=len(identity["identities"]), embedding_size=dim,
                                            sub_centers=identity.get("sub_centers", 3))
            self.criterion = None
        else:
            raise ValueError("loss must be subcenter_arcface or cross_entropy")

    def forward(self, images):
        features = self.encoder(images)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("Backbones/custom models must return an N×D feature tensor")
        return torch.nn.functional.normalize(self.projection(features), dim=1)

    def train(self, mode=True):
        super().train(mode)
        if self.identity.get("freeze_backbone"):
            self.encoder.eval()
        return self

    def _step(self, batch, stage):
        images, labels = batch
        embeddings = self(images)
        loss = self.criterion(self.head(embeddings), labels) if self.criterion is not None else self.head(embeddings, labels)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, batch_size=len(labels))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW((p for p in self.parameters() if p.requires_grad),
                                lr=self.identity.get("lr", 1e-3), weight_decay=self.identity.get("weight_decay", 1e-4))

    def on_save_checkpoint(self, checkpoint):
        checkpoint["identity"] = self.identity

    def on_train_epoch_start(self):
        # Rebuild epoch-local sampling/transforms reproducibly, including after resume.
        seed_everything(self.identity.get("seed", 7)+self.current_epoch, workers=True, verbose=False)
        self.trainer.datamodule.train_data.epoch = self.current_epoch
        self.trainer.datamodule.batch_sampler.epoch = self.current_epoch


class _Pixels(TorchDataset):
    def __init__(self, index, preprocessing, labels, augmentations=None, *, training=False, seed=7):
        self.index, self.preprocessing, self.labels = index, preprocessing, labels
        self.training, self.seed, self.epoch = training, seed, 0
        self.transform, self.serialized = augmentation_pipeline(augmentations, seed=seed) if training else (None, None)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        # Include draw ordinal so repeated images get fresh, reproducible augmentation.
        if isinstance(key, tuple):
            position, draw, epoch = key
        else:
            position, draw, epoch = key, key, self.epoch
        sample = self.index[position]
        image = sample.image_ref.read()
        if self.transform is not None:
            self.transform.set_random_seed((self.seed + epoch*1_000_003 + draw) % (2**32))
            image = self.transform(image=image)["image"]
        return preprocess(image, self.preprocessing), self.labels[sample.identity_id]


class _DiverseBatches(Sampler):
    """Pack library-weighted draws while honoring optional metadata diversity caps.

    Oversubscribed groups end a smaller batch. No unbounded retry/buffer and no
    discarded observations: the workload and weighted draw distribution are exact.
    """
    def __init__(self, index, batch_size, seed):
        self.index, self.batch_size, self.seed, self.epoch = index, batch_size, seed, 0
        self.weights = torch.as_tensor(index.weights(), dtype=torch.double)
        self.workload = index.sampling.get("samples_per_epoch") or len(index)
        self.caps = index.sampling.get("max_per_batch", {})
        self.keys = {}
        for field in self.caps:
            columns = ["clip_id", "tracklet_id"] if field == "tracklet_id" else [field]
            self.keys[field] = list(index.frame[columns].itertuples(index=False, name=None))

    def __iter__(self):
        epoch = self.epoch
        generator = torch.Generator().manual_seed(self.seed+epoch)
        sampler = WeightedRandomSampler(self.weights, self.workload, replacement=True, generator=generator)
        counts, batch = {}, []
        for draw, row in enumerate(sampler):
            keys = [(field, values[row]) for field, values in self.keys.items()]
            if batch and (len(batch) == self.batch_size or any(counts.get(key, 0) >= self.caps[key[0]] for key in keys)):
                yield batch
                counts, batch = {}, []
            batch.append((row, draw, epoch))
            for key in keys:
                counts[key] = counts.get(key, 0)+1
        if batch:
            yield batch

    # Exact length is deliberately unspecified with caps; iterating the sampler
    # solely to count batches would duplicate a potentially huge epoch draw.


class _IdentityDataModule(LightningDataModule):
    def __init__(self, dataset, module, config, augmentations):
        super().__init__()
        index = index_of(dataset)
        labels = {label: i for i, label in enumerate(module.identity["identities"])}
        train = index.where(split="train")
        train = train.take(np.flatnonzero(train.frame.identity_id.notna()))
        validation = index.where(split="val")
        validation = validation.take(np.flatnonzero(validation.frame.identity_id.isin(labels)))
        self.train_data = _Pixels(train, module.identity["preprocessing"], labels, augmentations, training=True, seed=config.seed)
        self.val_data = _Pixels(validation, module.identity["preprocessing"], labels)
        self.config = config
        self.batch_sampler = _DiverseBatches(train, config.batch_size or 64, config.seed)

    def train_dataloader(self):
        # Lightning starts prefetch during setup_data, before on_train_epoch_start.
        # Stamp the sampler before creating the iterator (also after full resume).
        self.batch_sampler.epoch = self.trainer.current_epoch
        self.train_data.epoch = self.trainer.current_epoch
        options = dict(num_workers=self.config.workers, persistent_workers=False)
        if self.config.workers:
            options["prefetch_factor"] = 2
        return DataLoader(self.train_data, batch_sampler=self.batch_sampler, **options)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.config.batch_size or 64, num_workers=self.config.workers)


class _TrainingBridge(ModelCheckpoint):
    """Use Lightning's writer, then forward completed files to the shared session."""
    def __init__(self, result):
        super().__init__(dirpath=result.output_dir / "checkpoints", filename="best-{epoch:03d}",
                         monitor="val_loss", mode="min", save_top_k=1, save_last=True,
                         save_on_train_epoch_end=True)
        self.result = result

    def load_state_dict(self, state_dict):
        # Lightning otherwise drops best-score history when the output folder
        # changes. Copy the previous best into this run before restoring it.
        import shutil
        state = dict(state_dict)
        if state.get("best_model_path"):
            source = self.result.selection.metadata.get("resume_best") or state["best_model_path"]
            source = Path(source)
            if not source.is_file():
                raise ValueError("Full resume requires the previous best checkpoint; use a full-state bundle or preserve the original checkpoint folder")
            self.result.best_epoch = load_checkpoint(source).get("epoch")
            self.result.best_value = float(state["best_model_score"])
            target = Path(self.dirpath) / "previous-best.ckpt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            state.update(dirpath=self.dirpath, best_model_path=str(target), kth_best_model_path=str(target),
                         best_k_models={str(target): state["best_model_score"]}, last_model_path="")
        super().load_state_dict(state)

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.result._emit("train_start", trainer=trainer)

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if not trainer.is_global_zero:
            return
        from .monitoring import monitor_epoch
        metrics = {key.replace("train_", "train/").replace("val_", "val/"): float(value)
                   for key, value in trainer.callback_metrics.items()}
        metrics["optim/lr"] = trainer.optimizers[0].param_groups[0]["lr"]
        metrics["time/epoch_seconds"] = time.monotonic()-self.epoch_started
        started = time.monotonic()
        metrics.update(monitor_epoch(self.result, pl_module, trainer.current_epoch))
        metrics["time/monitor_seconds"] = time.monotonic()-started
        self.result._capture(trainer, Checkpoints(Path(self.best_model_path) if self.best_model_path else None,
            Path(self.last_model_path) if self.last_model_path else None, trainer.current_epoch, "val/loss",
            float(self.best_model_score) if self.best_model_score is not None else None), metrics)
        self.result._emit("epoch_end", trainer=trainer, epoch=trainer.current_epoch, metrics=metrics)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self.result._emit("train_end", trainer=trainer, epoch=trainer.current_epoch)


def select_identity(dataset, *, backbone=None, weights=None, resume=None, config=None):
    from ..model_sources import resolve_model_source
    config = config or TrainingConfig()
    if weights is not None and resume is not None:
        raise ValueError("Use weights for initialization or resume for continuation, not both")
    options = dict(config.backend_options)
    source, meta, provenance = weights if weights is not None else resume, {}, {}
    path, resume_best = None, None
    if source is not None:
        resolved = resolve_model_source(source, progress=False)
        path = resolved.path
        if resolved.options.get("kind") != "id":
            raise ValueError("ID initialization requires an identity checkpoint or a registered custom model")
        if resume is not None:
            latest = (resolved.manifest.get("model") or {}).get("latest_checkpoint")
            if latest:
                resume_best = path
                candidate = path.parent / latest
                if not candidate.is_file():
                    candidates = list(path.parent.parent.rglob(latest))
                    candidate = candidates[0] if len(candidates) == 1 else candidate
                path = candidate
        saved = load_checkpoint(path)
        meta = dict(saved["identity"])
        if resume is not None and not saved.get("optimizer_states"):
            raise ValueError("This checkpoint has no optimizer state; use weights= or publish wandb_contents='full'")
        provenance = dict(source=resolved.source)
        if backbone is not None and backbone_name(backbone) != meta["backbone"]:
            raise ValueError("Backbone override is incompatible with initialization checkpoint")
        for key in ("embedding_dim", "preprocessing"):
            if key in options and options[key] != meta.get(key):
                raise ValueError(f"{key} override is incompatible with initialization checkpoint")
        if config.resolution is not None and config.resolution != meta["preprocessing"]["resolution"]:
            raise ValueError("Resolution override is incompatible with checkpoint preprocessing")
    name = meta.get("backbone") or (backbone_name(backbone) if backbone is not None else "resnet18")
    metadata = {"identity": meta, "resume_best": resume_best, "instance": backbone if backbone is not None and not isinstance(backbone, str) else None}
    return Selection(ModelTypes.ID, "id", name, path if weights is not None else None,
                     path if resume is not None else None, metadata, provenance,
                     meta.get("preprocessing", {}).get("resolution") or config.resolution)


def train_identity(result, prepared, augmentations):
    config, selection = result.config, result.selection
    seed_everything(config.seed, workers=True)
    allowed = {"embedding_dim", "loss", "sub_centers", "freeze_backbone", "lr", "weight_decay", "pretrained", "preprocessing", "trainer",
               "embedding_samples", "embedding_method"}
    unknown = set(config.backend_options)-allowed
    if unknown:
        raise ValueError(f"Unsupported ID backend options: {sorted(unknown)}; Lightning options belong under trainer")
    index = index_of(result.dataset)
    labels = sorted(index.frame.loc[(index.frame.split == "train") & index.frame.identity_id.notna(), "identity_id"].unique())
    saved_meta = selection.metadata.get("identity", {})
    options = dict(config.backend_options)
    identity = {**saved_meta, **{k: v for k, v in options.items() if k not in {"trainer", "pretrained"}},
                "framework": "id", "backbone": selection.name, "identities": labels,
                "sampling": index.sampling, "seed": config.seed, "batch_size": config.batch_size or 64,
                "training_fingerprint": index.fingerprint,
                "training_clips": sorted(index.frame.loc[index.frame.split == "train", "clip_id"].dropna().unique().tolist())}
    module = _IdentityModule(identity, pretrained=options.get("pretrained", True) and selection.weights is None and selection.resume is None,
                             instance=selection.metadata.get("instance"), resolution=config.resolution)
    if config.resolution is not None:
        module.identity["preprocessing"]["resolution"] = config.resolution
    if augmentations is None and selection.resume:
        augmentations = saved_meta.get("augmentations")
    _, serialized = augmentation_pipeline(augmentations, seed=config.seed)
    module.identity["augmentations"] = serialized
    if selection.weights:
        saved = load_checkpoint(selection.weights)
        state = {key: value for key, value in saved["state_dict"].items() if key.startswith(("encoder.", "projection."))}
        incompatible = module.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys or any(not k.startswith("head.") for k in incompatible.missing_keys):
            raise ValueError("Initialization checkpoint is incompatible with encoder/projection")
    if selection.resume:
        for key in ("backbone", "identities", "sampling", "seed", "batch_size", "training_fingerprint", "embedding_dim", "loss", "augmentations", "freeze_backbone", "lr", "weight_decay", "sub_centers"):
            if saved_meta.get(key) != module.identity.get(key):
                raise ValueError(f"Resume requires the same {key}; use weights= for a new run")
    result.metadata.update(identity=module.identity, resolution=module.identity["preprocessing"]["resolution"],
                           training={"loss": module.identity.get("loss", "subcenter_arcface"), "unlabeled_excluded": int(index.frame.identity_id.isna().sum())})
    result._announce(module.identity["preprocessing"]["resolution"])
    from ..training.preview import publish_augmentation_preview
    publish_augmentation_preview(result, serialized)
    data = _IdentityDataModule(result.dataset, module, config, serialized)
    from .monitoring import prepare_monitor
    prepare_monitor(result)
    trainer_options = dict(options.get("trainer", {}))
    managed = {"callbacks", "logger", "enable_checkpointing", "default_root_dir", "max_epochs", "use_distributed_sampler", "reload_dataloaders_every_n_epochs"}
    if managed & trainer_options.keys():
        raise ValueError(f"Lightning options managed by dataset-fixer: {sorted(managed & trainer_options.keys())}")
    device = config.device or "cpu"
    accelerator = "gpu" if device.startswith("cuda") else device
    devices = [int(device.split(":")[1])] if ":" in device else 1
    trainer_options.setdefault("accelerator", accelerator)
    trainer_options.setdefault("devices", devices)
    if trainer_options["devices"] != 1 and trainer_options["devices"] != devices:
        raise ValueError("Identity sampling currently supports one device per run")
    bridge = _TrainingBridge(result)
    trainer = Trainer(max_epochs=config.epochs or 30, default_root_dir=result.output_dir, logger=False,
                      callbacks=[bridge], use_distributed_sampler=False, reload_dataloaders_every_n_epochs=1,
                      **trainer_options)
    trainer.fit(module, datamodule=data, ckpt_path=str(selection.resume) if selection.resume else None)
