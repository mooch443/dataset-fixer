"""Adapters delegate optimization and checkpoint selection to native trainers."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..convert import Kind, prepare
from .config import Checkpoints, ModelTypes
from .selection import dataset_schema


def native_options(config, aliases):
    options = dict(config.backend_options)
    for public, native in aliases.items():
        value = getattr(config, public)
        if value is not None:
            if native in options and options[native] != value:
                raise ValueError(f"Conflicting {public} and backend_options[{native!r}]")
            options[native] = value
    return options


def prepare_data(result):
    selection, config = result.selection, result.config
    kind = Kind.YOLO_SEM if selection.task == "semantic_segment" else Kind.YOLO
    if selection.family == ModelTypes.RFDETR:
        kind = Kind.RFDETR
    if selection.family == ModelTypes.NNUNET:
        kind = Kind.NNUNET
    options = dict(workers=max(1, config.workers))
    if kind == Kind.NNUNET:
        planner = dict(selection.metadata.get("model") or {}).get("planner", selection.name)
        options.update(planner=config.backend_options.get("planner", planner), preprocess=True)
        # nnU-Net resolution is a preprocessing geometry, distinct from its native patch size.
        if config.resolution:
            options["input_size"] = config.resolution
    dataset = result.dataset
    if kind in {Kind.NNUNET, Kind.YOLO_SEM} and dataset.format != "semantic_masks":
        dataset = dataset.export(destination=result.output_dir / "semantic-data", format="semantic_masks", visualize=False)
        result.dataset = dataset
    prepared = prepare(dataset, kind, **options)
    result.prepared = prepared
    if prepared.data_yaml:
        result.auxiliary_files["data.yaml"] = prepared.data_yaml
    result.auxiliary_files["preparation.json"] = prepared.manifest
    return prepared


def _existing(path):
    return path if path.is_file() else None


def train_yolo(result, prepared, augmentations):
    from ultralytics import YOLO
    selection, config = result.selection, result.config
    options = native_options(config, {"resolution": "imgsz", "epochs": "epochs", "batch_size": "batch", "device": "device", "workers": "workers", "seed": "seed"})
    callbacks = options.pop("native_callbacks", {})
    trainer_class = options.pop("trainer", None)
    for forbidden in ("data", "project", "name", "exist_ok", "resume", "save", "save_dir"):
        if forbidden in options:
            raise ValueError(f"{forbidden} is managed by dataset-fixer; use its public settings")
    native = YOLO(str(selection.resume or selection.weights or selection.name))
    from ..model import _normalize_prediction_task
    if _normalize_prediction_task(native.task) != selection.task:
        raise ValueError(f"Installed Ultralytics resolved task {native.task!r}, incompatible with {selection.task!r}")
    if augmentations is not None:
        from .preview import yolo_augmentations
        overrides = yolo_augmentations(augmentations)
        if set(overrides) & {"imgsz", "epochs", "batch", "device", "data", "project", "name", "save", "resume", "save_dir"}:
            raise ValueError("Put training settings in TrainingConfig, not the augmentation configuration")
        options.update(overrides)
    resolution = options.get("imgsz", native.overrides.get("imgsz", 640))
    stride = max(32, int(native.model.stride.max()))
    if not isinstance(resolution, int) or resolution < stride or resolution % stride:
        raise ValueError(f"YOLO resolution {resolution} must be a positive multiple of the model stride ({stride}); it will not be rounded")
    options["imgsz"] = resolution
    if selection.resume and resolution != native.overrides.get("imgsz", resolution):
        raise ValueError("Change resolution with weights=, not full-state resume")
    result._announce(resolution)
    def remove_native_wandb(trainer):
        # Native W&B integration finishes the shared run at train end, before
        # evaluation/publication. Keep ownership in TrainingSession instead.
        for functions in trainer.callbacks.values():
            functions[:] = [f for f in functions if f.__module__ != "ultralytics.utils.callbacks.wb"]
        if selection.resume and config.epochs is not None:
            # Native check_resume only accepts a limited override list. Explicit
            # epochs means the new total; all optimizer/scheduler state resumes.
            trainer.args.epochs = trainer.epochs = config.epochs
    native.add_callback("on_pretrain_routine_start", remove_native_wandb)
    def saved(trainer):
        if getattr(trainer, "rank", int(os.environ.get("RANK", "-1"))) not in {-1, 0}:
            return
        result.metadata["resolution"] = int(trainer.args.imgsz)
        result.metadata["training"] = vars(trainer.args)
        result._capture(trainer, Checkpoints(_existing(Path(trainer.best)), _existing(Path(trainer.last)),
                                           trainer.epoch, "fitness", float(trainer.best_fitness)), getattr(trainer, "metrics", {}))
    def epoch(trainer):
        result._emit("epoch_end", trainer=trainer, epoch=trainer.epoch, metrics=getattr(trainer, "metrics", {}))
    native.add_callback("on_model_save", saved)
    native.add_callback("on_fit_epoch_end", epoch)
    native.add_callback("on_train_start", lambda trainer: result._emit("train_start", trainer=trainer))
    for event, functions in callbacks.items():
        if event not in native.callbacks:
            raise ValueError(f"Unknown Ultralytics callback event: {event}")
        for function in functions if isinstance(functions, (list, tuple)) else [functions]:
            native.add_callback(event, function)
    # Prevent native integrations from creating a second run; metrics go to the adopted run.
    native.train(data=str(prepared.data_yaml), project=str(result.output_dir), name="native", exist_ok=False,
                 save_dir=str(result.output_dir / "native"), resume=str(selection.resume) if selection.resume else False,
                 save=True, trainer=trainer_class, **options)
    result._emit("train_end", trainer=native.trainer)


def rfdetr_configs(selection, config, prepared, augmentations):
    import rfdetr
    from ..rfdetr_engine import checkpoint_metadata, resolution_overrides
    from rfdetr.datasets import infer_yolo_keypoint_schema
    if selection.weights or selection.resume:
        metadata = checkpoint_metadata(selection.weights or selection.resume)
        variant = getattr(rfdetr, metadata["model_name"])
        model_values = dict(metadata["model_config"])
        model_values.update(resolution_overrides(model_values, config.resolution))
        model_values["pretrain_weights"] = str(selection.weights or selection.resume)
    else:
        variant = getattr(rfdetr, selection.name)
        model_values = variant._model_config_class().model_dump()
        model_values.update(resolution_overrides(model_values, config.resolution))
    options = native_options(config, {"epochs": "epochs", "batch_size": "batch_size", "workers": "num_workers", "seed": "seed"})
    if selection.resume:
        saved_options = {key: value for key, value in metadata["training_config"].items()
                         if key in variant._train_config_class.model_fields
                         and key not in {"dataset_dir", "output_dir", "dataset_file", "wandb", "resume"}}
        options = {**saved_options, **options}
    model_options = options.pop("model", {})
    if config.resolution is not None and model_options.get("resolution", config.resolution) != config.resolution:
        raise ValueError("Conflicting resolution and backend_options['model']['resolution']")
    if (selection.weights or selection.resume) and "pretrain_weights" in model_options:
        raise ValueError("Use weights= to select RF-DETR initialization")
    if selection.weights or selection.resume:
        for key, value in model_options.items():
            if key not in {"device", "resolution", "positional_encoding_size"} and model_values.get(key) != value:
                raise ValueError(f"RF-DETR checkpoint architecture conflicts with model option {key!r}")
    if selection.resume and model_options.get("resolution", model_values["resolution"]) != model_values["resolution"]:
        raise ValueError("Change resolution with weights=, not full-state resume")
    model_values.update(resolution_overrides(model_values, model_options.get("resolution", model_values["resolution"])))
    model_values.update(model_options)
    resolution_overrides(model_values, model_values["resolution"])
    model_values["model_name"] = variant.__name__
    if config.device:
        import torch
        device = torch.device(config.device)
        model_values["device"] = config.device
        options["accelerator"] = "gpu" if device.type == "cuda" else device.type
        if device.index is not None:
            options["devices"] = [device.index]
    options.setdefault("devices", 1)
    options.setdefault("checkpoint_interval", (config.epochs or 100) + 1)
    options.setdefault("skip_best_epochs", 0)
    options.setdefault("smooth_alpha", 0.0)
    options.setdefault("multi_scale", False)
    options.setdefault("expanded_scales", False)
    options.setdefault("do_random_resize_via_padding", False)
    options.setdefault("use_ema", False)
    if selection.task == "pose":
        schema = infer_yolo_keypoint_schema(prepared.data_yaml)
        model_values.update(num_classes=len(schema.class_names), num_keypoints_per_class=schema.num_keypoints_per_class)
        options.update(class_names=schema.class_names, keypoint_flip_pairs=schema.keypoint_flip_pairs,
                       keypoint_oks_sigmas=schema.keypoint_oks_sigmas, augmentation_backend="cpu")
    else:
        import yaml
        names = yaml.safe_load(prepared.data_yaml.read_text())["names"]
        options["class_names"] = list(names.values()) if isinstance(names, dict) else names
        model_values["num_classes"] = len(names)
    if augmentations is not None:
        options["aug_config"] = augmentations
    if (selection.weights or selection.resume) and metadata["model_config"].get("num_classes") != model_values["num_classes"]:
        raise ValueError("RF-DETR checkpoint class count differs from the dataset; implicit head replacement is disabled")
    mc = variant._model_config_class(**model_values)
    return variant, mc, options


def train_rfdetr(result, prepared, augmentations):
    from rfdetr import RFDETRDataModule, RFDETRModelModule, build_trainer
    from rfdetr.models.weights import load_pretrain_weights
    from rfdetr.detr import RFDETR
    from pytorch_lightning.callbacks import Checkpoint, ModelCheckpoint
    from types import SimpleNamespace
    from rfdetr.utilities.reproducibility import seed_all
    seed_all(result.config.seed)
    variant, mc, options = rfdetr_configs(result.selection, result.config, prepared, augmentations)
    native_callbacks = options.pop("native_callbacks", [])
    trainer_factory = options.pop("trainer", None)
    trainer_options = options.pop("trainer_options", {})
    for forbidden in ("dataset_dir", "output_dir", "dataset_file", "wandb", "resume"):
        if forbidden in options:
            raise ValueError(f"{forbidden} is managed by dataset-fixer")
    options.setdefault("tensorboard", False)
    options.setdefault("run_test", False)
    tc = variant._train_config_class(dataset_dir=str(prepared.location), dataset_file="yolo", output_dir=str(result.output_dir / "native"),
                                     wandb=False, **options)
    result._announce(mc.resolution)
    result.metadata.update(model_name=variant.__name__, model_config=mc.model_dump(mode="json"), resolution=mc.resolution,
                           training=tc.model_dump(mode="json"))
    # Download through the native resolver without allocating an inference model.
    RFDETR.maybe_download_pretrain_weights(SimpleNamespace(model_config=mc))
    module = RFDETRModelModule(mc, tc)
    module.strict_loading = True
    if result.selection.weights:
        # Native pose initialization resets Gaussian parameters. Reload through
        # its canonical PE interpolation/remapping, rejecting every partial load.
        def require_all_weights(model, incompatible):
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(f"Trained RF-DETR layers are incompatible: {incompatible}")
        handle = module.model.register_load_state_dict_post_hook(require_all_weights)
        try:
            load_pretrain_weights(module.model, mc)
        finally:
            handle.remove()
    data = RFDETRDataModule(mc, tc)
    trainer = build_trainer(tc, mc, **trainer_options)
    if trainer_factory is not None:
        required_callbacks = list(trainer.callbacks)
        trainer = trainer_factory(trainer)
        for callback in required_callbacks:
            if callback not in trainer.callbacks:
                trainer.callbacks.append(callback)
    class Publish(Checkpoint):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            checkpoint.update(model_config=mc.model_dump(mode="json"), model_name=variant.__name__,
                              dataset_schema=dataset_schema(result.dataset), args=tc.model_dump(mode="json"))

        def on_train_start(self, trainer, pl_module):
            if trainer.is_global_zero:
                result._emit("train_start", trainer=trainer)

        def on_train_epoch_end(self, trainer, pl_module):
            if trainer.is_global_zero:
                self.save(trainer)
                result._emit("epoch_end", trainer=trainer, epoch=trainer.current_epoch, metrics=result.metrics)

        def on_fit_end(self, trainer, pl_module):
            if trainer.is_global_zero:
                self.save(trainer, final=True)
                result._emit("train_end", trainer=trainer)

        def save(self, trainer, final=False):
            root = Path(tc.output_dir)
            best = _existing(root / "checkpoint_best_total.pth") if final else None
            native_best = next((c for c in trainer.callbacks if hasattr(c, "_best_ema")), None)
            regular_score = float(getattr(native_best, "best_model_score", 0) or 0)
            if getattr(native_best, "_smooth_alpha", 0) > 0:
                regular_score = float(native_best._best_raw_regular)
            ema_score = float(getattr(native_best, "_best_ema", 0))
            use_ema = ema_score > regular_score and (root / "checkpoint_best_ema.pth").is_file()
            if best is None and native_best is not None:
                best = _existing(root / ("checkpoint_best_ema.pth" if use_ema else "checkpoint_best_regular.pth"))
            best = best or _existing(root / "checkpoint_best_regular.pth")
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if hasattr(v, "numel") and v.numel() == 1}
            metric = {"pose": "val/keypoint_map_50_95", "segment": "val/segm_mAP_50_95", "detect": "val/mAP_50_95"}[result.selection.task]
            result._capture(trainer, Checkpoints(best, _existing(root / "last.ckpt"), trainer.current_epoch, metric,
                                               ema_score if use_ema else regular_score), metrics, final=final)
    # Always keep one native full-state writer after the custom best-weight
    # writer. RF-DETR omits its last.ckpt callback when checkpoint_interval=1.
    # Saving at train-epoch end also captures the updated best-callback state.
    class LatestCheckpoint(ModelCheckpoint):
        pass
    trainer.callbacks = [c for c in trainer.callbacks if not (isinstance(c, ModelCheckpoint) and c.filename == "last")]
    latest = LatestCheckpoint(dirpath=tc.output_dir, filename="last", every_n_epochs=1,
                             save_top_k=1, save_on_train_epoch_end=True, enable_version_counter=False,
                             auto_insert_metric_name=False)
    trainer.callbacks.extend([*native_callbacks, latest, Publish()])
    trainer.fit(module, datamodule=data, ckpt_path=str(result.selection.resume) if result.selection.resume else None)


def train_nnunet(result, prepared, augmentations, *, preview=None):
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    config, selection = result.config, result.selection
    options = native_options(config, {"epochs": "num_epochs", "batch_size": "batch_size"})
    if selection.resume:
        saved_options = dict(selection.metadata.get("model", {}).get("training", {}))
        if "epochs" in saved_options:
            saved_options["num_epochs"] = saved_options.pop("epochs")
        options = {**saved_options, **options}
    for forbidden in ("disable_checkpointing", "output_folder", "output_folder_base", "log_file", "current_epoch", "local_rank", "device"):
        if forbidden in options:
            raise ValueError(f"{forbidden} is managed by dataset-fixer; use its public settings or a CheckpointProvider")
    base = options.pop("trainer", nnUNetTrainer)
    native_callbacks = options.pop("native_callbacks", {})
    if set(native_callbacks) - {"on_train_start", "on_epoch_end", "on_train_end"}:
        raise ValueError("nnU-Net native_callbacks supports on_train_start, on_epoch_end, and on_train_end; use a trainer subclass for other hooks")
    if augmentations is not None and not callable(augmentations):
        raise TypeError("nnU-Net augmentations must be a callable customizing its native transform pipeline")
    fold = options.pop("fold", 0)
    if fold != 0:
        raise ValueError("Dataset splits are preserved as nnU-Net fold 0; other folds would resplit the supplied validation set")
    configuration = options.pop("configuration", "2d")
    plans_name = options.pop("plans", None)
    options.pop("planner", None)
    environment = dict(prepared.backend["environment"])
    environment["nnUNet_wandb_enabled"] = "0"
    environment["nnUNet_n_proc_DA"] = str(config.workers)
    preprocessed = Path(environment["nnUNet_preprocessed"]) / prepared.backend["dataset_name"]
    plans_files = [preprocessed / f"{plans_name}.json"] if plans_name else list(preprocessed.glob("*Plans*.json"))
    if len(plans_files) != 1:
        raise ValueError("Select one nnU-Net plans identifier through backend_options['plans']")
    plans = json.loads(plans_files[0].read_text())
    if "batch_size" in options:
        plans["configurations"][configuration]["batch_size"] = options.pop("batch_size")
    plans["continue_training"] = bool(selection.resume)
    dataset_json = json.loads(prepared.paths["dataset_json"].read_text())
    if selection.weights or selection.resume:
        parent = Path(selection.metadata["model_folder"])
        previous = json.loads((parent / "plans.json").read_text())
        if previous["configurations"][configuration]["architecture"] != plans["configurations"][configuration]["architecture"]:
            raise ValueError("nnU-Net plans changed the network architecture; these weights cannot initialize it")
        previous_dataset = json.loads((parent / "dataset.json").read_text())
        if previous_dataset["labels"] != dataset_json["labels"] or previous_dataset["channel_names"] != dataset_json["channel_names"]:
            raise ValueError("nnU-Net class/channel schema differs from the dataset")
    class ManagedTrainer(base):
        def get_training_transforms(self, *args, **kwargs):
            transforms = super().get_training_transforms(*args, **kwargs)
            return augmentations(transforms) if augmentations is not None else transforms

        def on_train_start(self):
            super().on_train_start()
            result._emit("train_start", trainer=self)
            for function in native_callbacks.get("on_train_start", ()):
                function(self)

        def on_train_end(self):
            super().on_train_end()
            for function in native_callbacks.get("on_train_end", ()):
                function(self)

        def on_epoch_end(self):
            epoch = self.current_epoch
            super().on_epoch_end()
            # Last epoch also needs resumable state before native cleanup removes latest.
            self.current_epoch = epoch
            try:
                self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_latest.pth"))
            finally:
                self.current_epoch = epoch + 1
            if self.local_rank == 0:
                root = Path(self.output_folder)
                metrics = {"val/dice_ema": float(self._best_ema)}
                result._capture(self, Checkpoints(_existing(root / "checkpoint_best.pth"), _existing(root / "checkpoint_latest.pth"), epoch,
                                                "val/dice_ema", float(self._best_ema)), metrics)
                result._emit("epoch_end", trainer=self, epoch=epoch, metrics=metrics)
            for function in native_callbacks.get("on_epoch_end", ()):
                function(self)

    # The checkpoint's inference architecture is the user's native trainer;
    # the lifecycle wrapper itself must never leak into portable checkpoints.
    ManagedTrainer.__name__ = base.__name__

    # Native trainer resolves module-level paths, so assign instance folders immediately after construction.
    old_environment = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        import random
        import numpy as np
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        trainer = ManagedTrainer(plans, configuration, fold, dataset_json, device=torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu")))
        trainer.preprocessed_dataset_folder_base = str(preprocessed)
        trainer.preprocessed_dataset_folder = str(preprocessed / trainer.configuration_manager.data_identifier)
        trainer.output_folder_base = str(result.output_dir / "native")
        trainer.output_folder = str(Path(trainer.output_folder_base) / f"fold_{fold}")
        Path(trainer.output_folder).mkdir(parents=True, exist_ok=True)
        trainer.log_file = str(Path(trainer.output_folder) / "training.log")
        trainer.num_epochs = config.epochs or trainer.num_epochs
        trainer.save_every = 1
        for key, value in options.items():
            if not hasattr(trainer, key) or key.startswith("_"):
                raise ValueError(f"Unknown nnU-Net trainer option: {key}")
            setattr(trainer, key, value)
        result.metadata.update(model_folder=trainer.output_folder_base, fold=fold, folds=[fold], configuration=configuration,
                               resolution=config.resolution, planner=prepared.backend["planner"],
                               training={**options, "epochs": trainer.num_epochs, "batch_size": trainer.configuration_manager.batch_size, "initial_lr": trainer.initial_lr})
        if preview is not None:
            trainer._set_batch_size_and_oversample()
            return preview(trainer)
        result._announce(config.resolution or tuple(trainer.configuration_manager.patch_size))
        if selection.resume:
            trainer.load_checkpoint(str(selection.resume))
        elif selection.weights:
            trainer.initialize()
            state = torch.load(selection.weights, map_location="cpu", weights_only=False)
            raw = getattr(trainer.network, "_orig_mod", trainer.network)
            raw.load_state_dict(state["network_weights"], strict=True)
        trainer.run_training()
        result._emit("train_end", trainer=trainer)
    finally:
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


ADAPTERS = {ModelTypes.YOLO: train_yolo, ModelTypes.RFDETR: train_rfdetr, ModelTypes.NNUNET: train_nnunet}
