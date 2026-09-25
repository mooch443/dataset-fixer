"""Training lifecycle and confirmed publication, independent of native trainers."""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..bundle import Config, Outcome, create
from ..geometry import Geometry
from ..sources import _atomic_json, cache_root, in_colab, sha256_progress
from ..utils import to_jsonable
from .config import CheckpointConfig, Checkpoints, ModelTypes, TrainingEvent, WandbConfig
from .selection import dataset_schema


def verified_copy(source: Path, destination: Path) -> None:
    """Preserve the last verified backup until the replacement is complete."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".copy-", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if sha256_progress(source, progress=False) != sha256_progress(temporary, progress=False):
            raise OSError(f"Backup verification failed: {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class TrainingResult:
    """Training outputs; normally created by train(), not constructed directly.

    Parameters:
        dataset: Evaluation dataset with the original held-out split.
        selection: Resolved native model and parent provenance.
        config: Effective shared training settings.
        output_dir: Local run directory.
        session: Owning lifecycle manager.
        checkpoint_config: Publication and backup policy.
        checkpoint_provider: Optional custom completed-file selector.
        callbacks: Shared event handlers.
        wandb_run: Explicit run receiving artifacts and metrics.
        owns_run: Whether this result created the W&B run.

    Attributes:
        best_weights: Stable snapshot of the selected model weights.
        resumable_checkpoint: Stable full-state checkpoint.
        bundle: Most recently created model bundle.
        wandb_bundle: W&B publication bundle; optimizer-free best weights by default.
        metrics: Training and evaluation metrics.
        metadata: Native model/training configuration.
        auxiliary_files: Completed report files included in publication.
        checkpoints: Most recently captured checkpoint event.
        error: Original traceback, if training or evaluation failed.
        uploaded_digest: Last confirmed W&B bundle checksum.
        copied_digest: Last verified filesystem backup checksum.
        retained_digest: Last confirmed bundle whose W&B retention was applied.
        finished: Whether checkpoint publication and run finalization completed.
        best_epoch: Epoch at which the selected metric improved.
        best_value: Selected native metric value.
        prepared: Reusable prepared dataset.
        _model: Lazily loaded prediction model.
    """
    dataset: Any
    selection: Any
    config: Any
    output_dir: Path
    session: Any = field(repr=False)
    checkpoint_config: CheckpointConfig = field(default_factory=CheckpointConfig)
    checkpoint_provider: Any = field(default=None, repr=False)
    callbacks: tuple = field(default_factory=tuple, repr=False)
    wandb_run: Any = field(default=None, repr=False)
    owns_run: bool = False
    best_weights: Path | None = None
    resumable_checkpoint: Path | None = None
    bundle: Any = None
    wandb_bundle: Any = None
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    auxiliary_files: dict = field(default_factory=dict)
    checkpoints: Checkpoints = field(default_factory=Checkpoints)
    error: str | None = None
    uploaded_digest: str | None = None
    copied_digest: str | None = None
    retained_digest: str | None = None
    finished: bool = False
    best_epoch: int | None = None
    best_value: float | None = None
    prepared: Any = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)

    def _announce(self, resolution):
        source = self.selection.resume or self.selection.weights or "native pretrained/default"
        print(f"Training {self.selection.name}: task={self.selection.task}, resolution={resolution}, initialization={source}")

    @property
    def retention_pending(self) -> bool:
        """Whether confirmed W&B checkpoints still need old-version cleanup."""
        return bool(self.checkpoint_config.keep_wandb_versions is not None
                    and self.wandb_run is not None and self.wandb_bundle is not None
                    and (self.best_weights is not None or self.resumable_checkpoint is not None)
                    and self.uploaded_digest == self.wandb_bundle.sha256
                    and self.retained_digest != self.wandb_bundle.sha256)

    @property
    def model(self):
        if self.best_weights is None:
            raise RuntimeError("No best checkpoint was produced")
        if self._model is None:
            from ..model import Model
            options = dict(kind=self.selection.family.value, task=self.selection.task,
                           model_type=self.selection.name.removesuffix(".pt"), resolution=self.metadata.get("resolution", self.config.resolution),
                           device=self.config.device, name=self.output_dir.name, workers=max(1, self.config.workers))
            path = self.best_weights
            if self.selection.family == ModelTypes.NNUNET:
                path = self.best_weights.parent.parent
                options.update(folds=(self.best_weights.parent.name.removeprefix("fold_"),), checkpoint=self.best_weights.name)
            self._model = Model(path, **options)
        return self._model

    def _emit(self, name, *, trainer=None, epoch=None, metrics=None, checkpoints=None, error=None):
        event = TrainingEvent(name, trainer, epoch, metrics or {}, checkpoints, error)
        for callback in self.callbacks:
            callback(event)

    def _capture(self, trainer, checkpoints, metrics=None, *, final=False):
        """Called strictly after native writers. Immutable snapshots survive stripping."""
        if self.checkpoint_provider is not None:
            checkpoints = self.checkpoint_provider.checkpoints(trainer, checkpoints)
        paths = {role: Path(path) for role, path in (("best", checkpoints.best), ("latest", checkpoints.latest)) if path is not None}
        frozen = {}
        for role, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint provider returned an incomplete write: {path}")
            digest = sha256_progress(path, progress=False)
            relative = Path(f"fold_{self.metadata.get('fold', 0)}") / path.name if self.selection.family == ModelTypes.NNUNET else Path(path.name)
            snapshot = self.output_dir / "snapshots" / digest / relative
            if not snapshot.exists():
                verified_copy(path, snapshot)
            if self.selection.family == ModelTypes.NNUNET:
                for key in ("plans.json", "dataset.json"):
                    verified_copy(Path(self.metadata["model_folder"]) / key, snapshot.parent.parent / key)
            frozen[role] = snapshot
        self.best_weights = frozen.get("best", self.best_weights)
        if self.best_weights and (self.best_epoch is None or self.best_value != checkpoints.value):
            self.best_epoch, self.best_value = checkpoints.epoch, checkpoints.value
        self.resumable_checkpoint = frozen.get("latest", self.resumable_checkpoint)
        self.checkpoints = replace(checkpoints, best=self.best_weights, latest=self.resumable_checkpoint)
        self._model = None
        self.metrics.update(to_jsonable(metrics or {}))
        if self.wandb_run is not None and metrics:
            try:
                self.wandb_run.log({"epoch": checkpoints.epoch, **to_jsonable(metrics)})
            except Exception as exc:
                warnings.warn(f"W&B metrics logging failed: {exc}")
        self.auxiliary_files.update(checkpoints.files)
        self._emit("checkpoint_saved", trainer=trainer, epoch=checkpoints.epoch, checkpoints=self.checkpoints, metrics=self.metrics)
        if final or (checkpoints.epoch is not None and (checkpoints.epoch + 1) % self.checkpoint_config.every_n_epochs == 0):
            self._publish()

    def evaluate(self, *, samples=32, plots=6, split="val"):
        """Bounded evaluation through the existing comparison and rendering APIs.

        Parameters:
            samples: Maximum number of images in the evaluation cohort.
            plots: Maximum prediction examples; zero disables the example grid.
            split: Held-out dataset split, defaulting to validation.
        """
        try:
            cohort = self.dataset.sample(samples, split=split, seed=self.config.seed)
            result = self.model.compare(cohort, split=split, destination=self.output_dir / "evaluation", save_prediction_plots=False)
            self.metrics["evaluation"] = to_jsonable(result.ranking.to_dict(orient="records"))
            if plots:
                figure = self.output_dir / "predictions.png"
                self.model.visualize(cohort, split=split, samples=plots, destination=figure)
                self.auxiliary_files["evaluation/predictions.png"] = figure
            report = self.output_dir / "evaluation.json"
            _atomic_json(report, self.metrics["evaluation"])
            self.auxiliary_files["evaluation/metrics.json"] = report
            return result
        except BaseException as exc:
            self._record_error(exc)
            raise

    def _record_error(self, error):
        self.error = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        report = self.output_dir / "failure.txt"
        report.write_text(self.error)
        self.auxiliary_files["failure.txt"] = report

    def _bundle_inputs(self, *, resumable=True):
        has_checkpoint = self.best_weights is not None or self.resumable_checkpoint is not None
        files = {**self.auxiliary_files, "training.json": self.output_dir / "training.json"}
        selected = self.best_weights
        latest = self.resumable_checkpoint if resumable else None
        if selected and not resumable:
            from .backends import weights_checkpoint
            digest = sha256_progress(selected, progress=False)
            selected = weights_checkpoint(selected, self.selection.family,
                                          self.output_dir / "upload-weights" / digest / selected.name)
        if self.selection.family == ModelTypes.NNUNET and has_checkpoint:
            # Preserve the official folder layout without duplicating checkpoint bytes.
            chosen = self.best_weights or self.resumable_checkpoint
            fold = chosen.parent.name
            for name in ("plans.json", "dataset.json"):
                files[f"model/{name}"] = chosen.parent.parent / name
            if selected:
                files[f"model/{fold}/{selected.name}"] = selected
            if latest:
                files[f"model/{fold}/{latest.name}"] = latest
            outcome_path = None
        else:
            outcome_path = selected
            if latest and latest != selected:
                files[f"weights/{latest.name}"] = latest
        model_meta = {**self.metadata, "model_name": self.selection.name.removesuffix(".pt"), "dataset_schema": dataset_schema(self.dataset),
                      "checkpoint": (selected or latest).name if has_checkpoint else None, "latest_checkpoint": latest.name if latest else None,
                      "selection_role": "best" if selected else "latest" if latest else "none", "parent": self.selection.provenance}
        bundle_config = Config(name=self.output_dir.name, framework=self.selection.family.value, task=self.selection.task,
            geometry=Geometry.create(input_size=self.metadata.get("resolution", self.config.resolution)),
            dataset=self.prepared or {"dataset_source": self.dataset._source_name, "classes": dataset_schema(self.dataset)["classes"]},
            model=model_meta, training={**self.metadata.get("training", {}), **{k: v for k, v in {"epochs": self.config.epochs, "batch_size": self.config.batch_size, "seed": self.config.seed}.items() if v is not None}},
            run={"id": getattr(self.wandb_run, "id", None)}, files=files)
        outcome = Outcome(checkpoint=outcome_path, metrics=self.metrics, selected_epoch=self.best_epoch,
                          selection_metric=self.checkpoints.metric, selection_value=self.checkpoints.value)
        return bundle_config, outcome

    def _publish(self):
        has_checkpoint = self.best_weights is not None or self.resumable_checkpoint is not None
        if not has_checkpoint and self.error is None:
            return False
        report = self.output_dir / "training.json"
        _atomic_json(report, {"model": self.selection.name, "task": self.selection.task, "metrics": self.metrics,
                              "epoch": self.checkpoints.epoch, "failed": self.error is not None})
        self.bundle_config, outcome = self._bundle_inputs()
        try:
            self.bundle = create(self.bundle_config, outcome, destination=self.output_dir / "bundles", progress=False)
        except Exception as exc:
            warnings.warn(f"Could not bundle checkpoints; local snapshots remain at {self.output_dir}: {exc}")
            return False
        digest = self.bundle.sha256
        if self.checkpoint_config.backup_dir and self.copied_digest != digest:
            try:
                destination = Path(self.checkpoint_config.backup_dir) / self.output_dir.name / "model.zip"
                verified_copy(self.bundle.path, destination)
                _atomic_json(destination.parent / "latest.json", {"bundle": destination.name, "sha256": digest})
                self.copied_digest = digest
            except Exception as exc:
                warnings.warn(f"Checkpoint backup failed: {exc}")
        if self.wandb_run is not None:
            from ..wandb import configure, upload
            previous_upload = self.wandb_bundle
            self.wandb_bundle = None
            try:
                upload_config, upload_outcome = self.bundle_config, outcome
                self.wandb_bundle = self.bundle
                # Until a best checkpoint exists, publish full recovery state.
                if self.checkpoint_config.wandb_contents == "best" and self.best_weights:
                    self.wandb_bundle = None
                    upload_config, upload_outcome = self._bundle_inputs(resumable=False)
                    self.wandb_bundle = create(upload_config, upload_outcome, destination=self.output_dir / "bundles", progress=False)
                if self.uploaded_digest == self.wandb_bundle.sha256:
                    self.wandb_bundle = (previous_upload if previous_upload and previous_upload.sha256 == self.uploaded_digest
                                         else replace(self.wandb_bundle, uploaded=True))
                else:
                    configure(self.wandb_run, upload_config)
                    published = upload(self.wandb_run, self.wandb_bundle, upload_outcome if self.best_weights else None,
                                       artifact=True, timeout=self.checkpoint_config.upload_timeout)
                    if published.uploaded:
                        self.uploaded_digest = published.sha256
                        self.wandb_bundle = published
                        if published.path == self.bundle.path:
                            self.bundle = published
                        if self.best_weights and self.selection.family == ModelTypes.NNUNET:
                            self.wandb_run.summary["best_model_artifact"] = self.wandb_run.summary["checkpoint_artifact"]
            except Exception as exc:
                warnings.warn(f"Checkpoint upload failed: {exc}")
        safe = (has_checkpoint and (self.wandb_run is None or (
                    self.wandb_bundle is not None and self.uploaded_digest == self.wandb_bundle.sha256))
                and (not self.checkpoint_config.backup_dir or self.copied_digest == digest))
        if safe:
            if self.retention_pending:
                from ..wandb import _prune_checkpoint_artifacts
                try:
                    _prune_checkpoint_artifacts(self.wandb_run, sha256=self.uploaded_digest,
                                                keep_versions=self.checkpoint_config.keep_wandb_versions,
                                                timeout=self.checkpoint_config.upload_timeout)
                    self.retained_digest = self.uploaded_digest
                except Exception as exc:
                    warnings.warn(f"W&B checkpoint cleanup failed: {exc}. New checkpoints are safe; "
                                  "cleanup will retry on publication or session.finish().", RuntimeWarning)
            # Keep immutable files during publication, then retire superseded
            # local copies. Long Colab runs must not accumulate two models/epoch.
            retained = {path.relative_to(self.output_dir / "snapshots").parts[0]
                        for path in (self.best_weights, self.resumable_checkpoint) if path}
            for directory in ("snapshots", "upload-weights"):
                for folder in (self.output_dir / directory).glob("*"):
                    if folder.name not in retained:
                        shutil.rmtree(folder)
            bundles = {self.bundle.path}
            if self.wandb_bundle is not None:
                bundles.add(self.wandb_bundle.path)
            for bundle_path in (self.output_dir / "bundles").glob("*.zip"):
                if bundle_path not in bundles:
                    bundle_path.unlink()
        return safe


class TrainingSession:
    """Own publication and finalization around short train/evaluate commands.

    Per-call train() overrides take precedence over session defaults. A session
    keeps unsuccessful publications retryable with finish(), including after
    leaving its context. Disconnect requires a configured persistent destination.

    Parameters:
        checkpointing: Default CheckpointConfig or custom CheckpointProvider.
        wandb: Default WandbConfig; None/False disables managed W&B logging.
        callbacks: Shared event handlers supplemented by per-train callbacks.
        disconnect: False, or 'after_safe' to disconnect Colab after publication.
        disconnect_delay: Seconds to leave Colab connected after safety checks.
    """
    def __init__(self, *, checkpointing=None, wandb=None, callbacks=(), disconnect=False, disconnect_delay=30):
        if disconnect not in {False, "after_safe"}:
            raise ValueError("disconnect must be False or 'after_safe'")
        if disconnect_delay < 0:
            raise ValueError("disconnect_delay cannot be negative")
        self.checkpointing = checkpointing
        self.wandb = wandb
        self.callbacks = tuple(callbacks)
        self.disconnect = disconnect
        self.disconnect_delay = disconnect_delay
        self.results = []
        self.active = False
        self.disconnected = False

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, exc_type, error, tb):
        if error is not None:
            for result in self.results:
                result._record_error(error)
        try:
            self.finish()
        except Exception as failure:
            if error is None:
                raise
            warnings.warn(f"Final publication is incomplete; runtime kept connected: {failure}")
        finally:
            self.active = False
        return False

    def _register(self, dataset, selection, config, *, checkpointing=None, wandb=None, callbacks=()):
        import uuid
        chosen = checkpointing if checkpointing is not None else self.checkpointing
        provider = None
        if chosen is False:
            raise ValueError("Training requires checkpoints; supply a CheckpointProvider to customize them")
        if chosen is None:
            settings = CheckpointConfig()
        elif isinstance(chosen, CheckpointConfig):
            settings = chosen
        else:
            provider, settings = chosen, chosen.config
        output = Path(config.output_dir) if config.output_dir else cache_root() / "training" / f"{selection.family.name.lower()}-{uuid.uuid4().hex[:10]}"
        output = output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        selected_wandb = wandb if wandb is not None else self.wandb
        run, owns = None, False
        if selected_wandb not in (None, False):
            if not isinstance(selected_wandb, WandbConfig):
                raise TypeError("wandb must be WandbConfig or False")
            import wandb as sdk
            run = selected_wandb.run
            if run is None:
                if sdk.run is not None:
                    raise ValueError("An active W&B run exists; pass WandbConfig(run=wandb.run) to adopt it explicitly")
                options = dict(selected_wandb.init)
                options.update(project=selected_wandb.project, entity=selected_wandb.entity,
                               name=selected_wandb.name or output.name, dir=str(output))
                options.setdefault("mode", "online")
                run, owns = sdk.init(**options), True
        result = TrainingResult(dataset, selection, config, output, self, settings, provider,
                                (*self.callbacks, *callbacks), run, owns)
        self.results.append(result)
        if settings.backup_dir:
            folder = Path(settings.backup_dir)
            if Path("/content/drive") in folder.parents and not Path("/content/drive/MyDrive").is_dir():
                raise RuntimeError("Mount Google Drive before enabling a Drive checkpoint backup")
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=folder) as probe:
                probe.write(b"dataset-fixer backup probe")
                probe.flush()
                os.fsync(probe.fileno())
        if self.disconnect and run is None and not settings.backup_dir:
            raise ValueError("Safe disconnect needs W&B or a persistent backup destination")
        return result

    def finish(self):
        """Retry pending publications; only disconnect once every run is safe."""
        all_safe = bool(self.results)
        failures = []
        for result in self.results:
            if result.finished and not result.retention_pending:
                continue
            safe = False
            for attempt in range(result.checkpoint_config.final_attempts):
                try:
                    safe = result._publish()
                except Exception as failure:
                    warnings.warn(f"Publication failed for {result.output_dir}: {failure}")
                if (safe and not result.retention_pending) or (result.best_weights is None and result.resumable_checkpoint is None):
                    break
            all_safe = all_safe and safe
            if safe:
                try:
                    if not result.finished and result.wandb_run is not None and (result.owns_run or self.disconnect):
                        result.wandb_run.finish(exit_code=1 if result.error else 0)
                    result.finished = True
                    result._model = None
                except Exception as failure:
                    all_safe = False
                    failures.append(f"W&B finalization failed for {result.output_dir}: {failure}")
            elif result.best_weights is not None or result.resumable_checkpoint is not None:
                failures.append(f"Snapshots retained at {result.output_dir}")
            elif result.wandb_run is not None and result.owns_run:
                try:
                    result.wandb_run.finish(exit_code=1)
                except Exception as failure:
                    failures.append(f"W&B finalization failed for {result.output_dir}: {failure}")
        if failures:
            raise RuntimeError("Publication incomplete; " + "; ".join(failures) + ". Retry session.finish().")
        if self.disconnect and all_safe and not self.disconnected and in_colab():
            from google.colab import runtime
            print(f"Checkpoints confirmed safe. Disconnecting in {self.disconnect_delay} seconds.")
            time.sleep(self.disconnect_delay)
            runtime.unassign()
            self.disconnected = True
        return all_safe
