"""Small configuration objects shared by native training adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


class ModelTypes(str, Enum):
    YOLO = "ultralytics"
    RFDETR = "rfdetr"
    NNUNET = "nnunet"


@dataclass(frozen=True)
class TrainingConfig:
    """None-valued training options use native defaults; backend_options is the escape hatch.

    backend_options accepts native training arguments plus ``native_callbacks``
    and ``trainer`` (a native trainer class). Common options take precedence;
    conflicting native aliases are rejected rather than silently overridden.

    Parameters:
        resolution: Exact square input resolution; invalid sizes are rejected.
        epochs: Total epochs, including completed epochs for explicit resume.
        batch_size: Native batch size.
        device: Native device name, such as cpu or cuda:0.
        workers: Data-loader/preprocessing worker count; zero loads synchronously.
        seed: Reproducibility seed.
        output_dir: New run directory; defaults to a unique local cache folder.
        backend_options: Native options, callbacks, and custom trainer settings.
    """
    resolution: int | None = None
    epochs: int | None = None
    batch_size: int | None = None
    device: str | None = None
    workers: int = 4
    seed: int = 7
    output_dir: str | Path | None = None
    backend_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("resolution", "epochs", "batch_size"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 0:
            raise ValueError("workers must be a non-negative integer")


@dataclass(frozen=True)
class CheckpointConfig:
    """Keep best/latest locally; publish each completed epoch by default.

    Parameters:
        backup_dir: Optional persistent filesystem destination, such as Drive.
        every_n_epochs: Publish after this many completed epochs.
        upload_timeout: Seconds to await confirmed W&B artifact completion.
        final_attempts: Publication attempts during each finish() call.
    """
    backup_dir: str | Path | None = None
    every_n_epochs: int = 1
    upload_timeout: int = 600
    final_attempts: int = 3

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (self.every_n_epochs, self.upload_timeout, self.final_attempts)):
            raise ValueError("Checkpoint intervals, timeouts, and attempts must be positive")


@dataclass(frozen=True)
class WandbConfig:
    """Use an existing run or create an online run; login stays with the caller.

    Parameters:
        project: Project for a new run.
        entity: Optional W&B team/account.
        name: Optional run display name.
        run: Existing run to adopt explicitly, instead of creating one.
        init: Additional wandb.init() options.
    """
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    run: Any = None
    init: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Checkpoints:
    """Only completed checkpoint writes may be supplied to the publisher.

    Parameters:
        best: Selected model weights.
        latest: Full optimizer/scheduler state for continuation.
        epoch: Zero-based last completed epoch.
        metric: Native selection metric name.
        value: Native best selection score.
        files: Additional completed files keyed by archive-relative names.
    """
    best: Path | None = None
    latest: Path | None = None
    epoch: int | None = None
    metric: str | None = None
    value: float | None = None
    files: Mapping[str, Path] = field(default_factory=dict)


class CheckpointProvider(Protocol):
    """Override native selection by returning completed native checkpoint files."""
    config: CheckpointConfig

    def checkpoints(self, trainer: Any, defaults: Checkpoints) -> Checkpoints:
        """Select completed files.

        Parameters:
            trainer: Native backend trainer.
            defaults: Files selected by the native checkpoint writer.
        """
        ...


@dataclass(frozen=True)
class TrainingEvent:
    """Shared callback event. ``trainer`` also exposes backend-native controls.

    Parameters:
        name: train_start, epoch_end, checkpoint_saved, train_end, or error.
        trainer: Native trainer when available.
        epoch: Zero-based epoch when available.
        metrics: Native metric values.
        checkpoints: Completed immutable checkpoint snapshots, when available.
        error: Original exception for an error event.
    """
    name: str
    trainer: Any = None
    epoch: int | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    checkpoints: Checkpoints | None = None
    error: BaseException | None = None


TrainingCallback = Callable[[TrainingEvent], None]
