"""Unified training with native backend implementations and shared lifecycle."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from ..dataset import Dataset

from .config import (ModelTypes, TrainingConfig, CheckpointConfig, CheckpointProvider,
                     Checkpoints, WandbConfig, TrainingEvent, TrainingCallback)
from .session import TrainingResult, TrainingSession
from .preview import preview_augmentations


def train(dataset: Dataset | str | Path, *, type: ModelTypes | str | None = None, version: int | None = None,
          s: str | None = None, model_type: str | None = None, weights: str | Path | None = None,
          resume: str | Path | None = None, config: TrainingConfig | None = None, augmentations: Any = None,
          checkpointing: CheckpointConfig | CheckpointProvider | None = None, wandb: WandbConfig | bool | None = None,
          callbacks: Sequence[TrainingCallback] = (), session: TrainingSession | None = None) -> TrainingResult:
    """Train using a native backend and return selected weights and evaluation APIs.

    ``model_type=None`` infers the task from Dataset metadata. ``weights`` starts
    a new run; ``resume`` requires optimizer state and continues training. Pass
    a TrainingSession to include later evaluation in the same publication and
    cleanup lifecycle. Without one, training finalizes before returning.

    Parameters:
        dataset: Validated Dataset or a source accepted by Dataset.open().
        type: Native model family; inferred from weights, otherwise YOLO.
        version: Installed YOLO architecture generation, or nnU-Net version 2.
        s: Installed native model size/variant selector.
        model_type: Explicit task override; None infers validated annotations.
        weights: Best weights, bundle, or W&B run initializing a fresh optimizer.
        resume: Full-state checkpoint or bundle; mutually exclusive with weights.
        config: Shared training settings and advanced native backend options.
        augmentations: Backend-native augmentation configuration, shared with preview.
        checkpointing: Checkpoint settings/provider overriding session defaults.
        wandb: W&B settings overriding session defaults; False disables W&B.
        callbacks: Additional handlers receiving shared TrainingEvent objects.
        session: Active managed context owning later evaluation and publication.

    Returns:
        Selected best weights, resumable state, model, and evaluation methods.
    """
    from ..dataset import Dataset
    from .selection import select
    if not isinstance(dataset, Dataset):
        dataset = Dataset.open(dataset)
    dataset.assert_trainable()
    if not {"train", "val"} <= set(dataset.splits):
        raise ValueError("Training requires explicit train and validation splits")
    config = config or TrainingConfig()
    selection = select(dataset, type=type, version=version, s=s, model_type=model_type,
                       weights=weights, resume=resume, config=config)
    if session is None:
        with TrainingSession() as owned:
            return _train(dataset, selection, config, augmentations, checkpointing, wandb, callbacks, owned)
    if not session.active:
        raise ValueError("Use TrainingSession in a with-block")
    return _train(dataset, selection, config, augmentations, checkpointing, wandb, callbacks, session)


def _train(dataset, selection, config, augmentations, checkpointing, wandb, callbacks, session):
    from .backends import ADAPTERS, prepare_data
    result = session._register(dataset, selection, config, checkpointing=checkpointing, wandb=wandb, callbacks=callbacks)
    try:
        prepared = prepare_data(result)
        ADAPTERS[selection.family](result, prepared, augmentations)
        from ..comparison.inference import _clear_accelerator_memory
        _clear_accelerator_memory(config.device)
        if result.best_weights is None:
            raise RuntimeError("Trainer produced no best checkpoint; available recovery weights will still be published")
        return result
    except BaseException as error:
        result._record_error(error)
        try:
            result._emit("error", error=error)
        except Exception:
            pass  # preserve the original training error; session still finalizes
        raise


__all__ = ["train", "ModelTypes", "TrainingConfig", "CheckpointConfig", "CheckpointProvider", "Checkpoints",
           "WandbConfig", "TrainingEvent", "TrainingCallback", "TrainingSession", "TrainingResult", "preview_augmentations"]
