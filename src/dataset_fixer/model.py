from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .errors import (
    DatasetValidationError,
    PredictionCacheMissError,
    PredictionScoreUnavailableError,
    ValidationIssue,
)
from .geometry import Geometry, filter_inputs_by_size, normalize_errors
from .prediction_cache import PredictionCache
from .utils import IMAGE_SUFFIXES, sha256_file, slugify, to_jsonable

ModelKind = Literal["ultralytics", "nnunet"]
PredictionTask = Literal["detect", "segment", "pose", "polo", "semantic_segment"]
ModelTask = PredictionTask | Literal["auto", "locate", "semantic"]
_MAX_INFERENCE_BATCH_SIZE = 128


def _default_nnunet_device() -> Literal["cpu", "cuda", "mps"]:
    """Select the best device exposed by the current PyTorch runtime."""

    try:
        import torch
    except ImportError:
        # The official nnU-Net executable may live in a separate environment.
        # CPU is the only safe assumption when this process cannot inspect it.
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class ModelInput:
    """One image supplied to :meth:`Model.predict`.

    Parameters:
        image_id: Identifier stable within one prediction request.
        image_path: Resolved input image path.
        width: Input image width in pixels.
        height: Input image height in pixels.
        relative_path: Cohort-relative input image path.
        mask_path: Optional ground-truth mask used only by evaluation code;
            prediction never reads it.
        image_sha256: Optional already-computed image digest used to avoid
            hashing frozen comparison inputs again for prediction caching.
    """

    image_id: str
    image_path: Path
    width: int
    height: int
    relative_path: str
    mask_path: Path | None = None
    image_sha256: str | None = None


@dataclass(frozen=True)
class ImagePrediction:
    """Predictions associated with one input image.

    Parameters:
        image_id: Identifier copied from the corresponding model input.
        image_path: Resolved input image path.
        relative_path: Cohort-relative input image path.
        width: Output-space image width in pixels.
        height: Output-space image height in pixels.
        objects: Object predictions for instance-style tasks.
        mask: Projected semantic prediction in output-image space.
        native_mask: Optional prediction in native adapter output space.
        foreground_probability: Optional canonical-space foreground
            probability map. Unlike ``mask``, this is independent of the
            selected semantic operating threshold and can therefore be reused
            for calibration without rerunning inference.
        metadata: Backend-specific non-tensor prediction metadata.
    """

    image_id: str
    image_path: Path
    relative_path: str
    width: int
    height: int
    objects: tuple[Any, ...] = ()
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    native_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    foreground_probability: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of object predictions, or foreground pixels for a mask."""

        if self.mask is not None:
            return int(np.count_nonzero(self.mask))
        return len(self.objects)

    def foreground_score_map(self) -> np.ndarray | None:
        """Return reusable semantic scores or rasterized instance scores.

        Semantic records return their foreground-probability map. Instance
        segmentation records are rasterized lazily without changing or
        discarding their object polygons: each covered pixel receives the
        maximum score of any covering instance. This derived field is an
        object-confidence projection, not a calibrated per-pixel posterior.

        Returns:
            A float32 ``(height, width)`` score map, or ``None`` when neither
            semantic probabilities nor segmentation polygons are available.
        """

        if self.foreground_probability is not None:
            return np.asarray(self.foreground_probability, dtype=np.float32)
        polygon_predictions = [
            prediction
            for prediction in self.objects
            if prediction.polygons or prediction.polygon is not None
        ]
        if not polygon_predictions and self.objects:
            return None
        canvas = Image.new("F", (self.width, self.height), 0.0)
        draw = ImageDraw.Draw(canvas)
        # Painting low-to-high makes overlaps equal the maximum object score.
        for prediction in sorted(
            polygon_predictions,
            key=lambda value: float(value.score),
        ):
            polygons = prediction.polygons or (
                [prediction.polygon] if prediction.polygon is not None else []
            )
            for polygon in polygons:
                if len(polygon) < 3 or any(
                    not math.isfinite(float(x)) or not math.isfinite(float(y))
                    for x, y in polygon
                ):
                    continue
                draw.polygon(
                    [(float(x), float(y)) for x, y in polygon],
                    fill=float(prediction.score),
                )
        return np.asarray(canvas, dtype=np.float32)


@dataclass(frozen=True)
class PredictionResult:
    """Ordered, model-independent output returned by :meth:`Model.predict`.

    Parameters:
        model_name: Resolved display name of the predicting model.
        model_kind: Adapter family used for inference.
        task: Normalized prediction task.
        backend: Concrete inference backend identifier.
        records: Predictions in the same order as the model inputs.
        inference_seconds: Measured prediction wall time.
        settings: Effective device, batching, and inference configuration.
        cache_info: Verified prediction-cache status and location, when used.
    """

    model_name: str
    model_kind: ModelKind
    task: PredictionTask
    backend: str
    records: tuple[ImagePrediction, ...]
    inference_seconds: float
    settings: dict[str, Any] = field(default_factory=dict)
    cache_info: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ImagePrediction]:
        return iter(self.records)

    def __getitem__(self, value: int | str) -> ImagePrediction:
        if isinstance(value, int):
            return self.records[value]
        try:
            return self.by_id[value]
        except KeyError as exc:
            raise KeyError(f"Unknown prediction image_id {value!r}") from exc

    @property
    def by_id(self) -> dict[str, ImagePrediction]:
        """Predictions keyed by stable input image identifier."""

        return {record.image_id: record for record in self.records}

    @property
    def masks(self) -> dict[str, np.ndarray]:
        """Semantic masks keyed by image identifier."""

        return {
            record.image_id: record.mask
            for record in self.records
            if record.mask is not None
        }

    def save(
        self,
        destination: str | Path,
        *,
        include_probabilities: bool = False,
    ) -> Path:
        """Save predictions in a compact task-appropriate representation.

        Semantic masks are written to ``masks/*.png`` and object-style output
        to ``predictions.json``. Existing output files are never overwritten.

        Parameters:
            destination: New or empty prediction directory.
            include_probabilities: Also export semantic foreground scores as
                float16 ``foreground-probabilities/*.npy`` files. Unified
                prediction caches retain these independently; the default
                avoids duplicating large maps in report directories.

        Returns:
            The resolved output directory.
        """

        root = Path(destination).expanduser().resolve()
        if root.exists():
            if not root.is_dir() or any(root.iterdir()):
                raise FileExistsError(f"Prediction destination is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": 1,
            "kind": "model-predictions",
            "model": self.model_name,
            "model_kind": self.model_kind,
            "task": self.task,
            "backend": self.backend,
            "images": len(self.records),
            "inference_seconds": self.inference_seconds,
            "settings": to_jsonable(self.settings),
        }
        (root / "prediction-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if self.task == "semantic_segment":
            mask_root = root / "masks"
            mask_root.mkdir(parents=True, exist_ok=True)
            probability_root = root / "foreground-probabilities"
            if include_probabilities and any(
                record.foreground_probability is not None
                for record in self.records
            ):
                probability_root.mkdir(parents=True, exist_ok=True)
            for record in self.records:
                if record.mask is None:
                    raise DatasetValidationError(
                        f"Semantic prediction {record.image_id!r} has no mask"
                    )
                mask = np.asarray(record.mask)
                if mask.ndim != 2 or not np.all(np.isfinite(mask)):
                    raise DatasetValidationError(
                        f"Semantic prediction {record.image_id!r} is not a finite class map"
                    )
                if np.any(mask < 0) or np.any(mask != np.floor(mask)):
                    raise DatasetValidationError(
                        f"Semantic prediction {record.image_id!r} contains invalid class IDs"
                    )
                dtype = np.uint8 if int(mask.max(initial=0)) <= 255 else np.uint16
                Image.fromarray(mask.astype(dtype)).save(
                    mask_root / f"{record.image_id}.png", format="PNG"
                )
                if include_probabilities and record.foreground_probability is not None:
                    probability = np.asarray(
                        record.foreground_probability,
                        dtype=np.float32,
                    )
                    if probability.shape != (record.height, record.width):
                        raise DatasetValidationError(
                            f"Foreground probability {record.image_id!r} has "
                            f"shape {probability.shape}; expected "
                            f"{(record.height, record.width)}"
                        )
                    np.save(
                        probability_root / f"{record.image_id}.npy",
                        np.clip(probability, 0.0, 1.0).astype(np.float16),
                    )
        else:
            rows = []
            for record in self.records:
                rows.append(
                    {
                        "image_id": record.image_id,
                        "image_path": str(record.image_path),
                        "relative_path": record.relative_path,
                        "width": record.width,
                        "height": record.height,
                        "predictions": [
                            {
                                "class_id": value.class_id,
                                "score": value.score,
                                "bbox": value.bbox,
                                "point": value.point,
                                "radius": value.radius,
                                "polygon": value.polygon,
                                "polygons": value.polygons,
                                "keypoints": value.keypoints,
                                "metadata": value.metadata,
                            }
                            for value in record.objects
                        ],
                    }
                )
            (root / "predictions.json").write_text(
                json.dumps(to_jsonable(rows), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return root

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe prediction summary."""

        return {
            "model": self.model_name,
            "model_kind": self.model_kind,
            "task": self.task,
            "backend": self.backend,
            "images": len(self.records),
            "predictions": sum(record.count for record in self.records),
            "inference_seconds": self.inference_seconds,
            "images_per_second": (
                len(self.records) / self.inference_seconds
                if self.inference_seconds > 0
                else None
            ),
            "cache": to_jsonable(self.cache_info),
        }

    def visualize(
        self,
        *,
        samples: int = 8,
        columns: int = 2,
        seed: int = 42,
        panel_size: float = 3.0,
        destination: str | Path | None = None,
    ) -> Any:
        """Render sampled original/prediction pairs.

        Parameters:
            samples: Maximum number of images to render.
            columns: Number of independent image pairs per figure row.
            seed: Deterministic sampling seed.
            panel_size: Approximate width/height of each image panel in inches.
            destination: Optional PNG output path.

        Returns:
            A Matplotlib figure.
        """

        if samples <= 0 or columns <= 0:
            raise ValueError("samples and columns must be positive")
        if not math.isfinite(panel_size) or panel_size <= 0:
            raise ValueError("panel_size must be a positive finite number")
        if not self.records:
            raise ValueError("PredictionResult contains no images")
        import matplotlib.pyplot as plt

        count = min(samples, len(self.records))
        if count == len(self.records):
            selected = list(self.records)
        else:
            rng = np.random.default_rng(seed)
            indices = sorted(
                rng.choice(len(self.records), size=count, replace=False).tolist()
            )
            selected = [self.records[index] for index in indices]
        rows = math.ceil(len(selected) / columns)
        figure = plt.figure(
            figsize=(panel_size * 2 * columns, (panel_size + 0.38) * rows)
        )
        figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        outer = figure.add_gridspec(rows, columns, wspace=0.10, hspace=0.18)
        for index, record in enumerate(selected):
            row = index // columns
            column = index % columns
            cell = outer[row, column].subgridspec(
                2,
                2,
                height_ratios=(0.10, 0.90),
                hspace=0.02,
                wspace=0.07,
            )
            title = figure.add_subplot(cell[0, :])
            title.set_axis_off()
            title.text(
                0.5,
                0.5,
                _shorten_middle(record.relative_path, 72),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="semibold",
            )
            with Image.open(record.image_path) as opened:
                image = np.asarray(opened.convert("RGB"))
            original = figure.add_subplot(cell[1, 0])
            original.imshow(image)
            prediction = figure.add_subplot(cell[1, 1])
            if record.mask is not None:
                mask = np.asarray(record.mask)
                prediction.imshow(
                    mask,
                    cmap="gray" if int(mask.max(initial=0)) <= 1 else "tab20",
                    vmin=0,
                    vmax=max(int(mask.max(initial=0)), 1),
                    interpolation="nearest",
                )
            else:
                prediction.imshow(image)
                _draw_object_predictions(prediction, record.objects)
            for axis in (original, prediction):
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
            if row == 0:
                original.set_title("Original", fontsize=9, pad=4)
                prediction.set_title(
                    _shorten_middle(self.model_name, 28),
                    fontsize=9,
                    pad=4,
                )
        if destination is not None:
            path = Path(destination).expanduser().resolve()
            if path.suffix.lower() != ".png":
                raise ValueError("visualization destination must be a PNG file")
            if path.exists():
                raise FileExistsError(f"Visualization already exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        return figure


@dataclass(frozen=True)
class _PredictionCacheRequest:
    cache: PredictionCache
    key: str
    identity: dict[str, Any]
    namespace: Literal["predictions", "semantic"]
    cohort: Any | None = None
    package_payload: dict[str, Any] | None = None
    postprocess: float = 0.7
    keep_native: bool = False
    compatible_identities: tuple[dict[str, Any], ...] = ()


class Model:
    """Auto-detected model with a common prediction and comparison API.

    A path to an official nnU-Net trained-model folder is detected from its
    ``dataset.json``/``plans.json`` files. File checkpoints are treated as
    Ultralytics-compatible models. Model-specific adapter settings live on the
    model instead of being repeated at each evaluation call.

    Parameters:
        source: Ultralytics-compatible checkpoint or official nnU-Net model
            folder. Passing ``fold_N`` for nnU-Net is normalized to its parent.
        name: Display name; defaults to the file stem or folder name.
        kind: Explicit adapter, or ``"auto"`` to inspect ``source``.
        task: Optional task override. nnU-Net is always semantic segmentation;
            Ultralytics task metadata is read from adjacent ``args.yaml`` when
            available and otherwise resolved lazily by Ultralytics.
        source_key: Stable external source identity retained in reports. W&B
            references use ``wandb:entity/project/run-id``.
        model_type: Compact architecture identifier such as ``yolo26x-sem``
            or ``nnunet-m``.
        source_dataset_zip: Portable training-dataset archive name used in
            automatically generated W&B model names.
        resolution: Default Ultralytics image size.
        training_dataset: Optional training-data provenance path.
        inference: Default Ultralytics inference mode.
        device: Default inference device.
        folds: nnU-Net folds selected for prediction.
        checkpoint: nnU-Net checkpoint filename within each fold.
        native_tile_size: Source tile size used to create training samples.
        upscale_factor: Input adapter scale used during training. ``None``
            preserves unknown standalone-checkpoint geometry.
        input_size: Adapter/model input size after upscaling.
        batch_size: Inference batch size. ``-1`` probes a large batch and
            halves it on accelerator out-of-memory errors until it fits.
        workers: nnU-Net CPU worker count for preprocessing and probability
            conversion. This is not the neural-network batch size, which is
            derived from the model's own plan and the inference device.
        nnunet_tta: Whether nnU-Net inference averages mirrored test-time
            augmentations. Disabled by default because it multiplies inference
            work by up to four for a 2D model.
        confidence: Backward-compatible instance-score floor.
        postprocess: Default native IoU or SAHI match threshold.
        prediction_threshold: Task-aware retention threshold: foreground
            probability for semantic models and instance score otherwise.
        foreground_probability_threshold: Backward-compatible semantic alias
            for ``prediction_threshold``.
        sahi_slice_height: SAHI tile height in canonical source pixels.
        sahi_slice_width: SAHI tile width in canonical source pixels.
        sahi_overlap: Default SAHI overlap ratio for both axes.
        sahi_overlap_height_ratio: Optional vertical overlap override.
        sahi_overlap_width_ratio: Optional horizontal overlap override.
        sahi_postprocess_type: SAHI merge algorithm.
        sahi_postprocess_match_metric: SAHI matching metric.
        sahi_postprocess_class_agnostic: Whether SAHI may merge predictions
            from different classes.
        sahi_model_type: SAHI adapter name.
        settings: Additional low-level adapter defaults.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        name: str | None = None,
        kind: Literal["auto", "ultralytics", "nnunet"] = "auto",
        task: ModelTask | None = None,
        source_key: str | None = None,
        model_type: str | None = None,
        source_dataset_zip: str | None = None,
        resolution: int | None = None,
        training_dataset: str | Path | None = None,
        inference: Literal["native", "sahi"] = "native",
        device: str | None = None,
        folds: tuple[int | str, ...] = (0,),
        checkpoint: str = "checkpoint_final.pth",
        native_tile_size: int | tuple[int, int] | None = None,
        upscale_factor: int | None = None,
        input_size: int | tuple[int, int] | None = None,
        batch_size: int = -1,
        workers: int = 2,
        nnunet_tta: bool = False,
        confidence: float = 0.25,
        postprocess: float = 0.7,
        foreground_probability_threshold: float | None = None,
        prediction_threshold: float | None = None,
        sahi_slice_height: int | None = None,
        sahi_slice_width: int | None = None,
        sahi_overlap: float | None = None,
        sahi_overlap_height_ratio: float | None = None,
        sahi_overlap_width_ratio: float | None = None,
        sahi_postprocess_type: Literal["GREEDYNMM", "NMM", "NMS", "LSNMS"] | None = None,
        sahi_postprocess_match_metric: Literal["IOU", "IOS"] | None = None,
        sahi_postprocess_class_agnostic: bool | None = None,
        sahi_model_type: str | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        path = Path(source).expanduser().resolve()
        if path.name.startswith("fold_") and not (path / "plans.json").is_file():
            candidate = path.parent
            if (candidate / "dataset.json").is_file() and (candidate / "plans.json").is_file():
                if folds == (0,):
                    folds = (path.name.removeprefix("fold_"),)
                path = candidate
        resolved_kind = self._detect_kind(path) if kind == "auto" else kind
        if resolved_kind not in {"ultralytics", "nnunet"}:
            raise ValueError("kind must be 'auto', 'ultralytics', or 'nnunet'")
        parsed_name = str(name or (path.stem if path.is_file() else path.name)).strip()
        if not parsed_name:
            raise ValueError("Model name must be non-empty")
        if inference not in {"native", "sahi"}:
            raise ValueError("inference must be 'native' or 'sahi'; 'auto' was removed")
        try:
            geometry = Geometry.create(
                native_tile_size=native_tile_size,
                upscale_factor=upscale_factor,
                input_size=input_size if input_size is not None else resolution,
                source=parsed_name,
            )
        except ValueError as exc:
            raise DatasetValidationError(
                ValidationIssue(
                    str(exc),
                    source=parsed_name,
                    value={
                        "native_tile_size": native_tile_size,
                        "upscale_factor": upscale_factor,
                        "input_size": input_size,
                    },
                )
            ) from exc
        if resolution is None and geometry.input_size is not None:
            if geometry.input_size[0] == geometry.input_size[1]:
                resolution = geometry.input_size[0]
        if resolution is not None and (
            isinstance(resolution, bool) or int(resolution) <= 0
        ):
            raise ValueError("resolution must be a positive integer")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise ValueError("workers must be a positive integer")
        if not isinstance(nnunet_tta, bool):
            raise ValueError("nnunet_tta must be a boolean")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size == 0
            or batch_size < -1
            or batch_size > _MAX_INFERENCE_BATCH_SIZE
        ):
            raise ValueError("batch_size must be -1 or an integer from 1 through 128")

        self._path = path
        self._name = parsed_name
        self._slug = slugify(parsed_name)
        self._kind: ModelKind = resolved_kind
        self._source_key = str(source_key or source)
        self._model_type = str(model_type or "").strip() or (
            "nnunet" if resolved_kind == "nnunet" else "ultralytics"
        )
        self._source_dataset_zip = (
            Path(str(source_dataset_zip)).name if source_dataset_zip else None
        )
        self._resolution = int(resolution) if resolution is not None else None
        self._inference = inference
        self._device = device
        self._settings = dict(settings or {})
        if geometry.native_tile_size is not None:
            if sahi_slice_height is None:
                sahi_slice_height = geometry.native_tile_size[0]
            if sahi_slice_width is None:
                sahi_slice_width = geometry.native_tile_size[1]
        explicit_settings = {
            "confidence": confidence,
            "postprocess": postprocess,
            "foreground_probability_threshold": foreground_probability_threshold,
            "prediction_threshold": prediction_threshold,
            "sahi_slice_height": sahi_slice_height,
            "sahi_slice_width": sahi_slice_width,
            "sahi_overlap": sahi_overlap,
            "sahi_overlap_height_ratio": sahi_overlap_height_ratio,
            "sahi_overlap_width_ratio": sahi_overlap_width_ratio,
            "sahi_postprocess_type": sahi_postprocess_type,
            "sahi_postprocess_match_metric": sahi_postprocess_match_metric,
            "sahi_postprocess_class_agnostic": sahi_postprocess_class_agnostic,
            "sahi_model_type": sahi_model_type,
        }
        self._settings.update(
            {key: value for key, value in explicit_settings.items() if value is not None}
        )
        if (
            self._settings.get("prediction_threshold") is not None
            and self._settings.get("foreground_probability_threshold") is not None
        ):
            raise ValueError(
                "Use prediction_threshold; do not configure both threshold aliases"
            )
        removed_comparison_settings = sorted(
            {
                "baseline",
                "comparison_space",
                "protocol",
                "calibration_split",
                "training_provenance",
                "comparison_unit",
                "prediction_plots",
                "bootstrap_resamples",
                "seed",
            }
            & self._settings.keys()
        )
        if removed_comparison_settings:
            raise ValueError(
                "Removed comparison options cannot be stored on a model: "
                + ", ".join(removed_comparison_settings)
            )
        from .sahi_support import reject_legacy_sahi_settings, resolve_sahi_settings

        reject_legacy_sahi_settings(self._settings)
        if inference == "sahi" or any(
            key.startswith("sahi_") for key in self._settings
        ):
            resolve_sahi_settings(
                self._settings,
                resolution=int(resolution) if resolution is not None else 480,
            )
        for key in (
            "confidence",
            "postprocess",
            "foreground_probability_threshold",
            "prediction_threshold",
        ):
            if key in self._settings:
                value = float(self._settings[key])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{key} must be finite and in [0, 1]")
                self._settings[key] = value
        removed_thresholds = {
            key: "confidence" if key == "confidence_thresholds" else "postprocess"
            for key in ("confidence_thresholds", "postprocess_thresholds")
            if key in self._settings
        }
        if removed_thresholds:
            migration = ", ".join(
                f"{old}->{new}" for old, new in removed_thresholds.items()
            )
            raise ValueError(
                "Threshold sweeps were removed; configure one fixed value per model: "
                f"{migration}"
            )
        self._geometry = geometry
        self._upscale_factor = geometry.upscale_factor or 1
        self._batch_size = batch_size
        self._workers = workers
        self._nnunet_tta = nnunet_tta
        self._folds: tuple[str, ...] = ()
        self._checkpoint = checkpoint
        self._checkpoint_files: tuple[Path, ...] = ()
        self._checkpoint_sha256 = ""
        self._digest = ""
        self._runtime: dict[Any, Any] = {}
        self._resolved_task = _normalize_prediction_task(task)

        if resolved_kind == "nnunet":
            self._initialize_nnunet(folds, checkpoint)
            if self._resolved_task not in {None, "semantic_segment"}:
                raise ValueError("Official nnU-Net models use task='semantic_segment'")
            self._resolved_task = "semantic_segment"
            if device is not None and device not in {"cpu", "cuda", "mps"}:
                raise ValueError("nnU-Net device must be 'cpu', 'cuda', or 'mps'")
        else:
            if not path.is_file():
                raise DatasetValidationError(
                    ValidationIssue(
                        "Ultralytics model source is not a file",
                        value=str(path),
                        expected="a model checkpoint or exported model file",
                    )
                )
            self._digest = sha256_file(path)
            if self._resolved_task is None:
                self._resolved_task = _task_from_args(path)

        inferred_training = training_dataset or (
            _training_dataset_from_args(path) if resolved_kind == "ultralytics" else None
        )
        self._training_dataset = (
            Path(inferred_training).expanduser().resolve() if inferred_training else None
        )

    @staticmethod
    def _detect_kind(path: Path) -> ModelKind:
        if path.is_dir() and (path / "dataset.json").is_file() and (path / "plans.json").is_file():
            return "nnunet"
        if path.is_file():
            return "ultralytics"
        raise DatasetValidationError(
            ValidationIssue(
                "Could not detect a supported model",
                value=str(path),
                expected=(
                    "an Ultralytics-compatible model file or an official nnU-Net "
                    "folder containing dataset.json and plans.json"
                ),
            )
        )

    def _initialize_nnunet(
        self,
        folds: tuple[int | str, ...],
        checkpoint: str,
    ) -> None:
        dataset_json = self._path / "dataset.json"
        plans_json = self._path / "plans.json"
        selected_folds = _normalize_folds(folds)
        if not checkpoint or Path(checkpoint).name != checkpoint:
            raise ValueError("checkpoint must be a filename within each selected fold")
        _validate_nnunet_dataset(dataset_json, self._name)
        if not plans_json.is_file():
            raise DatasetValidationError(
                ValidationIssue(
                    "Incomplete nnU-Net model folder",
                    source=self._name,
                    value=str(plans_json),
                    expected="plans.json",
                )
            )
        checkpoint_files = tuple(
            self._path / f"fold_{fold}" / checkpoint for fold in selected_folds
        )
        missing = [str(path) for path in checkpoint_files if not path.is_file()]
        if missing:
            raise DatasetValidationError(
                ValidationIssue(
                    "nnU-Net checkpoint is missing",
                    source=self._name,
                    value=missing,
                    expected=f"{checkpoint} in every selected fold",
                )
            )
        self._folds = selected_folds
        self._checkpoint_files = checkpoint_files
        self._checkpoint_sha256 = _combined_sha256(
            checkpoint_files,
            relative_to=self._path,
        )
        self._digest = _combined_sha256(
            (dataset_json, plans_json, *checkpoint_files),
            relative_to=self._path,
        )

    @property
    def name(self) -> str:
        """Human-readable model name."""

        return self._name

    @property
    def slug(self) -> str:
        """Filesystem-safe model name."""

        return self._slug

    @property
    def source_key(self) -> str:
        """Stable source reference used to load this model."""

        return self._source_key

    @property
    def model_type(self) -> str:
        """Compact architecture identifier used in reports."""

        return self._model_type

    @property
    def source_dataset_zip(self) -> str | None:
        """Portable source training-dataset archive name, when recorded."""

        return self._source_dataset_zip

    @property
    def path(self) -> Path:
        """Resolved checkpoint or trained-model folder."""

        return self._path

    @property
    def model_folder(self) -> Path:
        """Official nnU-Net model folder.

        Raises for non-nnU-Net models so adapter mistakes fail early.
        """

        if self.kind != "nnunet":
            raise AttributeError("model_folder is only available for nnU-Net models")
        return self._path

    @property
    def kind(self) -> ModelKind:
        """Detected model adapter."""

        return self._kind

    @property
    def task(self) -> PredictionTask | None:
        """Detected/configured task, or ``None`` until lazy model loading."""

        return self._resolved_task

    @property
    def resolution(self) -> int | None:
        """Default Ultralytics inference resolution."""

        return self._resolution

    @property
    def inference(self) -> str:
        """Default inference mode."""

        return self._inference

    @property
    def device(self) -> str | None:
        """Configured inference device, or ``None`` for runtime selection."""

        return self._device

    def _resolved_device(self, override: str | None = None) -> str | None:
        """Resolve an execution device without expanding the public API."""

        configured = override if override is not None else self.device
        if self.kind == "nnunet" and configured is None:
            return _default_nnunet_device()
        return configured

    @property
    def confidence(self) -> float:
        """Fixed prediction/evaluation confidence floor."""

        if not self._uses_semantic_prediction_threshold() and self._settings.get(
            "prediction_threshold"
        ) is not None:
            return float(self._settings["prediction_threshold"])
        return float(self._settings["confidence"])

    @property
    def postprocess(self) -> float:
        """Fixed native IoU or SAHI match threshold."""

        return float(self._settings["postprocess"])

    @property
    def foreground_probability_threshold(self) -> float | None:
        """Foreground cutoff for semantic models, or ``None`` for argmax."""

        value = self._settings.get("foreground_probability_threshold")
        if self._uses_semantic_prediction_threshold() and self._settings.get(
            "prediction_threshold"
        ) is not None:
            value = self._settings["prediction_threshold"]
        return None if value is None else float(value)

    @property
    def prediction_threshold(self) -> float:
        """Task-aware cutoff used to retain semantic pixels or instances."""

        if self._uses_semantic_prediction_threshold():
            value = self.foreground_probability_threshold
            return 0.5 if value is None else value
        return self.confidence

    def _uses_semantic_prediction_threshold(self) -> bool:
        """Recognize semantic checkpoints before lazy backend task loading."""

        return (
            self.kind == "nnunet"
            or self.task == "semantic_segment"
            or "-sem" in self.model_type.lower()
            or "-sem" in self.path.stem.lower()
        )

    @property
    def training_dataset(self) -> Path | None:
        """Training-data provenance path when known."""

        return self._training_dataset

    @property
    def folds(self) -> tuple[str, ...]:
        """Selected nnU-Net folds."""

        return self._folds

    @property
    def checkpoint(self) -> str:
        """Selected nnU-Net checkpoint filename."""

        return self._checkpoint

    @property
    def checkpoint_files(self) -> tuple[Path, ...]:
        """Resolved nnU-Net checkpoint files."""

        return self._checkpoint_files

    @property
    def checkpoint_sha256(self) -> str:
        """Combined selected-checkpoint hash for nnU-Net models."""

        return self._checkpoint_sha256

    @property
    def digest(self) -> str:
        """Content digest used for cache and comparison identity."""

        return self._digest

    @property
    def upscale_factor(self) -> int:
        """Input adapter scale; defaults to one when geometry is unknown."""

        return self._upscale_factor

    @property
    def geometry(self) -> Geometry:
        """Normalized training/evaluation geometry known for this model."""

        return self._geometry

    @property
    def native_tile_size(self) -> tuple[int, int] | None:
        """Source tile size used during training, when proven."""

        return self.geometry.native_tile_size

    @property
    def input_size(self) -> tuple[int, int] | None:
        """Adapter/model input size after preprocessing, when known."""

        return self.geometry.input_size

    @property
    def effective_resolution(self) -> tuple[int, int] | None:
        """Model input height/width used for prediction, when known."""

        if self.input_size is not None:
            return self.input_size
        if self.resolution is not None:
            return (self.resolution, self.resolution)
        return None

    @property
    def workers(self) -> int:
        """Default nnU-Net worker count."""

        return self._workers

    @property
    def batch_size(self) -> int:
        """Requested inference batch size; ``-1`` enables adaptive sizing."""

        return self._batch_size

    @property
    def nnunet_tta(self) -> bool:
        """Whether nnU-Net inference averages mirrored test-time inputs."""

        return self._nnunet_tta

    @property
    def settings(self) -> dict[str, Any]:
        """Copy of additional adapter defaults."""

        return dict(self._settings)

    @property
    def loaded(self) -> bool:
        """Whether a heavyweight runtime has been initialized lazily."""

        return bool(self._runtime)

    def unload(self) -> None:
        """Release cached inference runtimes while preserving model metadata."""

        self._runtime.clear()

    def _runtime_model(self, key: Any, factory: Callable[[], Any]) -> Any:
        if key not in self._runtime:
            self._runtime[key] = factory()
        return self._runtime[key]

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe model identity and adapter summary."""

        return {
            "name": self.name,
            "source_key": self.source_key,
            "model_type": self.model_type,
            "source_dataset_zip": self.source_dataset_zip,
            "kind": self.kind,
            "task": self.task,
            "path": str(self.path),
            "digest": self.digest,
            "resolution": self.resolution,
            "training_dataset": (
                str(self.training_dataset) if self.training_dataset else None
            ),
            "inference": self.inference,
            "device": self.device,
            "confidence": self.confidence,
            "postprocess": self.postprocess,
            "foreground_probability_threshold": self.foreground_probability_threshold,
            "prediction_threshold": self.prediction_threshold,
            "folds": self.folds,
            "checkpoint": self.checkpoint if self.kind == "nnunet" else None,
            "checkpoint_sha256": (
                self.checkpoint_sha256 if self.kind == "nnunet" else None
            ),
            "geometry": self.geometry.as_dict(),
            "native_tile_size": self.native_tile_size,
            "upscale_factor": self.geometry.upscale_factor,
            "input_size": self.input_size,
            "effective_resolution": self.effective_resolution,
            "batch_size": self.batch_size,
            "workers": self.workers if self.kind == "nnunet" else None,
            **({"nnunet_tta": self.nnunet_tta} if self.kind == "nnunet" else {}),
            "settings": to_jsonable(self.settings),
        }

    def _configured_copy(
        self,
        *,
        name: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "Model":
        """Clone this lazy model without sharing or initializing its runtime."""

        values = dict(overrides or {})
        settings = {**self.settings, **dict(values.pop("settings", {}) or {})}
        requested_prediction_threshold = values.get(
            "prediction_threshold",
            self._settings.get("prediction_threshold"),
        )
        if requested_prediction_threshold is not None:
            settings.pop("foreground_probability_threshold", None)
        native = values.pop("native_tile_size", self.geometry.native_tile_size)
        factor = values.pop("upscale_factor", self.geometry.upscale_factor)
        model_input = values.pop(
            "input_size",
            values.pop("model_input_size", self.geometry.input_size),
        )
        defaults: dict[str, Any] = {
            "kind": self.kind,
            "task": self.task,
            "source_key": self.source_key,
            "model_type": self.model_type,
            "source_dataset_zip": self.source_dataset_zip,
            "resolution": self.resolution,
            "training_dataset": self.training_dataset,
            "inference": self.inference,
            "device": self.device,
            "folds": self.folds or (0,),
            "checkpoint": self.checkpoint,
            "native_tile_size": native,
            "upscale_factor": factor,
            "input_size": model_input,
            "batch_size": self.batch_size,
            "workers": self.workers,
            "nnunet_tta": self.nnunet_tta,
            "confidence": self.confidence,
            "postprocess": self.postprocess,
            "foreground_probability_threshold": (
                None
                if requested_prediction_threshold is not None
                else self.foreground_probability_threshold
            ),
            "prediction_threshold": requested_prediction_threshold,
            "settings": settings,
        }
        defaults.update(values)
        return Model(self.path, name=name or self.name, **defaults)

    def _apply_automatic_name(self) -> None:
        """Build a compact dataset/run/architecture/resolution identity."""

        resolution = self.effective_resolution
        if resolution is None:
            resolution_label = "unknown-resolution"
        elif resolution[0] == resolution[1]:
            resolution_label = f"{resolution[0]}px"
        else:
            resolution_label = f"{resolution[1]}x{resolution[0]}px"
        run_id = self.source_key.rsplit("/", 1)[-1]
        dataset_prefix = (
            Path(self.source_dataset_zip).stem
            if self.source_dataset_zip
            else "unknown-dataset"
        )
        self._name = "__".join(
            (
                slugify(dataset_prefix),
                slugify(run_id),
                slugify(self.model_type),
                resolution_label,
            )
        )
        self._slug = slugify(self._name)

    def predict(
        self,
        source: Any,
        *,
        split: Literal["train", "val", "test"] | None = None,
        inference: Literal["native", "sahi"] | None = None,
        resolution: int | None = None,
        confidence: float | None = None,
        postprocess: float | None = None,
        prediction_threshold: float | None = None,
        foreground_probability_threshold: float | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        errors: Literal["raise", "skip"] = "raise",
        progress: bool = True,
        destination: str | Path | None = None,
        prediction_cache: bool | str | Path | PredictionCache = False,
        cache_only: bool = False,
        require_probability_maps: bool = False,
        settings: Mapping[str, Any] | None = None,
        sahi_slice_height: int | None = None,
        sahi_slice_width: int | None = None,
        sahi_overlap: float | None = None,
        sahi_overlap_height_ratio: float | None = None,
        sahi_overlap_width_ratio: float | None = None,
        sahi_postprocess_type: Literal["GREEDYNMM", "NMM", "NMS", "LSNMS"] | None = None,
        sahi_postprocess_match_metric: Literal["IOU", "IOS"] | None = None,
        sahi_postprocess_class_agnostic: bool | None = None,
        sahi_model_type: str | None = None,
        _keep_native: bool = False,
    ) -> PredictionResult:
        """Predict images, directories, datasets, exports, or frozen inputs.

        This is the shared inference entry point used by manual prediction,
        collection prediction, and comparison. Heavyweight runtimes and a
        successfully resolved adaptive batch size are retained on this model
        instance until :meth:`unload` is called.

        Parameters:
            source: Image path, image directory, sequence of paths or
                :class:`ModelInput` values, a :class:`Dataset`, or an internal
                frozen cohort.
            split: Dataset/export split. Defaults to ``"val"`` for dataset-like
                sources and is ignored for direct image inputs.
            inference: Explicit native or sliced SAHI inference.
            resolution: Ultralytics input size override.
            confidence: Prediction confidence floor override. If omitted, use
                the model's ``confidence`` setting, then ``0.25``.
            postprocess: Native IoU or SAHI match-threshold override. If
                omitted, use the model's ``postprocess`` setting, then ``0.7``.
            prediction_threshold: Task-aware retention threshold. For semantic
                models this is a foreground-probability cutoff after full-image
                reconstruction; for instance models it is the minimum score.
            foreground_probability_threshold: Semantic foreground-probability
                cutoff retained as a backward-compatible alias for
                ``prediction_threshold``.
            device: Device override.
            batch_size: Inference batch override. ``-1`` adaptively retries
                accelerator OOM failures with smaller batches.
            errors: Oversized-input policy for native inference. Images at or
                below the model's declared ``native_tile_size`` are always
                predicted. ``"raise"`` rejects an image exceeding either
                dimension; ``"skip"`` omits it and records the omission in the
                result settings. SAHI accepts larger full images and slices them
                with the configured native tile geometry.
            progress: Show package-managed progress bars.
            destination: Optional new/empty directory receiving saved output.
            prediction_cache: Opt-in verified prediction caching. ``True`` uses
                the dataset-local comparison cache for :class:`Dataset`
                inputs and package-managed storage otherwise. A path is an
                explicit cache base; :class:`PredictionCache` is also accepted.
            cache_only: Never run inference. Raise
                :class:`PredictionCacheMissError` when a complete compatible
                cache entry is unavailable.
            require_probability_maps: Require canonical semantic foreground
                probabilities rather than accepting a legacy hard-mask-only
                cache. This is intended for threshold calibration.
            settings: Additional per-call adapter overrides.
            sahi_slice_height: Optional SAHI tile height in source pixels.
            sahi_slice_width: Optional SAHI tile width in source pixels.
            sahi_overlap: Optional default overlap ratio for both tile axes.
            sahi_overlap_height_ratio: Optional vertical overlap override.
            sahi_overlap_width_ratio: Optional horizontal overlap override.
            sahi_postprocess_type: Optional SAHI merge algorithm override.
            sahi_postprocess_match_metric: Optional SAHI matching metric.
            sahi_postprocess_class_agnostic: Whether SAHI may merge predictions
                from different classes.
            sahi_model_type: Optional SAHI adapter name.
            _keep_native: Internal nnU-Net evaluation flag retaining the
                adapter-scale semantic mask.

        Returns:
            Ordered :class:`PredictionResult` values with stable image IDs.
        """

        errors = normalize_errors(errors)
        cache_override = getattr(self, "_prediction_cache_override", None)
        if cache_override is None:
            cache_identity_override = None
            cache_namespace_override = None
            cache_compatible_identity_overrides: tuple[dict[str, Any], ...] = ()
        else:
            if len(cache_override) == 3:
                (
                    prediction_cache,
                    cache_identity_override,
                    cache_namespace_override,
                ) = cache_override
                cache_compatible_identity_overrides = ()
            else:
                (
                    prediction_cache,
                    cache_identity_override,
                    cache_namespace_override,
                    cache_compatible_identity_overrides,
                ) = cache_override
        combined_settings = {**self.settings, **dict(settings or {})}
        if (
            prediction_threshold is not None
            and foreground_probability_threshold is not None
        ):
            raise ValueError(
                "Pass prediction_threshold, not both threshold aliases"
            )
        semantic_task = self._uses_semantic_prediction_threshold()
        if prediction_threshold is not None:
            combined_settings["prediction_threshold"] = float(prediction_threshold)
        elif foreground_probability_threshold is not None:
            combined_settings["foreground_probability_threshold"] = float(
                foreground_probability_threshold
            )
        effective_confidence = (
            float(prediction_threshold)
            if prediction_threshold is not None and not semantic_task
            else float(confidence)
            if confidence is not None
            else self.confidence
        )
        effective_postprocess = (
            float(postprocess)
            if postprocess is not None
            else self.postprocess
        )
        effective_foreground_threshold = (
            float(prediction_threshold)
            if prediction_threshold is not None and semantic_task
            else float(foreground_probability_threshold)
            if foreground_probability_threshold is not None
            else self.foreground_probability_threshold
        )
        if not math.isfinite(effective_confidence) or not 0 <= effective_confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        if not math.isfinite(effective_postprocess) or not 0 <= effective_postprocess <= 1:
            raise ValueError("postprocess must be finite and in [0, 1]")
        if effective_foreground_threshold is not None and (
            not math.isfinite(effective_foreground_threshold)
            or not 0 <= effective_foreground_threshold <= 1
        ):
            raise ValueError(
                "foreground_probability_threshold must be finite and in [0, 1]"
            )
        if cache_only and prediction_cache is False:
            raise ValueError("cache_only=True requires prediction_cache to be enabled")
        requested = inference or self.inference
        if requested not in {"native", "sahi"}:
            raise ValueError("inference must be 'native' or 'sahi'; 'auto' was removed")
        normalized_override = getattr(
            self,
            "_normalized_prediction_override",
            None,
        )
        if (
            isinstance(normalized_override, tuple)
            and len(normalized_override) == 5
            and normalized_override[0] is source
            and normalized_override[1] == split
        ):
            inputs = tuple(normalized_override[2])
            source_task = normalized_override[3]
            cache_context = dict(normalized_override[4])
        else:
            inputs, source_task, cache_context = normalize_model_inputs(
                source, split=split, progress=progress
            )
        inputs, skipped_inputs = filter_inputs_by_size(
            inputs,
            maximum=None if requested == "sahi" else self.native_tile_size,
            errors=errors,
            source=self.name,
        )
        if not inputs:
            raise ValueError("Prediction source contains no supported images")
        selected_device = self._resolved_device(device)
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if (
            isinstance(effective_batch_size, bool)
            or not isinstance(effective_batch_size, int)
            or effective_batch_size == 0
            or effective_batch_size < -1
            or effective_batch_size > _MAX_INFERENCE_BATCH_SIZE
        ):
            raise ValueError("batch_size must be -1 or an integer from 1 through 128")
        explicit_sahi = {
            "sahi_overlap": sahi_overlap,
            "sahi_postprocess_type": sahi_postprocess_type,
            "sahi_postprocess_match_metric": sahi_postprocess_match_metric,
            "sahi_postprocess_class_agnostic": sahi_postprocess_class_agnostic,
            "sahi_model_type": sahi_model_type,
        }
        combined_settings.update(
            {key: value for key, value in explicit_sahi.items() if value is not None}
        )
        if sahi_slice_height is not None:
            combined_settings["sahi_slice_height"] = sahi_slice_height
        if sahi_slice_width is not None:
            combined_settings["sahi_slice_width"] = sahi_slice_width
        if sahi_overlap_height_ratio is not None:
            combined_settings["sahi_overlap_height_ratio"] = sahi_overlap_height_ratio
        if sahi_overlap_width_ratio is not None:
            combined_settings["sahi_overlap_width_ratio"] = sahi_overlap_width_ratio
        from .sahi_support import reject_legacy_sahi_settings, resolve_sahi_settings

        reject_legacy_sahi_settings(combined_settings)
        effective_resolution = resolution or self.resolution or 480
        resolved_sahi = resolve_sahi_settings(
            combined_settings,
            resolution=effective_resolution,
        )
        from .comparison.inference import resolve_backend

        known_task = (
            "semantic_segment"
            if self.kind == "nnunet"
            else self.task
            or (source_task if source_task != "semantic_segment" else None)
        )
        selected_backend = resolve_backend(
            requested,
            known_task or "detect",
        )
        maximum_size = (
            None
            if requested == "sahi"
            else list(self.native_tile_size) if self.native_tile_size else None
        )
        if requested == "sahi":
            oversized_action = "retain-for-sahi-slicing"
        elif maximum_size is None:
            oversized_action = "retain"
        else:
            oversized_action = "skip" if errors == "skip" else "raise"
        source_size_policy = {
            "errors": errors,
            "maximum_size": maximum_size,
            "smaller_or_equal": "retain",
            "oversized": oversized_action,
            "skipped_inputs": list(skipped_inputs),
        }
        cache_request = _prepare_prediction_cache_request(
            model=self,
            source=source,
            inputs=inputs,
            source_task=source_task,
            cache_context=cache_context,
            prediction_cache=prediction_cache,
            backend=selected_backend,
            inference=requested,
            resolution=effective_resolution,
            confidence=effective_confidence,
            postprocess=effective_postprocess,
            combined_settings=combined_settings,
            resolved_sahi=resolved_sahi.as_dict(),
            keep_native=_keep_native,
            identity_override=cache_identity_override,
            namespace_override=cache_namespace_override,
            compatible_identity_overrides=cache_compatible_identity_overrides,
        )
        cached_result = _load_prediction_cache_request(
            cache_request,
            inputs=inputs,
            model=self,
            backend=selected_backend,
            source_size_policy=source_size_policy,
            device=selected_device,
            batch_size=effective_batch_size,
            resolution=effective_resolution,
            confidence=effective_confidence,
            postprocess=effective_postprocess,
            combined_settings=combined_settings,
            resolved_sahi=resolved_sahi.as_dict(),
            foreground_probability_threshold=effective_foreground_threshold,
            require_probability_maps=require_probability_maps,
        )
        if cached_result is not None:
            if destination is not None:
                cached_result.save(destination)
            return cached_result
        if cache_only:
            detail = (
                "a verified cache entry with foreground probability maps"
                if require_probability_maps
                else "a verified complete prediction cache entry"
            )
            raise PredictionCacheMissError(
                f"Cache-only prediction for {self.name!r} requires {detail}",
                reason=(
                    "missing-probability-maps"
                    if require_probability_maps
                    else "missing"
                ),
            )

        started = time.perf_counter()
        if self.kind == "nnunet":
            from .semantic_comparison import predict_nnunet_model

            records = predict_nnunet_model(
                self,
                inputs,
                device=str(selected_device),
                progress=progress,
                keep_native=_keep_native,
                inference=selected_backend,
                resolution=effective_resolution,
                settings={**combined_settings, **resolved_sahi.as_dict()},
                batch_size=effective_batch_size,
                foreground_probability_threshold=effective_foreground_threshold,
            )
            backend = selected_backend
            task: PredictionTask = "semantic_segment"
            from .semantic_comparison import SEMANTIC_PREDICTION_SCHEMA

            resolved_settings = {
                "schema": SEMANTIC_PREDICTION_SCHEMA,
                "device": selected_device,
                "folds": self.folds,
                "checkpoint": self.checkpoint,
                "upscale_factor": self.upscale_factor,
                "workers": self.workers,
                "batch_size": effective_batch_size,
                "nnunet_tta": self.nnunet_tta,
                "inference": selected_backend,
                **(resolved_sahi.as_dict() if selected_backend == "sahi" else {}),
                **{
                    key: value
                    for key, value in (records[0].metadata if records else {}).items()
                    if key.startswith("nnunet_")
                },
            }
        else:
            from .comparison.inference import predict_model_inputs

            by_id, resolved_task, inference_telemetry = predict_model_inputs(
                self,
                inputs,
                task=known_task,
                backend=selected_backend,
                resolution=effective_resolution,
                confidence=effective_confidence,
                postprocess=effective_postprocess,
                device=selected_device,
                progress=progress,
                settings={**combined_settings, **resolved_sahi.as_dict()},
                batch_size=effective_batch_size,
                foreground_probability_threshold=effective_foreground_threshold,
            )
            self._resolved_task = resolved_task
            task = resolved_task
            backend = selected_backend
            if task == "semantic_segment":
                semantic_records: list[ImagePrediction] = []
                for value in inputs:
                    semantic = by_id[value.image_id]
                    class_map = getattr(semantic, "class_map", semantic)
                    probability = getattr(
                        semantic,
                        "foreground_probability",
                        None,
                    )
                    semantic_records.append(
                        ImagePrediction(
                            image_id=value.image_id,
                            image_path=value.image_path,
                            relative_path=value.relative_path,
                            width=value.width,
                            height=value.height,
                            mask=np.asarray(class_map),
                            foreground_probability=(
                                None
                                if probability is None
                                else np.asarray(probability, dtype=np.float16)
                            ),
                            metadata={
                                "backend": backend,
                                "probability_source": getattr(
                                    semantic,
                                    "probability_source",
                                    "class-map-only",
                                ),
                            },
                        )
                    )
                records = tuple(semantic_records)
            else:
                records = tuple(
                    ImagePrediction(
                        image_id=value.image_id,
                        image_path=value.image_path,
                        relative_path=value.relative_path,
                        width=value.width,
                        height=value.height,
                        objects=tuple(by_id[value.image_id]),
                        metadata={"backend": backend},
                    )
                    for value in inputs
                )
            resolved_settings = {
                "device": selected_device,
                "resolution": effective_resolution,
                "confidence": effective_confidence,
                "postprocess": effective_postprocess,
                "foreground_probability_threshold": effective_foreground_threshold,
                "prediction_threshold": (
                    effective_foreground_threshold
                    if task == "semantic_segment"
                    else effective_confidence
                ),
                "batch_size": effective_batch_size,
                **inference_telemetry,
                **combined_settings,
                **(resolved_sahi.as_dict() if selected_backend == "sahi" else {}),
            }
        resolved_settings["source_size_policy"] = source_size_policy
        result = PredictionResult(
            model_name=self.name,
            model_kind=self.kind,
            task=task,
            backend=backend,
            records=tuple(records),
            inference_seconds=time.perf_counter() - started,
            settings=resolved_settings,
        )
        result = _save_prediction_cache_request(
            cache_request,
            inputs=inputs,
            result=result,
        )
        if result.task == "semantic_segment" and (
            effective_foreground_threshold is not None
            or require_probability_maps
        ):
            missing_scores = [
                record.image_id
                for record in result.records
                if record.foreground_probability is None
            ]
            if missing_scores:
                # The hard-mask result was deliberately cached above. It is
                # still valid for ordinary argmax/class-map use, but cannot be
                # re-thresholded or calibrated without real model scores.
                raise PredictionScoreUnavailableError(
                    f"Semantic thresholding for {self.name!r} requires foreground "
                    "probabilities or logits, but the backend returned only hard "
                    f"class maps for {len(missing_scores)} image(s)",
                    reason="backend-returned-no-semantic-probabilities",
                )
        if destination is not None:
            result.save(destination)
        return result

    def compare(
        self,
        source: Any,
        *,
        split: Literal["train", "val", "test"] = "val",
        save_prediction_plots: bool = False,
        errors: Literal["raise", "skip"] = "raise",
        progress: bool = True,
        destination: str | Path | None = None,
        prediction_cache: bool | str | Path | PredictionCache | None = None,
        trust_legacy_cache: bool = False,
        min_connected_component_area: float | None = None,
        group_by: Callable[[Path], Hashable] | None = None,
    ) -> Any:
        """Evaluate this model on one frozen dataset cohort.

        Parameters:
            source: Fixed, on-disk :class:`Dataset` to evaluate.
            split: Dataset split forming the frozen evaluation cohort.
            save_prediction_plots: Write annotated comparison grids under
                ``predictions/`` for the cases the report keeps in
                ``worst_cases``. Cases with neither a reference nor a
                prediction are skipped, since their panels are empty.
            errors: Oversized-image policy shared by all evaluated models.
                Smaller images are always retained; ``"skip"`` omits images
                exceeding the common native-size limit and audits them.
            progress: Show package-managed progress bars.
            destination: Optional report directory. By default the report is
                content-addressed below ``<dataset>/evaluations/``.
            prediction_cache: Optional prediction-cache override. Omission or
                ``True`` preserves the established dataset-local cache;
                ``False`` disables persistent prediction caching.
            trust_legacy_cache: Reuse and migrate one structurally complete
                legacy semantic prediction cache when its exact model name and
                frozen cohort paths match, even if the old entry lacks model
                bytes/settings identity. Use only when you trust the cache's
                provenance; the default remains verified reuse only.
            min_connected_component_area: Minimum 8-connected predicted
                foreground-component area used for the filtered image-level
                presence metrics. ``None`` resolves to the held-out reference
                object-area p10. Raw any-pixel presence metrics are always
                reported alongside the filtered variants.
            group_by: Optional callback from each evaluation image path
                to a stable, hashable group label. Adds group-pooled metrics
                and a separate plot without changing inference or ranking.

        Returns:
            A task-appropriate comparison result. Prediction and evaluation
            settings come exclusively from this model.
        """

        return ModelCollection((self,)).compare(
            source,
            split=split,
            save_prediction_plots=save_prediction_plots,
            errors=errors,
            progress=progress,
            destination=destination,
            prediction_cache=prediction_cache,
            trust_legacy_cache=trust_legacy_cache,
            min_connected_component_area=min_connected_component_area,
            group_by=group_by,
        )

    def visualize(
        self,
        source: Any,
        *,
        split: Literal["train", "val", "test"] | None = None,
        samples: int = 8,
        columns: int = 2,
        seed: int = 42,
        panel_size: float = 3.0,
        destination: str | Path | None = None,
        errors: Literal["raise", "skip"] = "raise",
        progress: bool = True,
        prediction_options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Predict a source and render sampled original/prediction pairs.

        Parameters:
            source: Any source accepted by :meth:`predict`.
            split: Dataset split. Direct image inputs ignore this value.
            samples: Maximum number of images to render.
            columns: Number of image pairs per figure row.
            seed: Deterministic sampling seed.
            panel_size: Width and height, in inches, of each image panel.
            destination: Optional PNG output path.
            errors: Oversized-input policy forwarded to :meth:`predict`.
            progress: Show package-managed progress bars.
            prediction_options: Optional per-call overrides forwarded to
                :meth:`predict`.

        Returns:
            The rendered Matplotlib figure.
        """

        options = dict(prediction_options or {})
        options.setdefault("split", split)
        options.setdefault("errors", errors)
        options.setdefault("progress", progress)
        result = self.predict(source, **options)
        return result.visualize(
            samples=samples,
            columns=columns,
            seed=seed,
            panel_size=panel_size,
            destination=destination,
        )

    @classmethod
    def load_many(
        cls,
        models: Any,
    ) -> "ModelCollection":
        """Normalize paths/configurations into an ordered model collection.

        Parameters:
            models: A model, local checkpoint/folder/bundle, W&B run reference,
                sequence of sources, source specification, or ordered
                name-to-source mapping. Source specifications accept either
                ``source`` or the legacy ``path`` key.

        Returns:
            An unbound :class:`ModelCollection` preserving input order.
        """

        if isinstance(models, ModelCollection):
            return models
        if isinstance(models, Model):
            return ModelCollection((models,))
        if isinstance(models, (str, Path)):
            items = [(None, models)]
        elif isinstance(models, Mapping):
            if "source" in models or "path" in models:
                items = [(models.get("name"), models)]
            else:
                items = list(models.items())
        elif isinstance(models, Sequence):
            items = [(None, value) for value in models]
        else:
            raise TypeError(
                "models must be Model values, model paths, a sequence, or a name-to-model mapping"
            )
        if not items:
            raise ValueError("At least one model is required")
        resolved: list[Model] = []
        seen_names: set[str] = set()
        seen_slugs: set[str] = set()
        for raw_name, value in items:
            if isinstance(value, Model):
                if raw_name is None or str(raw_name) == value.name:
                    model = value
                else:
                    model = value._configured_copy(name=str(raw_name))
            else:
                if isinstance(value, Mapping):
                    configuration = dict(value)
                    raw_source = configuration.pop(
                        "source", configuration.pop("path", None)
                    )
                    explicit_name = configuration.pop("name", None)
                    run_file = configuration.pop(
                        "run_file", configuration.pop("bundle_file", None)
                    )
                else:
                    configuration = {}
                    raw_source = value
                    explicit_name = None
                    run_file = None
                if raw_source is None:
                    raise DatasetValidationError(
                        ValidationIssue(
                            "Model specification is missing source/path",
                            source=str(raw_name) if raw_name is not None else None,
                        )
                    )
                from .model_sources import resolve_model_source

                resolved_source = resolve_model_source(
                    raw_source,
                    name=(
                        str(raw_name)
                        if raw_name is not None
                        else str(explicit_name) if explicit_name is not None else None
                    ),
                    run_file=run_file,
                    progress=True,
                )
                known = {
                    "kind",
                    "task",
                    "source_key",
                    "model_type",
                    "source_dataset_zip",
                    "resolution",
                    "training_dataset",
                    "inference",
                    "device",
                    "folds",
                    "checkpoint",
                    "native_tile_size",
                    "upscale_factor",
                    "input_size",
                    "model_input_size",
                    "batch_size",
                    "workers",
                    "nnunet_tta",
                    "confidence",
                    "postprocess",
                    "foreground_probability_threshold",
                    "prediction_threshold",
                    "sahi_slice_height",
                    "sahi_slice_width",
                    "sahi_overlap",
                    "sahi_overlap_height_ratio",
                    "sahi_overlap_width_ratio",
                    "sahi_postprocess_type",
                    "sahi_postprocess_match_metric",
                    "sahi_postprocess_class_agnostic",
                    "sahi_model_type",
                }
                adapter_settings = dict(configuration.pop("settings", {}) or {})
                adapter_settings.update(
                    {
                        key: configuration.pop(key)
                        for key in list(configuration)
                        if key not in known
                    }
                )
                constructor = dict(resolved_source.options)
                constructor.update(configuration)
                constructor.setdefault(
                    "native_tile_size", resolved_source.geometry.native_tile_size
                )
                constructor.setdefault(
                    "upscale_factor", resolved_source.geometry.upscale_factor
                )
                constructor.setdefault("input_size", resolved_source.geometry.input_size)
                constructor.setdefault(
                    "source_key", resolved_source.source or str(raw_source)
                )
                model = Model(
                    resolved_source.path,
                    name=(
                        str(raw_name)
                        if raw_name is not None
                        else str(explicit_name)
                        if explicit_name is not None
                        else resolved_source.name
                    ),
                    settings=adapter_settings,
                    **constructor,
                )
                if (
                    raw_name is None
                    and explicit_name is None
                    and model.source_key.startswith("wandb:")
                ):
                    model._apply_automatic_name()
            if model.name in seen_names or model.slug in seen_slugs:
                raise DatasetValidationError(
                    ValidationIssue(
                        "Model names must be unique before and after filename normalization",
                        value=model.name,
                    )
                )
            seen_names.add(model.name)
            seen_slugs.add(model.slug)
            resolved.append(model)
        return ModelCollection(tuple(resolved))

    def __repr__(self) -> str:
        return (
            f"Model(name={self.name!r}, kind={self.kind!r}, task={self.task!r}, "
            f"path={str(self.path)!r})"
        )


@dataclass(frozen=True)
class ModelCollection:
    """Ordered, independently configured models.

    Parameters:
        models: Non-empty tuple of unique :class:`Model` values.
    """

    models: tuple[Model, ...]

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("At least one model is required")

    @property
    def names(self) -> tuple[str, ...]:
        """Model names in prediction order."""

        return tuple(model.name for model in self.models)

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self) -> Iterator[Model]:
        return iter(self.models)

    def __getitem__(self, value: int | str) -> Model:
        if isinstance(value, int):
            return self.models[value]
        for model in self.models:
            if model.name == value:
                return model
        raise KeyError(f"Unknown model {value!r}")

    def configure(self, mapping: Mapping[str, Mapping[str, Any]]) -> "ModelCollection":
        """Return an immutable per-model configuration update.

        Names are resolved after loading, making this suitable for standalone
        checkpoints whose display names were inferred from a file or W&B run.
        Unmentioned models are retained as the same objects.

        Parameters:
            mapping: Resolved model names mapped to device, inference, worker,
                SAHI, task, or geometry overrides.

        Returns:
            A new collection. The original collection and models are unchanged.
        """

        if not isinstance(mapping, Mapping):
            raise TypeError("configuration must map resolved model names to mappings")
        unknown = sorted(set(mapping) - set(self.names))
        if unknown:
            raise KeyError(
                f"Unknown model name(s): {', '.join(map(str, unknown))}; "
                f"available: {', '.join(self.names)}"
            )
        allowed = {
            "task",
            "resolution",
            "training_dataset",
            "inference",
            "device",
            "folds",
            "checkpoint",
            "native_tile_size",
            "upscale_factor",
            "input_size",
            "model_input_size",
            "workers",
            "batch_size",
            "nnunet_tta",
            "confidence",
            "postprocess",
            "foreground_probability_threshold",
            "prediction_threshold",
            "sahi_slice_height",
            "sahi_slice_width",
            "sahi_overlap",
            "sahi_overlap_height_ratio",
            "sahi_overlap_width_ratio",
            "sahi_postprocess_type",
            "sahi_postprocess_match_metric",
            "sahi_postprocess_class_agnostic",
            "sahi_model_type",
            "settings",
        }
        configured: list[Model] = []
        for model in self.models:
            if model.name not in mapping:
                configured.append(model)
                continue
            overrides = mapping[model.name]
            if not isinstance(overrides, Mapping):
                raise TypeError(f"Configuration for {model.name!r} must be a mapping")
            unexpected = sorted(set(overrides) - allowed)
            if unexpected:
                raise ValueError(
                    f"Unsupported configuration for {model.name!r}: "
                    + ", ".join(unexpected)
                )
            configured.append(model._configured_copy(overrides=overrides))
        return ModelCollection(tuple(configured))

    def predict(
        self,
        source: Any,
        *,
        split: Literal["train", "val", "test"] | None = None,
        errors: Literal["raise", "skip"] = "raise",
        progress: bool = True,
    ) -> dict[str, PredictionResult]:
        """Run every independently configured model on the same inputs.

        Each item delegates to :meth:`Model.predict`; this collection method
        does not maintain a second inference or batching implementation.

        Parameters:
            source: Any source accepted by :meth:`Model.predict`.
            split: Dataset split. Direct image inputs ignore this value.
            errors: Oversized-input policy applied independently using each
                model's declared native size.
            progress: Show package-managed progress bars.

        Returns:
            Prediction results keyed by model name. Resolution, backend,
            thresholds, device, and SAHI options come from each model.
        """

        return {
            model.name: model.predict(
                source, split=split, errors=errors, progress=progress
            )
            for model in self.models
        }

    def compare(
        self,
        source: Any,
        *,
        split: Literal["train", "val", "test"] = "val",
        save_prediction_plots: bool = False,
        errors: Literal["raise", "skip"] = "raise",
        progress: bool = True,
        destination: str | Path | None = None,
        prediction_cache: bool | str | Path | PredictionCache | None = None,
        trust_legacy_cache: bool = False,
        min_connected_component_area: float | None = None,
        group_by: Callable[[Path], Hashable] | None = None,
    ) -> Any:
        """Compare the configured models on one frozen dataset cohort.

        Verified per-model predictions and metrics are cached below
        ``<dataset>/.cache/evaluations/`` using the dataset cohort, model
        content, and output-affecting prediction/SAHI settings. Execution-only
        choices such as device, worker count, batch size, and installed package
        versions do not invalidate predictions. A matching rerun reuses those
        results even when a different report destination is requested.
        Complete default reports are stored below ``<dataset>/evaluations/``.

        Parameters:
            source: Fixed, on-disk :class:`Dataset` to evaluate.
            split: Dataset split forming the frozen evaluation cohort.
            save_prediction_plots: Write annotated comparison grids under
                ``predictions/`` for the cases the report keeps in
                ``worst_cases``, with at most two model panels per row.
                Cases with neither a reference nor a prediction are
                skipped, since their panels are empty.
            errors: Oversized-image policy. Smaller images are valid for every
                backend. SAHI models accept larger full images and use
                ``native_tile_size`` as their slice geometry. ``"skip"`` omits
                images exceeding any native-inference model's shared size limit
                and records them in report settings.
            progress: Show package-managed progress bars.
            destination: Optional report directory. By default the report is
                content-addressed below ``<dataset>/evaluations/``.
            prediction_cache: Optional prediction-cache override. Omission or
                ``True`` preserves the established dataset-local cache;
                ``False`` disables persistent prediction caching.
            trust_legacy_cache: Reuse and migrate one structurally complete
                legacy semantic prediction cache when its exact model name and
                frozen cohort paths match, even if the old entry lacks model
                bytes/settings identity. Use only when you trust the cache's
                provenance; the default remains verified reuse only.
            min_connected_component_area: Minimum 8-connected predicted
                foreground-component area used for filtered image-level
                presence metrics. ``None`` resolves to the held-out reference
                object-area p10. Raw any-pixel values remain available.
            group_by: Optional callback from each evaluation image path
                to a stable, hashable group label. Adds group-pooled metrics
                and a separate plot; it does not change inference or ranking.

        Returns:
            A task-appropriate comparison result. Comparison space is inferred
            from the dataset and model tasks; all inference settings come from
            each :class:`Model`.
        """

        active = source
        from .dataset import Dataset

        if not isinstance(active, Dataset):
            raise TypeError("Model comparison requires a Dataset")
        if active._plan:
            raise DatasetValidationError(
                "Model comparison requires a fixed on-disk cohort; call dataset.export(...) first"
            )
        from .geometry import validate_collection_geometry

        active = validate_collection_geometry(active, self, split=split, errors=errors)

        if isinstance(active, Dataset) and active.format == "semantic_masks":
            model_tasks = {model.task for model in self.models}
            all_nnunet = all(model.kind == "nnunet" for model in self.models)
            has_semantic_threshold = any(
                model._settings.get("prediction_threshold") is not None
                or model._settings.get("foreground_probability_threshold") is not None
                for model in self.models
            )
            semantic_compatible = model_tasks <= {"segment", "semantic_segment"}
            if not all_nnunet or has_semantic_threshold:
                if not semantic_compatible:
                    raise DatasetValidationError(
                        ValidationIssue(
                            "No common semantic denominator exists for this model collection",
                            value=[
                                {"model": model.name, "kind": model.kind, "task": model.task}
                                for model in self.models
                            ],
                            expected="only segment and semantic_segment tasks",
                            suggestion=(
                                "set task='segment' for YOLO segmentation checkpoints, "
                                "or compare incompatible tasks separately"
                            ),
                        )
                    )
                from .semantic_comparison import compare_semantic_models

                return compare_semantic_models(
                    active,
                    self,
                    split=split,
                    save_prediction_plots=save_prediction_plots,
                    progress=progress,
                    destination=destination,
                    prediction_cache=prediction_cache,
                    trust_legacy_cache=trust_legacy_cache,
                    errors=normalize_errors(errors),
                    min_connected_component_area=min_connected_component_area,
                    group_by=group_by,
                )
            from .semantic_comparison import compare_nnunet_models

            return compare_nnunet_models(
                active,
                self,
                split=split,
                save_prediction_plots=save_prediction_plots,
                progress=progress,
                destination=destination,
                prediction_cache=prediction_cache,
                trust_legacy_cache=trust_legacy_cache,
                errors=normalize_errors(errors),
                min_connected_component_area=min_connected_component_area,
                group_by=group_by,
            )
        if any(model.kind != "ultralytics" for model in self.models):
            raise DatasetValidationError(
                ValidationIssue(
                    "Dataset-native comparison cannot evaluate semantic model folders",
                    value=[model.name for model in self.models if model.kind != "ultralytics"],
                    suggestion="compare nnU-Net folders against a semantic-mask Dataset",
                )
            )
        known_tasks = {model.task for model in self.models if model.task is not None}
        if len(known_tasks) > 1:
            raise DatasetValidationError(
                ValidationIssue(
                    "No implemented native common denominator exists for these model tasks",
                    value=sorted(known_tasks),
                    expected="one shared native task",
                    suggestion="compare same-task models separately",
                )
            )
        if known_tasks and known_tasks != {active.task.value}:
            raise DatasetValidationError(
                ValidationIssue(
                    "Model task does not match the native evaluation dataset",
                    value=sorted(known_tasks),
                    expected=active.task.value,
                )
            )

        from .comparison.engine import _compare_models

        if trust_legacy_cache:
            raise ValueError(
                "trust_legacy_cache is only supported for semantic-mask comparisons"
            )

        return _compare_models(
            active,
            self,
            split=split,
            save_prediction_plots=save_prediction_plots,
            progress=progress,
            destination=destination,
            prediction_cache=prediction_cache,
            errors=normalize_errors(errors),
            min_connected_component_area=min_connected_component_area,
            group_by=group_by,
        )

    def visualize(
        self,
        source: Any,
        *,
        split: Literal["train", "val", "test"] = "val",
        samples: int = 8,
        examples_per_row: int = 1,
        include_empty: bool = False,
        seed: int = 42,
        panel_size: float = 3.0,
        model_title_length: int = 30,
        image_title_length: int = 72,
        progress: bool = True,
        destination: str | Path | None = None,
        errors: Literal["raise", "skip"] = "raise",
    ) -> Any:
        """Render a sampled semantic cohort with the shared comparison grid.

        Parameters:
            source: Semantic-mask :class:`Dataset` to visualize.
            split: Dataset split to sample.
            samples: Maximum number of source images to render.
            examples_per_row: Source images placed in each grid row.
            include_empty: Permit samples whose reference mask is empty.
            seed: Deterministic sampling seed.
            panel_size: Width and height, in inches, of each grid panel.
            model_title_length: Maximum characters per displayed model-title line.
            image_title_length: Maximum displayed source-path length.
            progress: Show package-managed progress bars.
            destination: Optional PNG output path or directory.
            errors: Oversized-image policy shared by the visualized models.

        Returns:
            The rendered Matplotlib figure.
        """

        active = source
        from .dataset import Dataset

        if not isinstance(active, Dataset) or active.format != "semantic_masks":
            raise TypeError(
                "Collection visualization currently requires a semantic-mask Dataset; "
                "use Model.predict for other sources"
            )
        from .geometry import validate_collection_geometry

        active = validate_collection_geometry(active, self, split=split, errors=errors)
        from .semantic_comparison import visualize_nnunet_models

        return visualize_nnunet_models(
            active,
            self,
            split=split,
            samples=samples,
            examples_per_row=examples_per_row,
            include_empty=include_empty,
            seed=seed,
            panel_size=panel_size,
            model_title_length=model_title_length,
            image_title_length=image_title_length,
            progress=progress,
            destination=destination,
        )

    def __repr__(self) -> str:
        return f"ModelCollection(models={self.names!r})"


def _semantic_image_prediction_cache_identity(
    model: Model,
    *,
    inputs: Sequence[ModelInput],
    inference: str,
    resolution: int,
    confidence: float,
    postprocess: float,
    combined_settings: Mapping[str, Any],
    resolved_sahi: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Identify inference from images and functional settings, never labels."""

    from .prediction_cache import prediction_input_fingerprint

    settings = dict(combined_settings)
    for key in tuple(settings):
        if key.startswith("sahi_"):
            settings.pop(key)
    settings.pop("prediction_threshold", None)
    settings.pop("foreground_probability_threshold", None)
    if model._uses_semantic_prediction_threshold():
        # Ultralytics semantic logits and nnU-Net probabilities are produced
        # before these detection/postprocessing compatibility settings.
        settings.pop("confidence", None)
        settings.pop("postprocess", None)
    else:
        settings["confidence"] = confidence
        settings["postprocess"] = postprocess
    return {
        "schema": 3,
        "space": "semantic-image-prediction",
        "input_fingerprint": prediction_input_fingerprint(inputs),
        "model_sha256": model.digest,
        "kind": model.kind,
        "task": (
            "semantic_segment"
            if model._uses_semantic_prediction_threshold()
            else model.task
        ),
        "folds": model.folds,
        "checkpoint": model.checkpoint,
        "upscale_factor": model.upscale_factor,
        "inference": inference,
        "resolution": resolution,
        "nnunet_tta": model.nnunet_tta if model.kind == "nnunet" else None,
        "sahi": dict(resolved_sahi or {}) if inference == "sahi" else None,
        "settings": settings,
    }


def _prepare_prediction_cache_request(
    *,
    model: Model,
    source: Any,
    inputs: Sequence[ModelInput],
    source_task: PredictionTask | None,
    cache_context: Mapping[str, Any],
    prediction_cache: bool | str | Path | PredictionCache,
    backend: str,
    inference: str,
    resolution: int,
    confidence: float,
    postprocess: float,
    combined_settings: Mapping[str, Any],
    resolved_sahi: Mapping[str, Any],
    keep_native: bool,
    identity_override: Mapping[str, Any] | None,
    namespace_override: Literal["predictions", "semantic"] | None,
    compatible_identity_overrides: Sequence[Mapping[str, Any]] = (),
) -> _PredictionCacheRequest | None:
    from .prediction_cache import (
        prediction_cache_key,
        prediction_input_fingerprint,
        resolve_prediction_cache,
    )

    cache = resolve_prediction_cache(
        prediction_cache,
        source=source,
        default=False,
    )
    if cache is None:
        return None

    if identity_override is not None:
        identity = dict(identity_override)
        namespace = namespace_override or "predictions"
        return _PredictionCacheRequest(
            cache=cache,
            key=prediction_cache_key(identity),
            identity=identity,
            namespace=namespace,
            postprocess=postprocess,
            keep_native=keep_native,
            compatible_identities=tuple(
                dict(value) for value in compatible_identity_overrides
            ),
        )

    # An explicit cache belonging to another Dataset is a shared, image-level
    # cache. Use the same semantic namespace as full-image comparison so
    # instance polygons and semantic probabilities can cross reference-label
    # variants without changing ordinary Dataset.compare() defaults.
    from .dataset import Dataset

    shared_dataset_cache = (
        isinstance(source, Dataset)
        and cache.location != PredictionCache.for_dataset(source).location
        and next(
            cache.namespace("semantic").glob("*/raw-result/manifest.json"),
            None,
        )
        is not None
    )
    if shared_dataset_cache:
        image_identity = _semantic_image_prediction_cache_identity(
            model,
            inputs=inputs,
            inference=inference,
            resolution=resolution,
            confidence=confidence,
            postprocess=postprocess,
            combined_settings=combined_settings,
            resolved_sahi=resolved_sahi,
        )
        return _PredictionCacheRequest(
            cache=cache,
            key=prediction_cache_key(image_identity),
            identity=image_identity,
            namespace="semantic",
            postprocess=postprocess,
            keep_native=keep_native,
        )

    semantic_cohort = cache_context.get("semantic_cohort_fingerprint")
    if semantic_cohort:
        image_identity = _semantic_image_prediction_cache_identity(
            model,
            inputs=inputs,
            inference=inference,
            resolution=resolution,
            confidence=confidence,
            postprocess=postprocess,
            combined_settings=combined_settings,
            resolved_sahi=resolved_sahi,
        )
        if model.kind == "nnunet":
            cohort_identity = {
                "schema": 2,
                "space": "nnunet-semantic",
                "cohort": str(semantic_cohort),
                "model_sha256": model.digest,
                "backend": inference,
                "folds": model.folds,
                "checkpoint": model.checkpoint,
                "upscale_factor": model.upscale_factor,
                "resolution": resolution,
                "nnunet_tta": model.nnunet_tta,
                "sahi": dict(resolved_sahi) if inference == "sahi" else None,
            }
            return _PredictionCacheRequest(
                cache=cache,
                key=prediction_cache_key(image_identity),
                identity=image_identity,
                namespace="semantic",
                postprocess=postprocess,
                keep_native=keep_native,
                compatible_identities=(cohort_identity,),
            )
        identity_settings = dict(combined_settings)
        # Probability maps are the threshold-independent inference product.
        # The selected cutoff belongs to evaluation/report identity, not the
        # expensive prediction identity.
        identity_settings.pop("foreground_probability_threshold", None)
        if model._uses_semantic_prediction_threshold():
            identity_settings.pop("prediction_threshold", None)
        identity_settings["confidence"] = confidence
        identity_settings["postprocess"] = postprocess
        cohort_identity = {
            "schema": 2,
            "space": "binary-semantic",
            "cohort": str(semantic_cohort),
            "model_sha256": model.digest,
            "kind": model.kind,
            "task": model.task,
            "folds": model.folds,
            "checkpoint": model.checkpoint,
            "upscale_factor": model.upscale_factor,
            "inference": inference,
            "resolution": resolution,
            "settings": identity_settings,
            **(
                {"nnunet_tta": model.nnunet_tta}
                if model.kind == "nnunet"
                else {}
            ),
        }
        return _PredictionCacheRequest(
            cache=cache,
            key=prediction_cache_key(image_identity),
            identity=image_identity,
            namespace="semantic",
            postprocess=postprocess,
            keep_native=keep_native,
            compatible_identities=(cohort_identity,),
        )

    cohort = cache_context.get("cohort")
    prediction_task = model.task or source_task
    if (
        cohort is not None
        and prediction_task == cohort.task
        and prediction_task != "semantic_segment"
        and len(inputs) == len(cohort.records)
        and all(
            value.image_id == record.image_id
            for value, record in zip(inputs, cohort.records)
        )
    ):
        adapter_settings = {
            "inference": inference,
            **dict(combined_settings),
        }
        adapter_settings.pop("confidence", None)
        adapter_settings.pop("postprocess", None)
        identity = {
            "model_sha256": model.digest,
            "cohort_fingerprint": cohort.fingerprint,
            "task": cohort.task,
            "classes": cohort.classes,
            "backend": backend,
            "resolution": resolution,
            "confidence_floor": confidence,
            "settings": adapter_settings,
        }
        payload = {
            **identity,
            "postprocess_thresholds": (postprocess,),
        }
        return _PredictionCacheRequest(
            cache=cache,
            key=prediction_cache_key(identity),
            identity=identity,
            namespace="predictions",
            cohort=cohort,
            package_payload=payload,
            postprocess=postprocess,
            keep_native=keep_native,
        )

    identity_settings = dict(combined_settings)
    if (
        prediction_task == "semantic_segment"
        or model._uses_semantic_prediction_threshold()
    ):
        identity_settings.pop("foreground_probability_threshold", None)
        identity_settings.pop("prediction_threshold", None)
    identity_settings["confidence"] = confidence
    identity_settings["postprocess"] = postprocess
    identity = {
        "schema": 1,
        "space": "raw-prediction-result",
        "input_fingerprint": prediction_input_fingerprint(inputs),
        "model_sha256": model.digest,
        "kind": model.kind,
        "task": prediction_task,
        "source_task": source_task,
        "backend": backend,
        "resolution": resolution,
        "inference": inference,
        "folds": model.folds,
        "checkpoint": model.checkpoint,
        "upscale_factor": model.upscale_factor,
        "nnunet_tta": model.nnunet_tta,
        "keep_native": keep_native,
        "settings": identity_settings,
    }
    return _PredictionCacheRequest(
        cache=cache,
        key=prediction_cache_key(identity),
        identity=identity,
        namespace=namespace_override
        or str(cache_context.get("namespace") or "predictions"),
        postprocess=postprocess,
        keep_native=keep_native,
    )


def _load_prediction_cache_request(
    request: _PredictionCacheRequest | None,
    *,
    inputs: Sequence[ModelInput],
    model: Model,
    backend: str,
    source_size_policy: Mapping[str, Any],
    device: str | None,
    batch_size: int,
    resolution: int,
    confidence: float,
    postprocess: float,
    combined_settings: Mapping[str, Any],
    resolved_sahi: Mapping[str, Any],
    foreground_probability_threshold: float | None,
    require_probability_maps: bool,
) -> PredictionResult | None:
    if request is None:
        return None

    if request.package_payload is not None:
        from .comparison.cache import load_package_cache

        root = request.cache.entry(request.key, namespace="predictions")
        loaded, shards, complete = load_package_cache(
            root,
            request.cohort,
            (request.postprocess,),
            progress=False,
        )
        if complete:
            by_image = loaded[float(request.postprocess)]
            records = tuple(
                ImagePrediction(
                    image_id=value.image_id,
                    image_path=value.image_path,
                    relative_path=value.relative_path,
                    width=value.width,
                    height=value.height,
                    objects=tuple(by_image[value.image_id]),
                    metadata={"backend": backend},
                )
                for value in inputs
            )
            return PredictionResult(
                model_name=model.name,
                model_kind=model.kind,
                task=request.cohort.task,
                backend=backend,
                records=records,
                inference_seconds=0.0,
                settings=_cached_result_settings(
                    device=device,
                    batch_size=batch_size,
                    resolution=resolution,
                    confidence=confidence,
                    postprocess=postprocess,
                    combined_settings=combined_settings,
                    resolved_sahi=resolved_sahi,
                    source_size_policy=source_size_policy,
                ),
                cache_info={
                    "status": "hit",
                    "verified": True,
                    "key": request.key,
                    "namespace": "predictions",
                    "location": str(root),
                    "shards": shards,
                },
            )

    cached = request.cache.load(
        request.key,
        namespace=request.namespace,
        identity=request.identity,
        inputs=inputs,
    )
    if cached is None:
        from .prediction_cache import prediction_cache_key

        for compatible_identity in request.compatible_identities:
            compatible_key = prediction_cache_key(compatible_identity)
            compatible = request.cache.load(
                compatible_key,
                namespace=request.namespace,
                identity=compatible_identity,
                inputs=inputs,
            )
            if compatible is None:
                continue
            promoted = request.cache.save(
                request.key,
                compatible,
                namespace=request.namespace,
                identity=request.identity,
                inputs=inputs,
            )
            cached = replace(
                promoted,
                cache_info={
                    **promoted.cache_info,
                    "status": "compatible-hit",
                    "compatible_key": compatible_key,
                },
            )
            break
    if cached is None and request.namespace == "semantic":
        cached = request.cache.find_image_compatible(
            namespace=request.namespace,
            identity=request.identity,
            inputs=inputs,
        )
    if cached is not None:
        if request.keep_native and any(
            record.native_mask is None for record in cached.records
        ):
            cached = None
        elif not request.keep_native and any(
            record.native_mask is not None for record in cached.records
        ):
            cached = replace(
                cached,
                records=tuple(
                    replace(record, native_mask=None)
                    for record in cached.records
                ),
            )
        if cached is not None and cached.task == "semantic_segment":
            missing_probabilities = any(
                record.foreground_probability is None
                for record in cached.records
            )
            if require_probability_maps and missing_probabilities:
                cached = None
            elif foreground_probability_threshold is not None:
                if missing_probabilities:
                    cached = None
                else:
                    cached = replace(
                        cached,
                        records=tuple(
                            replace(
                                record,
                                mask=(
                                    np.asarray(record.foreground_probability)
                                    >= foreground_probability_threshold
                                ).astype(np.uint8),
                            )
                            for record in cached.records
                        ),
                        settings={
                            **cached.settings,
                            "foreground_probability_threshold": (
                                foreground_probability_threshold
                            ),
                            "prediction_threshold": foreground_probability_threshold,
                        },
                    )
        elif cached is not None:
            cached = replace(
                cached,
                records=tuple(
                    replace(
                        record,
                        objects=tuple(
                            value
                            for value in record.objects
                            if float(value.score) >= confidence
                        ),
                    )
                    for record in cached.records
                ),
                settings={
                    **cached.settings,
                    "confidence": confidence,
                    "prediction_threshold": confidence,
                },
            )
    if cached is not None:
        return replace(cached, model_name=model.name)

    if (
        request.namespace == "semantic"
        and not request.keep_native
        and not require_probability_maps
        and foreground_probability_threshold is None
    ):
        from .prediction_cache import prediction_cache_key

        legacy_requests = (request,) + tuple(
            replace(
                request,
                key=prediction_cache_key(identity),
                identity=dict(identity),
                compatible_identities=(),
            )
            for identity in request.compatible_identities
        )
        for legacy_request in legacy_requests:
            legacy = _load_legacy_semantic_prediction(
                legacy_request,
                inputs=inputs,
                model=model,
                backend=backend,
                settings=_cached_result_settings(
                    device=device,
                    batch_size=batch_size,
                    resolution=resolution,
                    confidence=confidence,
                    postprocess=postprocess,
                    combined_settings=combined_settings,
                    resolved_sahi=resolved_sahi,
                    source_size_policy=source_size_policy,
                ),
            )
            if legacy is None:
                continue
            promoted = request.cache.save(
                request.key,
                legacy,
                namespace=request.namespace,
                identity=request.identity,
                inputs=inputs,
            )
            return replace(
                promoted,
                cache_info={
                    **promoted.cache_info,
                    "status": "legacy-hit",
                    "compatible_key": legacy_request.key,
                },
            )
    return None


def _save_prediction_cache_request(
    request: _PredictionCacheRequest | None,
    *,
    inputs: Sequence[ModelInput],
    result: PredictionResult,
) -> PredictionResult:
    if request is None:
        return result
    if (
        request.package_payload is not None
        and result.task == request.cohort.task
        and all(record.mask is None and record.native_mask is None for record in result)
    ):
        from .comparison.cache import save_package_cache

        root = request.cache.entry(request.key, namespace="predictions")
        save_package_cache(
            root,
            request.cohort,
            request.package_payload,
            {
                float(request.postprocess): {
                    record.image_id: list(record.objects)
                    for record in result
                }
            },
            progress=False,
        )
        return replace(
            result,
            cache_info={
                "status": "fresh",
                "verified": True,
                "key": request.key,
                "namespace": "predictions",
                "location": str(root),
            },
        )
    return request.cache.save(
        request.key,
        result,
        namespace=request.namespace,
        identity=request.identity,
        inputs=inputs,
    )


def _cached_result_settings(
    *,
    device: str | None,
    batch_size: int,
    resolution: int,
    confidence: float,
    postprocess: float,
    combined_settings: Mapping[str, Any],
    resolved_sahi: Mapping[str, Any],
    source_size_policy: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "device": device,
        "resolution": resolution,
        "confidence": confidence,
        "postprocess": postprocess,
        "batch_size": batch_size,
        **dict(combined_settings),
        **dict(resolved_sahi),
        "source_size_policy": dict(source_size_policy),
    }


def _load_legacy_semantic_prediction(
    request: _PredictionCacheRequest,
    *,
    inputs: Sequence[ModelInput],
    model: Model,
    backend: str,
    settings: Mapping[str, Any],
) -> PredictionResult | None:
    if not model._uses_semantic_prediction_threshold():
        return None
    root = request.cache.entry(request.key, namespace="semantic")
    metadata_path = root / "evaluation.json"
    predictions = root / "predictions"
    if not metadata_path.is_file() or not predictions.is_dir():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    from .prediction_cache import prediction_cache_key

    stored_identity = metadata.get("cache_identity") if isinstance(metadata, dict) else None
    if not isinstance(stored_identity, dict) or prediction_cache_key(stored_identity) != request.key:
        return None
    expected = {f"{value.image_id}.png" for value in inputs}
    actual = {path.name for path in predictions.glob("*.png") if path.is_file()}
    if actual != expected:
        return None
    records: list[ImagePrediction] = []
    for value in inputs:
        try:
            with Image.open(predictions / f"{value.image_id}.png") as opened:
                mask = np.asarray(opened.copy())
        except OSError:
            return None
        if mask.ndim != 2 or mask.shape != (value.height, value.width):
            return None
        records.append(
            ImagePrediction(
                image_id=value.image_id,
                image_path=value.image_path,
                relative_path=value.relative_path,
                width=value.width,
                height=value.height,
                mask=mask,
                metadata={"backend": backend, "legacy_semantic_cache": True},
            )
        )
    return PredictionResult(
        model_name=model.name,
        model_kind=model.kind,
        task="semantic_segment",
        backend=backend,
        records=tuple(records),
        inference_seconds=float(metadata.get("inference_seconds", 0.0)),
        settings=dict(settings),
    )


def normalize_model_inputs(
    source: Any,
    *,
    split: str | None,
    progress: bool = False,
) -> tuple[tuple[ModelInput, ...], PredictionTask | None, dict[str, Any]]:
    """Normalize public prediction sources without reading their annotations."""

    if isinstance(source, ModelInput):
        return (source,), None, {}
    if _is_model_input_sequence(source):
        return tuple(source), None, {}

    from .dataset import Dataset

    if isinstance(source, Dataset) and source.format == "semantic_masks":
        selected_split = split or "val"
        if selected_split not in source.splits:
            raise ValueError(
                f"Unknown semantic-mask split {selected_split!r}; "
                f"available splits are {source.splits}"
            )
        from .semantic_comparison import _freeze_cohort

        cases, cohort_fingerprint = _freeze_cohort(
            source, selected_split, progress=progress
        )
        return (
            tuple(
                ModelInput(
                    image_id=case.case_id,
                    image_path=case.image_path,
                    width=case.width,
                    height=case.height,
                    relative_path=case.relative_path.as_posix(),
                    mask_path=case.mask_path,
                    image_sha256=case.image_sha256,
                )
                for case in cases
            ),
            "semantic_segment",
            {
                "semantic_cohort_fingerprint": cohort_fingerprint,
                "namespace": "semantic",
            },
        )

    from .comparison.types import Cohort

    if isinstance(source, Cohort):
        return (
            tuple(
                ModelInput(
                    image_id=record.image_id,
                    image_path=record.image_path,
                    width=record.width,
                    height=record.height,
                    relative_path=record.relative_path,
                    image_sha256=record.image_sha256,
                )
                for record in source.records
            ),
            source.task,
            {"cohort": source, "namespace": "predictions"},
        )

    if isinstance(source, Dataset):
        if source._plan:
            raise DatasetValidationError(
                "Model prediction requires fixed on-disk images; export the plan first"
            )
        from .comparison.cohort import freeze_cohort

        cohort = freeze_cohort(source, split or "val", progress=progress)
        return normalize_model_inputs(cohort, split=None)

    paths: list[Path]
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            paths = sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            paths = [path]
    elif isinstance(source, Sequence):
        paths = [Path(value).expanduser().resolve() for value in source]
    else:
        raise TypeError(
            "source must be an image, directory, image sequence, Dataset, "
            "Cohort, or ModelInput sequence"
        )
    inputs: list[ModelInput] = []
    for index, path in enumerate(paths):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise DatasetValidationError(
                ValidationIssue(
                    "Prediction input is not a supported image",
                    value=str(path),
                    expected=f"one of {sorted(IMAGE_SUFFIXES)}",
                )
            )
        with Image.open(path) as opened:
            width, height = opened.size
        inputs.append(
            ModelInput(
                image_id=f"image_{index:06d}",
                image_path=path,
                width=width,
                height=height,
                relative_path=path.name,
            )
        )
    return tuple(inputs), None, {}


def _is_model_input_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, Path))
        and bool(value)
        and all(isinstance(item, ModelInput) for item in value)
    )


def _normalize_folds(value: Any) -> tuple[str, ...]:
    raw_folds = (value,) if isinstance(value, (str, int)) else tuple(value)
    folds: list[str] = []
    for raw in raw_folds:
        fold = str(raw).strip()
        if fold != "all":
            try:
                if int(fold) < 0 or str(int(fold)) != fold:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(
                    f"Invalid nnU-Net fold {raw!r}; expected a non-negative integer or 'all'"
                ) from exc
        if fold not in folds:
            folds.append(fold)
    if not folds:
        raise ValueError("At least one nnU-Net fold is required")
    return tuple(folds)


def _validate_nnunet_dataset(path: Path, name: str) -> None:
    try:
        dataset = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetValidationError(
            ValidationIssue(
                "Incomplete nnU-Net model folder",
                source=name,
                value=str(path),
                expected="dataset.json",
            )
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            f"Unreadable nnU-Net dataset.json for {name}: {exc}"
        ) from exc
    if dataset.get("file_ending") != ".png":
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic-mask prediction requires an nnU-Net PNG model",
                source=name,
                value=dataset.get("file_ending"),
                expected="file_ending='.png'",
            )
        )
    labels = dataset.get("labels") or {}
    values: set[int] = set()
    try:
        for value in labels.values():
            if isinstance(value, list):
                values.update(int(item) for item in value)
            else:
                values.add(int(value))
    except (TypeError, ValueError):
        values = set()
    if values != {0, 1}:
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic-mask prediction requires binary nnU-Net labels",
                source=name,
                value=labels,
                expected="background=0 and one foreground label=1",
            )
        )


def _combined_sha256(paths: tuple[Path, ...], *, relative_to: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(relative_to).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _task_from_args(checkpoint: Path) -> PredictionTask | None:
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        path = directory / "args.yaml"
        if not path.is_file():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, TypeError, yaml.YAMLError):
            continue
        task = str(payload.get("task", "")).lower()
        if task in {
            "detect",
            "segment",
            "pose",
            "polo",
            "locate",
            "semantic_segment",
            "semantic",
        }:
            return _normalize_prediction_task(task)
    return None


def _normalize_prediction_task(value: Any) -> PredictionTask | None:
    if value in {None, "auto"}:
        return None
    normalized = str(value).strip().lower()
    aliases = {"locate": "polo", "semantic": "semantic_segment"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"detect", "segment", "pose", "polo", "semantic_segment"}:
        raise ValueError(
            "task must be 'auto', 'detect', 'segment', 'pose', 'polo'/'locate', "
            "or 'semantic_segment'/'semantic'"
        )
    return normalized  # type: ignore[return-value]


def _training_dataset_from_args(checkpoint: Path) -> str | None:
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        path = directory / "args.yaml"
        if not path.is_file():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data = payload.get("data")
            if not data:
                continue
            candidate = Path(str(data)).expanduser()
            if not candidate.is_absolute():
                candidate = (path.parent / candidate).resolve()
            return str(candidate)
        except (OSError, TypeError, yaml.YAMLError):
            continue
    return None


def _shorten_middle(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    left = (maximum - 1) // 2
    right = maximum - 1 - left
    return f"{value[:left]}…{value[-right:]}"


def _draw_object_predictions(axis: Any, values: tuple[Any, ...]) -> None:
    import matplotlib.pyplot as plt

    color = "#00D084"
    for value in values:
        if value.bbox is not None:
            x1, y1, x2, y2 = value.bbox
            axis.add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    color=color,
                    linewidth=1.5,
                )
            )
        polygons = value.polygons or ([value.polygon] if value.polygon else [])
        for raw_polygon in polygons:
            polygon = np.asarray(raw_polygon, dtype=float)
            axis.plot(
                np.r_[polygon[:, 0], polygon[0, 0]],
                np.r_[polygon[:, 1], polygon[0, 1]],
                color=color,
                linewidth=1.5,
            )
        if value.point is not None:
            axis.scatter(
                value.point[0],
                value.point[1],
                s=25,
                color=color,
                edgecolor="white",
                linewidth=0.6,
            )
        if value.keypoints:
            keypoints = np.asarray(
                [
                    point[:2]
                    for point in value.keypoints
                    if len(point) < 3 or point[2] is None or point[2] > 0
                ],
                dtype=float,
            )
            if len(keypoints):
                axis.scatter(
                    keypoints[:, 0],
                    keypoints[:, 1],
                    s=14,
                    color=color,
                )
