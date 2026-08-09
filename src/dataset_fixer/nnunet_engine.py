"""In-process nnU-Net v2 inference for SAHI tiles.

This engine replaces one ``nnUNetv2_predict_from_modelfolder`` subprocess per
tile batch. It keeps every numerical step nnU-Net itself performs — the official
preprocessor, sliding-window slicers, Gaussian patch weighting, mirroring TTA,
fold averaging, and probability conversion — but reorganizes the work so that:

* each fold's weights are loaded once instead of once per tile,
* equally shaped tiles are evaluated as real network minibatches, and
* probabilities are returned in memory instead of via a PNG/NPZ pair per tile.

The reorganization is exact. Patches are always the configured patch size and
nnU-Net's 2D architectures normalize per sample (instance normalization), so
evaluating N patches in one forward pass computes the same function as N passes
of one patch. Only floating-point summation order may differ.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .errors import DatasetValidationError, ValidationIssue


ACCELERATOR_BATCH_CEILING = 16
CPU_BATCH_CEILING = 4

_OUT_OF_MEMORY_MARKERS = (
    "out of memory",
    "can't allocate memory",
    "cannot allocate memory",
    "failed to allocate",
    "insufficient memory",
    "mps backend out of memory",
)


@dataclass
class EngineTelemetry:
    """Resolved execution facts recorded in prediction and comparison manifests."""

    engine: str = "in-process-minibatched"
    device: str = "cpu"
    plan_batch_size: int = 1
    requested_batch_size: int = 1
    resolved_batch_size: int = 1
    oom_retries: int = 0
    tiles: int = 0
    sources: int = 0
    folds: tuple[str, ...] = ()
    weight_loads: int = 0
    forward_passes: int = 0
    tta: bool = True
    workers: int = 1
    preprocess_seconds: float = 0.0
    inference_seconds: float = 0.0
    conversion_seconds: float = 0.0
    stitch_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "nnunet_execution_engine": self.engine,
            "nnunet_device": self.device,
            "nnunet_plan_batch_size": self.plan_batch_size,
            "nnunet_requested_batch_size": self.requested_batch_size,
            "nnunet_resolved_batch_size": self.resolved_batch_size,
            "nnunet_oom_retries": self.oom_retries,
            "nnunet_tiles": self.tiles,
            "nnunet_sources": self.sources,
            "nnunet_folds": list(self.folds),
            "nnunet_weight_loads": self.weight_loads,
            "nnunet_forward_passes": self.forward_passes,
            "nnunet_tta": self.tta,
            "nnunet_workers": self.workers,
            "nnunet_phase_seconds": {
                "preprocess": round(self.preprocess_seconds, 4),
                "inference": round(self.inference_seconds, 4),
                "probability_conversion": round(self.conversion_seconds, 4),
                "stitch": round(self.stitch_seconds, 4),
            },
        }


class NnUNetSession:
    """A loaded nnU-Net model that predicts batches of equally shaped tiles."""

    def __init__(
        self,
        *,
        model_folder: Path,
        folds: Sequence[str],
        checkpoint: str,
        device: str,
        workers: int,
        batch_size: int = -1,
        use_tta: bool = True,
    ) -> None:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        self._torch = torch
        self.device_name = device
        self.workers = max(1, int(workers))
        self.use_tta = bool(use_tta)
        self.folds = tuple(str(fold) for fold in folds)
        torch_device = torch.device(device)
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=self.use_tta,
            # nnU-Net only supports keeping result arrays on the device for
            # CUDA; its own constructor downgrades every other device.
            perform_everything_on_device=device == "cuda",
            device=torch_device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(model_folder),
            [int(fold) if fold != "all" else fold for fold in self.folds],
            checkpoint_name=checkpoint,
        )
        self._predictor = predictor
        self._network_on_device = False
        self._loaded_fold: int | None = None
        self.weight_loads = 0
        self.forward_passes = 0
        self.patch_size = tuple(int(value) for value in predictor.configuration_manager.patch_size)
        self.plan_batch_size = int(predictor.configuration_manager.batch_size)
        self.num_classes = int(predictor.label_manager.num_segmentation_heads)
        self.fold_count = len(predictor.list_of_parameters)
        ceiling = (
            ACCELERATOR_BATCH_CEILING if device in {"cuda", "mps"} else CPU_BATCH_CEILING
        )
        self.requested_batch_size = (
            max(1, min(self.plan_batch_size, ceiling))
            if batch_size == -1
            else max(1, int(batch_size))
        )
        self.resolved_batch_size = self.requested_batch_size
        self.oom_retries = 0

    @property
    def results_device(self) -> Any:
        torch = self._torch
        return (
            self._predictor.device
            if self._predictor.perform_everything_on_device
            else torch.device("cpu")
        )

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Run nnU-Net's official preprocessing for one natural-image tile."""

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise DatasetValidationError(
                ValidationIssue(
                    "nnU-Net tile input must be an RGB or RGBA image array",
                    value=tuple(array.shape),
                    expected="(height, width, 3) or (height, width, 4)",
                )
            )
        # Matches NaturalImage2DIO.read_images: channels first, one z slice,
        # float32, with nnU-Net's synthetic 2D spacing.
        data = array.transpose(2, 0, 1)[:, None].astype(np.float32, copy=True)
        properties: dict[str, Any] = {"spacing": (999, 1, 1)}
        preprocessor = self._predictor.configuration_manager.preprocessor_class(verbose=False)
        prepared, _, properties = preprocessor.run_case_npy(
            data,
            None,
            properties,
            self._predictor.plans_manager,
            self._predictor.configuration_manager,
            self._predictor.dataset_json,
        )
        return prepared, properties

    def preprocess_many(
        self,
        images: Sequence[np.ndarray],
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        if len(images) <= 1 or self.workers <= 1:
            return [self.preprocess(image) for image in images]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self.preprocess, images))

    def predict_logits(
        self,
        prepared: Sequence[np.ndarray],
        *,
        on_batch: Callable[[int], None] | None = None,
    ) -> list[Any]:
        """Predict fold-averaged logits for equally shaped preprocessed tiles.

        Folds are the outermost loop: each fold's weights are loaded once and
        then applied to every tile, instead of once per tile.
        """

        torch = self._torch
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian

        if not prepared:
            return []
        shapes = {tuple(value.shape) for value in prepared}
        if len(shapes) != 1:
            raise DatasetValidationError(
                ValidationIssue(
                    "nnU-Net minibatches require equally shaped preprocessed tiles",
                    value=sorted(str(shape) for shape in shapes),
                )
            )
        self._ensure_network_on_device()
        results_device = self.results_device
        stacked = torch.from_numpy(np.stack(prepared, axis=0))
        padded, revert_padding = pad_nd_image(
            stacked, self.patch_size, "constant", {"value": 0}, True, None
        )
        # padded is (batch, channels, *spatial); nnU-Net's slicers are built for
        # a single case, so they address the spatial dimensions from index 1.
        slicers = self._predictor._internal_get_sliding_window_slicers(padded.shape[2:])
        gaussian = compute_gaussian(
            tuple(self.patch_size),
            sigma_scale=1.0 / 8,
            value_scaling_factor=10,
            device=results_device,
        )

        count = len(prepared)
        totals: list[Any] = [None] * count
        with torch.inference_mode():
            for fold_index in range(self.fold_count):
                self._load_fold(fold_index)
                logits = torch.zeros(
                    (count, self.num_classes, *padded.shape[2:]),
                    dtype=torch.half,
                    device=results_device,
                )
                weights = torch.zeros(
                    padded.shape[2:], dtype=torch.half, device=results_device
                )
                for slicer in slicers:
                    patches = torch.clone(
                        padded[(slice(None), *slicer)],
                        memory_format=torch.contiguous_format,
                    ).to(self._predictor.device)
                    predicted = self._forward(patches, on_batch=on_batch)
                    logits[(slice(None), *slicer)] += (
                        predicted.to(results_device) * gaussian
                    )
                    weights[slicer[1:]] += gaussian
                torch.div(logits, weights, out=logits)
                if torch.any(torch.isinf(logits)):
                    raise DatasetValidationError(
                        "nnU-Net produced non-finite logits; reduce the Gaussian scaling factor"
                    )
                logits = logits[(slice(None), slice(None), *revert_padding[2:])]
                for index in range(count):
                    tile_logits = logits[index].to("cpu").clone()
                    totals[index] = (
                        tile_logits
                        if totals[index] is None
                        else totals[index] + tile_logits
                    )
                del logits, weights
            if self.fold_count > 1:
                for index in range(count):
                    totals[index] /= self.fold_count
        # Clone out of inference mode: the official probability converter is
        # free to update these tensors in place.
        return [value.clone() for value in totals]

    def to_probabilities(
        self,
        logits: Any,
        properties: dict[str, Any],
    ) -> np.ndarray:
        """Convert logits to source-shaped probabilities with the official API."""

        from nnunetv2.inference.export_prediction import (
            convert_predicted_logits_to_segmentation_with_correct_shape,
        )

        _, probabilities = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits,
            self._predictor.plans_manager,
            self._predictor.configuration_manager,
            self._predictor.label_manager,
            properties,
            return_probabilities=True,
        )
        return np.asarray(probabilities, dtype=np.float32)

    def to_probabilities_many(
        self,
        pairs: Sequence[tuple[Any, dict[str, Any]]],
    ) -> list[np.ndarray]:
        if len(pairs) <= 1 or self.workers <= 1:
            return [self.to_probabilities(logits, props) for logits, props in pairs]
        # The official converter sets and restores the global torch thread count,
        # which races between workers; restore it once the pool has drained.
        threads = self._torch.get_num_threads()
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                return list(pool.map(lambda pair: self.to_probabilities(*pair), pairs))
        finally:
            self._torch.set_num_threads(threads)

    def release(self) -> None:
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        from nnunetv2.utilities.helpers import empty_cache

        compute_gaussian.cache_clear()
        empty_cache(self._predictor.device)

    def _ensure_network_on_device(self) -> None:
        if self._network_on_device:
            return
        self._predictor.network = self._predictor.network.to(self._predictor.device)
        self._predictor.network.eval()
        self._network_on_device = True

    def _load_fold(self, fold_index: int) -> None:
        if self._loaded_fold == fold_index:
            return
        from torch._dynamo import OptimizedModule

        parameters = self._predictor.list_of_parameters[fold_index]
        network = self._predictor.network
        if isinstance(network, OptimizedModule):
            network._orig_mod.load_state_dict(parameters)
        else:
            network.load_state_dict(parameters)
        self._loaded_fold = fold_index
        self.weight_loads += 1

    def _forward(self, patches: Any, *, on_batch: Callable[[int], None] | None) -> Any:
        """Run mirroring TTA over minibatches, backing off on out-of-memory."""

        torch = self._torch
        from nnunetv2.utilities.helpers import dummy_context

        total = patches.shape[0]
        device_type = self._predictor.device.type
        outputs: list[Any] = []
        start = 0
        while start < total:
            size = min(self.resolved_batch_size, total - start)
            chunk = patches[start : start + size]
            try:
                context = (
                    torch.autocast(device_type, enabled=True)
                    if device_type == "cuda"
                    else dummy_context()
                )
                with context:
                    predicted = self._predictor._internal_maybe_mirror_and_predict(chunk)
            except (RuntimeError, MemoryError) as error:
                if not _is_out_of_memory(error) or self.resolved_batch_size <= 1:
                    raise
                self.resolved_batch_size = max(1, self.resolved_batch_size // 2)
                self.oom_retries += 1
                self.release()
                continue
            outputs.append(predicted)
            self.forward_passes += 1
            if on_batch is not None:
                on_batch(size)
            start += size
        return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]


def load_session(
    *,
    model_folder: Path,
    folds: Sequence[str],
    checkpoint: str,
    device: str,
    workers: int,
    batch_size: int = -1,
    use_tta: bool = True,
) -> NnUNetSession:
    """Load one nnU-Net model for a whole prediction run."""

    require_nnunet()
    return NnUNetSession(
        model_folder=Path(model_folder),
        folds=folds,
        checkpoint=checkpoint,
        device=device,
        workers=workers,
        batch_size=batch_size,
        use_tta=use_tta,
    )


def require_nnunet() -> None:
    """Privately import-probe the coherent Python 3.12 nnU-Net stack."""

    import importlib
    import importlib.metadata
    import importlib.util

    if getattr(require_nnunet, "_succeeded", False):
        return
    if importlib.util.find_spec("nnunetv2") is None:
        raise ImportError(
            "nnU-Net prediction was requested but nnunetv2 is not installed; "
            "reinstall dataset-fixer"
        )
    packages = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit-image": "skimage",
        "batchgeneratorsv2": "batchgeneratorsv2",
        "nnunetv2": "nnunetv2",
    }
    install = (
        "python -m pip install "
        "\"numpy>=1.24\" \"scipy>=1.11.4\" "
        "\"scikit-image>=0.19.3\" \"batchgeneratorsv2>=0.3.2\" "
        "\"nnunetv2>=2.8.1,<3\""
    )
    try:
        for module in packages.values():
            importlib.import_module(module)
        # Trainer discovery imports the same morphology/scientific path that
        # exposed the Colab NumPy ``_blas_supports_fpe`` ABI mismatch.
        importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
        )
    except Exception as exc:
        versions = []
        for distribution in packages:
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = "missing"
            versions.append(f"{distribution}={version}")
        restart = (
            " In Colab, start a fresh runtime and run the installation cell before "
            "importing NumPy, SciPy, scikit-image, batchgeneratorsv2, or nnU-Net."
        )
        raise RuntimeError(
            "The nnU-Net scientific stack cannot be imported coherently "
            f"({', '.join(versions)}). Install a compatible stack with:\n"
            f"{install}.{restart} "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    setattr(require_nnunet, "_succeeded", True)


def _is_out_of_memory(error: BaseException) -> bool:
    if error.__class__.__name__ in {"OutOfMemoryError", "OutOfMemoryError_"}:
        return True
    message = str(error).lower()
    return any(marker in message for marker in _OUT_OF_MEMORY_MARKERS)
