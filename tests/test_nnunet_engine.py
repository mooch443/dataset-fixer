"""Unit tests for the in-process nnU-Net minibatch engine.

These exercise the engine's own scheduling — batch sizing, out-of-memory
backoff, fold ordering, and TTA wiring — against a stub network, so they run
without a trained model. Numerical equivalence with sequential prediction is
covered separately against a real checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset_fixer.errors import DatasetValidationError
from dataset_fixer.nnunet_engine import (
    ACCELERATOR_BATCH_CEILING,
    CPU_BATCH_CEILING,
    EngineTelemetry,
    NnUNetSession,
    _initialize_predictor_from_trained_model_folder,
    _is_out_of_memory,
)

torch = pytest.importorskip("torch")


class StubPredictor:
    """The slice of nnUNetPredictor the engine's batching path relies on."""

    def __init__(self, *, classes: int = 2, patch: tuple[int, int] = (8, 8)) -> None:
        self.device = torch.device("cpu")
        self.perform_everything_on_device = False
        self.classes = classes
        self.patch = patch
        self.network = torch.nn.Identity()
        self.list_of_parameters = [{}]
        self.loaded_folds: list[int] = []
        self.batch_sizes: list[int] = []
        self.mirrored = 0

    def _internal_get_sliding_window_slicers(self, image_size):
        return [(slice(None), 0, slice(0, self.patch[0]), slice(0, self.patch[1]))]

    def _internal_maybe_mirror_and_predict(self, x):
        self.batch_sizes.append(int(x.shape[0]))
        self.mirrored += 1
        return torch.ones(
            (x.shape[0], self.classes, *x.shape[2:]),
            dtype=torch.float32,
        )


class TrainerResolvingPredictor:
    def __init__(self, trainer_name: str) -> None:
        self.trainer_name = trainer_name
        self.trainer_class = None
        self.initialization = None

    def initialize_from_trained_model_folder(
        self,
        model_folder: str,
        folds: list[int | str],
        *,
        checkpoint_name: str,
    ) -> None:
        from nnunetv2.inference import predict_from_raw_data

        self.trainer_class = predict_from_raw_data.recursive_find_trainer_class_by_name(
            self.trainer_name
        )
        self.initialization = (model_folder, folds, checkpoint_name)


def _session(
    predictor: StubPredictor,
    *,
    device: str = "cpu",
    batch_size: int = 4,
    folds: tuple[str, ...] = ("0",),
) -> NnUNetSession:
    session = NnUNetSession.__new__(NnUNetSession)
    session._torch = torch
    session._predictor = predictor
    session._network_on_device = True
    session._loaded_fold = None
    session.device_name = device
    session.workers = 1
    session.use_tta = True
    session.folds = folds
    session.weight_loads = 0
    session.forward_passes = 0
    session.patch_size = predictor.patch
    session.plan_batch_size = 50
    session.num_classes = predictor.classes
    session.fold_count = len(predictor.list_of_parameters)
    session.requested_batch_size = batch_size
    session.resolved_batch_size = batch_size
    session.oom_retries = 0

    def load_fold(index: int) -> None:
        if session._loaded_fold == index:
            return
        predictor.loaded_folds.append(index)
        session._loaded_fold = index
        session.weight_loads += 1

    session._load_fold = load_fold
    return session


def test_missing_training_length_trainer_uses_base_inference_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nnunetv2.inference import predict_from_raw_data
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    def missing(trainer_name: str):
        raise RuntimeError(f"Could not find requested nnunet trainer {trainer_name}")

    monkeypatch.setattr(
        predict_from_raw_data,
        "recursive_find_trainer_class_by_name",
        missing,
    )
    predictor = TrainerResolvingPredictor("nnUNetTrainer_184epochs")

    with pytest.warns(RuntimeWarning, match="training duration only"):
        _initialize_predictor_from_trained_model_folder(
            predictor,
            model_folder=Path("/models/custom"),
            folds=("0", "all"),
            checkpoint="checkpoint_best.pth",
        )

    trainer = predictor.trainer_class
    assert trainer is not None
    assert trainer.__name__ == "nnUNetTrainer_184epochs"
    assert issubclass(trainer, nnUNetTrainer)
    assert trainer.build_network_architecture is nnUNetTrainer.build_network_architecture
    assert trainer.dataset_fixer_training_epochs == 184
    assert predictor.initialization == (
        "/models/custom",
        [0, "all"],
        "checkpoint_best.pth",
    )
    assert predict_from_raw_data.recursive_find_trainer_class_by_name is missing


def test_missing_architecture_customizing_trainer_keeps_official_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nnunetv2.inference import predict_from_raw_data

    def missing(trainer_name: str):
        raise RuntimeError(f"Could not find requested nnunet trainer {trainer_name}")

    monkeypatch.setattr(
        predict_from_raw_data,
        "recursive_find_trainer_class_by_name",
        missing,
    )
    predictor = TrainerResolvingPredictor("MyCustomArchitectureTrainer")

    with pytest.raises(RuntimeError, match="MyCustomArchitectureTrainer"):
        _initialize_predictor_from_trained_model_folder(
            predictor,
            model_folder=Path("/models/custom"),
            folds=("0",),
            checkpoint="checkpoint_best.pth",
        )

    assert predict_from_raw_data.recursive_find_trainer_class_by_name is missing


def _tiles(count: int, *, channels: int = 3, size: int = 8) -> list[np.ndarray]:
    return [
        np.full((channels, 1, size, size), index + 1, dtype=np.float32)
        for index in range(count)
    ]


def test_accelerator_and_cpu_batch_ceilings_bound_the_plan_batch_size() -> None:
    assert ACCELERATOR_BATCH_CEILING == 16
    assert CPU_BATCH_CEILING == 4
    assert min(50, ACCELERATOR_BATCH_CEILING) == 16
    assert min(2, CPU_BATCH_CEILING) == 2


def test_equal_shaped_tiles_run_as_one_network_minibatch() -> None:
    predictor = StubPredictor()
    session = _session(predictor, batch_size=4)

    logits = session.predict_logits(_tiles(4))

    assert len(logits) == 4
    assert predictor.batch_sizes == [4]
    assert all(value.shape == (2, 1, 8, 8) for value in logits)


def test_a_minibatch_larger_than_the_batch_size_is_split_without_dropping_tiles() -> None:
    predictor = StubPredictor()
    session = _session(predictor, batch_size=4)

    logits = session.predict_logits(_tiles(10))

    assert len(logits) == 10
    assert predictor.batch_sizes == [4, 4, 2]


def test_differently_shaped_tiles_are_rejected_rather_than_silently_padded() -> None:
    predictor = StubPredictor()
    session = _session(predictor)
    mixed = [*_tiles(1), np.zeros((3, 1, 8, 12), dtype=np.float32)]

    with pytest.raises(DatasetValidationError, match="equally shaped"):
        session.predict_logits(mixed)


def test_preprocessing_padding_consolidates_shapes_and_is_removed_before_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CroppingPreprocessor:
        def __init__(self, *, verbose: bool) -> None:
            assert not verbose

        def run_case_npy(self, data, seg, properties, *args):
            properties = {
                **properties,
                "shape_before_cropping": (1, 8, 8),
                "shape_after_cropping_and_before_resampling": (1, 5, 7),
                "bbox_used_for_cropping": [[0, 1], [1, 6], [0, 7]],
            }
            return data[:, :, 1:6, :7], seg, properties

    predictor = StubPredictor()
    predictor.configuration_manager = type(
        "ConfigurationManager",
        (),
        {"preprocessor_class": CroppingPreprocessor},
    )()
    predictor.plans_manager = object()
    predictor.dataset_json = {}
    predictor.label_manager = object()
    session = _session(predictor)

    prepared, properties = session.preprocess(
        np.ones((8, 8, 3), dtype=np.uint8)
    )

    assert prepared.shape == (3, 1, 8, 8)
    assert properties["_dataset_fixer_preprocess_padding_slicer"] == (
        (0, 3),
        (0, 1),
        (1, 6),
        (0, 7),
    )

    captured: dict[str, object] = {}

    def convert(logits, *args, **kwargs):
        captured["shape"] = tuple(logits.shape)
        captured["properties"] = args[3]
        return np.zeros((1, 8, 8), dtype=np.uint8), np.zeros(
            (2, 1, 8, 8), dtype=np.float32
        )

    monkeypatch.setattr(
        "nnunetv2.inference.export_prediction."
        "convert_predicted_logits_to_segmentation_with_correct_shape",
        convert,
    )
    session.to_probabilities(
        torch.zeros((2, 1, 8, 8), dtype=torch.float32),
        properties,
    )

    assert captured["shape"] == (2, 1, 5, 7)
    assert "_dataset_fixer_preprocess_padding_slicer" not in captured["properties"]


def test_out_of_memory_halves_the_batch_and_retries_down_to_one() -> None:
    predictor = StubPredictor()
    session = _session(predictor, batch_size=8)
    failures = {"remaining": 3}
    original = predictor._internal_maybe_mirror_and_predict

    def failing(x):
        if failures["remaining"] > 0 and x.shape[0] > 1:
            failures["remaining"] -= 1
            raise RuntimeError("MPS backend out of memory (MPS allocated: 9.00 GB)")
        return original(x)

    predictor._internal_maybe_mirror_and_predict = failing
    session.release = lambda: None

    backoffs: list[tuple[int, int, int, str]] = []
    logits = session.predict_logits(
        _tiles(8),
        on_oom=lambda attempted, retry, number, error: backoffs.append(
            (attempted, retry, number, error)
        ),
    )

    assert len(logits) == 8
    assert session.resolved_batch_size == 1
    assert session.oom_retries == 3
    assert predictor.batch_sizes == [1] * 8
    assert [(attempted, retry, number) for attempted, retry, number, _ in backoffs] == [
        (8, 4, 1),
        (4, 2, 2),
        (2, 1, 3),
    ]
    assert all("out of memory" in error for *_, error in backoffs)


def test_a_non_memory_runtime_error_is_not_retried() -> None:
    predictor = StubPredictor()
    session = _session(predictor, batch_size=4)

    def broken(x):
        raise RuntimeError("shape mismatch in convolution")

    predictor._internal_maybe_mirror_and_predict = broken

    with pytest.raises(RuntimeError, match="shape mismatch"):
        session.predict_logits(_tiles(4))
    assert session.oom_retries == 0
    assert session.resolved_batch_size == 4


def test_backoff_stops_at_one_instead_of_looping_forever() -> None:
    predictor = StubPredictor()
    session = _session(predictor, batch_size=2)

    def always_out_of_memory(x):
        raise RuntimeError("CUDA out of memory")

    predictor._internal_maybe_mirror_and_predict = always_out_of_memory
    session.release = lambda: None

    with pytest.raises(RuntimeError, match="out of memory"):
        session.predict_logits(_tiles(2))
    assert session.resolved_batch_size == 1


def test_a_single_fold_loads_its_weights_once_for_every_tile() -> None:
    predictor = StubPredictor()
    session = _session(predictor, batch_size=2)

    session.predict_logits(_tiles(6))
    session.predict_logits(_tiles(6))

    assert predictor.loaded_folds == [0]
    assert session.weight_loads == 1


def test_multi_fold_models_iterate_folds_outside_the_tile_batches() -> None:
    predictor = StubPredictor()
    predictor.list_of_parameters = [{}, {}, {}]
    session = _session(predictor, batch_size=4, folds=("0", "1", "2"))

    logits = session.predict_logits(_tiles(8))

    # Three folds over eight tiles: three weight loads, not one per tile.
    assert predictor.loaded_folds == [0, 1, 2]
    assert session.weight_loads == 3
    assert predictor.batch_sizes == [4, 4] * 3
    assert len(logits) == 8


def test_mirroring_tta_is_delegated_to_the_official_predictor() -> None:
    predictor = StubPredictor()
    session = _session(predictor, batch_size=4)

    session.predict_logits(_tiles(4))

    # nnU-Net owns the mirroring policy; the engine only feeds it batches.
    assert predictor.mirrored == 1


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory",
        "MPS backend out of memory",
        "RuntimeError: failed to allocate 8 GB",
        "Invalid buffer size: cannot allocate memory",
    ],
)
def test_recognized_out_of_memory_messages(message: str) -> None:
    assert _is_out_of_memory(RuntimeError(message))


def test_unrelated_runtime_errors_are_not_treated_as_out_of_memory() -> None:
    assert not _is_out_of_memory(RuntimeError("expected 3 channels, found 4"))
    assert not _is_out_of_memory(ValueError("bad dtype"))


def test_telemetry_records_the_resolved_engine_facts() -> None:
    telemetry = EngineTelemetry(
        device="mps",
        plan_batch_size=50,
        requested_batch_size=16,
        resolved_batch_size=8,
        oom_retries=1,
        tiles=36893,
        sources=1179,
        folds=("0", "1"),
        weight_loads=2,
        workers=4,
        preprocess_seconds=1.5,
        inference_seconds=120.25,
    )

    payload = telemetry.as_dict()

    assert payload["nnunet_execution_engine"] == "in-process-minibatched"
    assert payload["nnunet_device"] == "mps"
    assert payload["nnunet_plan_batch_size"] == 50
    assert payload["nnunet_resolved_batch_size"] == 8
    assert payload["nnunet_oom_retries"] == 1
    assert payload["nnunet_tiles"] == 36893
    assert payload["nnunet_folds"] == ["0", "1"]
    assert payload["nnunet_phase_seconds"]["inference"] == 120.25
