from __future__ import annotations

import json
import shutil
import subprocess
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dataset_fixer import (
    Dataset,
    DatasetValidationError,
    ImagePrediction,
    Model,
    ModelCollection,
    PredictionResult,
    SemanticComparisonResult,
    model_label,
)
from dataset_fixer.comparison.cache import cache_key, default_cache_root
from dataset_fixer.comparison.types import Prediction
from dataset_fixer.semantic_comparison import (
    SEMANTIC_REPORT_SCHEMA,
    _SemanticCase,
    _all_pairwise_statistics,
    _binary_metric_breakdown,
    _canonicalize_predictions,
    _freeze_cohort,
    _project_semantic_predictions,
    _run_command,
    _select_visual_cases,
)
from dataset_fixer.utils import package_versions
from conftest import make_yolo_dataset


@pytest.fixture(autouse=True)
def _stub_nnunet_dependency_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """These adapter tests fake nnU-Net commands/sessions, not its installation."""

    monkeypatch.setattr("dataset_fixer.nnunet_engine.require_nnunet", lambda: None)


def _semantic_export(tmp_path: Path) -> Dataset:
    source = make_yolo_dataset(
        tmp_path / "segments",
        task="segment",
        names=["school"],
        train_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        val_rows=[
            "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8",
            "0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7",
        ],
        size=(40, 30),
    )
    exported = Dataset.open(source, task="segment", progress=False).export(
        destination=tmp_path / "semantic",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )
    assert isinstance(exported, Dataset)
    assert exported.format == "semantic_masks"
    return exported


def _nnunet_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": {"0": "rgb_to_0_1", "1": "rgb_to_0_1", "2": "rgb_to_0_1"},
                "labels": {"background": 0, "school": 1},
                "numTraining": 3,
                "file_ending": ".png",
                "overwrite_image_reader_writer": "NaturalImage2DIO",
            }
        ),
        encoding="utf-8",
    )
    (root / "plans.json").write_text(
        json.dumps({"dataset_name": "Dataset999_Test", "configurations": {"2d": {}}}),
        encoding="utf-8",
    )
    fold = root / "fold_0"
    fold.mkdir()
    (fold / "checkpoint_final.pth").write_bytes(f"checkpoint:{root.name}".encode())
    return root


def test_semantic_cohort_freeze_honors_oversized_skip_audit(tmp_path: Path) -> None:
    exported = _semantic_export(tmp_path)
    val_samples = [sample for sample in exported._samples if sample.split == "val"]
    exported._geometry_skip_audit = (
        {"source": str(val_samples[1].image_path), "split": "val"},
    )

    cases, _ = _freeze_cohort(exported, "val", progress=False)

    assert [case.image_path for case in cases] == [val_samples[0].image_path]


def test_nnunet_prediction_uses_shared_smaller_and_oversized_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = _semantic_export(tmp_path)
    val_samples = [sample for sample in exported._samples if sample.split == "val"]
    oversized = val_samples[1]
    oversized_mask = exported._mask_paths[oversized.image_path.resolve()]
    with Image.open(oversized.image_path) as opened:
        opened.resize((41, 30), Image.Resampling.BICUBIC).save(oversized.image_path)
    with Image.open(oversized_mask) as opened:
        opened.resize((41, 30), Image.Resampling.NEAREST).save(oversized_mask)
    model = Model(
        _nnunet_model(tmp_path / "size-policy-nnunet"),
        native_tile_size=(30, 40),
        upscale_factor=1,
        workers=1,
    )

    def fake_predict(_model: Model, inputs, **_kwargs):
        return tuple(
            ImagePrediction(
                image_id=value.image_id,
                image_path=value.image_path,
                relative_path=value.relative_path,
                width=value.width,
                height=value.height,
                mask=np.zeros((value.height, value.width), dtype=np.uint8),
            )
            for value in inputs
        )

    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.predict_nnunet_model",
        fake_predict,
    )

    with pytest.raises(DatasetValidationError, match="exceeds native_tile_size"):
        model.predict(exported, split="val", progress=False)
    result = model.predict(exported, split="val", errors="skip", progress=False)

    assert [record.image_path for record in result.records] == [val_samples[0].image_path]
    assert len(result.settings["source_size_policy"]["skipped_inputs"]) == 1


def test_model_label_uses_normalized_run_identifier() -> None:
    label = model_label(
        {
            "model": (
                "islands-128-08.08.2026-merged-1class_masks"
                "__gnsuhtfc__yolo26x-sem__512px"
            ),
            "model_source": (
                "wandb:max-planck-institute-for-animal-behavior/"
                "schools-segmentation/gnsuhtfc"
            ),
            "source_created_at": "2026-08-10T14:07:09+00:00",
            "model_sha256_short": "abcdef12",
        }
    )

    assert label == "2026-08-10 14:07:09 · abcdef12"
    assert "wandb:" not in label


def test_model_label_uses_local_basename_without_dataset_prefix() -> None:
    label = model_label(
        {
            "model": (
                "08.08.2026-merged-1class_masks-yolo26x-sem-512px-"
                "2026-08-11_00-21_20260811_002334"
            ),
            "model_source": (
                "/models/08.08.2026-merged-1class_masks-yolo26x-sem-512px-"
                "2026-08-11_00-21_20260811_002334.pt"
            ),
            "model_type": "yolo26x-sem",
            "upscale_factor": 4,
            "effective_prediction_resolution": "512px",
            "checkpoint_sha256_short": "74c3e770",
        }
    )

    assert label == "2026-08-11 00:23:34 · 74c3e770"
    assert "local checkpoint" not in label


def test_model_label_normalizes_named_wandb_run_and_adds_hash() -> None:
    label = model_label(
        {
            "model": (
                "islands-128-08.08.2026-merged-1class_masks__run__"
                "yolo26m-sem__1024px"
            ),
            "model_source": (
                "wandb:team/project/islands-128-08.08.2026-merged-1class_masks-"
                "yolo26m-sem-1024px-8x-2026-08-10_11-46_20260810_114819"
            ),
            "model_type": "yolo26m-sem",
            "upscale_factor": 8,
            "effective_prediction_resolution": "1024px",
            "model_sha256_short": "a1b2c3d4",
        }
    )

    assert label == "2026-08-10 11:48:19 · a1b2c3d4"


def test_nnunet_command_progress_suppresses_per_case_chatter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    predictions = tmp_path / "predictions"
    predictions.mkdir()

    def noisy_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        stream = kwargs["stdout"]
        for index in range(2):
            stream.write(
                f"Predicting val_{index:06d}:\n"
                "perform_everything_on_device: False\n"
                "sending off prediction to background worker for resampling and export\n"
                f"done with val_{index:06d}\n"
            )
            Image.new("L", (1, 1), 0).save(
                predictions / f"val_{index:06d}.png"
            )
        stream.flush()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.subprocess.run",
        noisy_run,
    )
    _run_command(
        ["nnUNetv2_predict_from_modelfolder", "--example"],
        progress=True,
        progress_total=2,
        progress_directory=predictions,
        progress_description="nnU-Net native prediction",
    )

    output = capsys.readouterr()
    assert "Running nnU-Net command:" in output.out
    assert "Predicting val_" not in output.out
    assert "sending off prediction" not in output.out
    assert "nnU-Net native prediction" in output.err
    assert "2/2" in output.err


def test_all_pairwise_statistics_have_no_baseline() -> None:
    rows = {
        "alpha": [
            {"case_id": "a", "dice": 0.2},
            {"case_id": "b", "dice": 0.5},
        ],
        "beta": [
            {"case_id": "a", "dice": 0.4},
            {"case_id": "b", "dice": 0.5},
        ],
        "gamma": [
            {"case_id": "a", "dice": 0.1},
            {"case_id": "b", "dice": 0.3},
        ],
    }

    paired = _all_pairwise_statistics(rows, resamples=200, seed=42)

    assert len(paired) == 3
    assert {(row["model_a"], row["model_b"]) for row in paired} == {
        ("alpha", "beta"),
        ("alpha", "gamma"),
        ("beta", "gamma"),
    }
    assert all("baseline" not in row for row in paired)
    assert all("p_value_holm" in row for row in paired)


def test_invalid_instance_polygons_become_cached_style_warnings(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (8, 8), "white").save(image)
    Image.new("L", (8, 8), 0).save(mask)
    case = _SemanticCase(
        case_id="val_000000",
        relative_path=Path("image.png"),
        image_path=image,
        mask_path=mask,
        width=8,
        height=8,
        image_sha256="a" * 64,
        mask_sha256="b" * 64,
    )
    record = ImagePrediction(
        image_id=case.case_id,
        image_path=image,
        relative_path="image.png",
        width=8,
        height=8,
        objects=(
            Prediction(
                class_id=0,
                score=0.9,
                polygons=[[(1, 1), (6, 1), (6, 6)], [(0, 0), (1, 1)]],
            ),
            Prediction(class_id=0, score=0.8, polygons=None),
        ),
    )

    projected, projection, warnings = _project_semantic_predictions(
        (record,), "segment", [case], "segmenter", confidence=0.25
    )

    assert projection == "polygon-foreground-union"
    assert projected[0].mask.any()
    assert projected[0].foreground_probability is not None
    assert projected[0].metadata["probability_source"] == (
        "rasterized-instance-confidence"
    )
    assert float(np.max(projected[0].foreground_probability)) == pytest.approx(
        0.9,
        abs=5e-4,
    )
    assert [warning["action"] for warning in warnings] == [
        "skipped-object",
        "skipped-object",
    ]
    assert {warning["reason"] for warning in warnings} == {
        "fewer than three polygon points",
        "no polygon returned by segmentation model",
    }


def test_instance_foreground_score_map_uses_maximum_overlapping_score(
    tmp_path: Path,
) -> None:
    image = tmp_path / "overlap.png"
    Image.new("RGB", (10, 10)).save(image)
    record = ImagePrediction(
        image_id="overlap",
        image_path=image,
        relative_path="overlap.png",
        width=10,
        height=10,
        objects=(
            Prediction(
                class_id=0,
                score=0.35,
                polygon=[(1, 1), (7, 1), (7, 7), (1, 7)],
            ),
            Prediction(
                class_id=0,
                score=0.8,
                polygon=[(4, 4), (9, 4), (9, 9), (4, 9)],
            ),
        ),
    )

    score_map = record.foreground_score_map()

    assert score_map is not None
    assert score_map[2, 2] == pytest.approx(0.35)
    assert score_map[5, 5] == pytest.approx(0.8)
    assert score_map[8, 8] == pytest.approx(0.8)
    assert score_map[0, 0] == pytest.approx(0.0)
    assert len(record.objects) == 2


def test_prediction_result_projects_scored_instances_to_semantic_masks(
    tmp_path: Path,
) -> None:
    image = tmp_path / "project.png"
    Image.new("RGB", (10, 10)).save(image)
    native = PredictionResult(
        model_name="segmenter",
        model_kind="ultralytics",
        task="segment",
        backend="sahi",
        records=(
            ImagePrediction(
                image_id="project",
                image_path=image,
                relative_path="project.png",
                width=10,
                height=10,
                objects=(
                    Prediction(
                        class_id=0,
                        score=0.4,
                        polygon=[(1, 1), (4, 1), (4, 4), (1, 4)],
                    ),
                    Prediction(
                        class_id=0,
                        score=0.8,
                        polygon=[(5, 5), (9, 5), (9, 9), (5, 9)],
                    ),
                ),
            ),
        ),
        inference_seconds=1.0,
        settings={"prediction_threshold": 0.6},
        cache_info={"status": "hit"},
    )

    projected = native.as_semantic()

    assert projected.task == "semantic_segment"
    assert not projected.records[0].mask[2, 2]
    assert projected.records[0].mask[7, 7]
    assert len(projected.records[0].objects) == 2
    assert projected.cache_info["status"] == "hit"
    assert projected.cache_info["projection_status"] == (
        "derived-from-native-predictions"
    )


class FakeSession:
    """Stand-in for a loaded nnU-Net model that records how it was driven."""

    def __init__(
        self,
        *,
        num_classes: int = 2,
        plan_batch_size: int = 50,
        requested_batch_size: int = 16,
        workers: int = 1,
        foreground: float = 1.0,
    ) -> None:
        self.num_classes = num_classes
        self.plan_batch_size = plan_batch_size
        self.requested_batch_size = requested_batch_size
        self.resolved_batch_size = requested_batch_size
        self.oom_retries = 0
        self.use_tta = False
        self.workers = workers
        self.weight_loads = 0
        self.forward_passes = 0
        self.foreground = foreground
        self.batch_sizes: list[int] = []
        self.released = 0

    def preprocess_many(self, images):
        return [
            (
                np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[:, None],
                {"tile_shape": np.asarray(image).shape[:2]},
            )
            for image in images
        ]

    def predict_logits(self, prepared, *, on_batch=None, on_oom=None):
        shapes = {tuple(value.shape) for value in prepared}
        assert len(shapes) == 1, f"minibatch received mixed shapes: {shapes}"
        self.batch_sizes.append(len(prepared))
        if on_batch is not None:
            on_batch(len(prepared))
        self.weight_loads += 1
        self.forward_passes += 1
        return [value[:1] for value in prepared]

    def to_probabilities_many(self, pairs):
        output = []
        for _, properties in pairs:
            height, width = properties["tile_shape"]
            foreground = np.full((height, width), self.foreground, dtype=np.float32)
            output.append(np.stack((1.0 - foreground, foreground))[:, None])
        return output

    def release(self) -> None:
        self.released += 1


def _fake_nnunet_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    if command[0] == "nnUNetv2_predict_from_modelfolder":
        assert "--save_probabilities" in command
        image_dir = Path(command[command.index("-i") + 1])
        prediction_dir = Path(command[command.index("-o") + 1])
        model_folder = Path(command[command.index("-m") + 1])
        prediction_dir.mkdir(parents=True, exist_ok=True)
        canonical_label_dir = image_dir.parents[2] / "canonical" / "labels"
        for image_path in sorted(image_dir.glob("*_0000.png")):
            case_id = image_path.stem.removesuffix("_0000")
            label_path = canonical_label_dir / f"{case_id}.png"
            with Image.open(image_path) as input_image, Image.open(label_path) as opened_label:
                label = opened_label.convert("L").resize(
                    input_image.size,
                    Image.Resampling.NEAREST,
                )
            if "weak" in model_folder.name:
                label = Image.new("L", label.size, 0)
            label.save(prediction_dir / f"{case_id}.png")
            foreground = np.asarray(label, dtype=np.float32)
            probabilities = np.stack((1.0 - foreground, foreground), axis=0)[:, None]
            np.savez_compressed(
                prediction_dir / f"{case_id}.npz",
                probabilities=probabilities,
            )
        return subprocess.CompletedProcess(command, 0)

    assert command[0] == "nnUNetv2_evaluate_folder"
    reference_dir = Path(command[1])
    prediction_dir = Path(command[2])
    summary_path = Path(command[command.index("-o") + 1])
    per_case = []
    for reference_path in sorted(reference_dir.glob("*.png")):
        prediction_path = prediction_dir / reference_path.name
        with Image.open(reference_path) as reference_image:
            reference = np.asarray(reference_image) > 0
        with Image.open(prediction_path) as prediction_image:
            prediction = np.asarray(prediction_image) > 0
        tp = int(np.sum(reference & prediction))
        fp = int(np.sum(~reference & prediction))
        fn = int(np.sum(reference & ~prediction))
        tn = int(np.sum(~reference & ~prediction))
        dice_denominator = 2 * tp + fp + fn
        iou_denominator = tp + fp + fn
        dice = 2 * tp / dice_denominator if dice_denominator else float("nan")
        iou = tp / iou_denominator if iou_denominator else float("nan")
        metrics = {
            "Dice": dice,
            "IoU": iou,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "n_pred": int(np.sum(prediction)),
            "n_ref": int(np.sum(reference)),
        }
        per_case.append(
            {
                "reference_file": str(reference_path),
                "prediction_file": str(prediction_path),
                "metrics": {"1": metrics},
            }
        )
    foreground_mean = {}
    for key in ("Dice", "IoU", "TP", "FP", "FN", "TN", "n_pred", "n_ref"):
        values = [case["metrics"]["1"][key] for case in per_case]
        foreground_mean[key] = float(np.nanmean(values))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "metric_per_case": per_case,
                "mean": {"1": foreground_mean},
                "foreground_mean": foreground_mean,
            }
        ),
        encoding="utf-8",
    )
    return subprocess.CompletedProcess(command, 0)


def test_canonical_projection_area_pools_probabilities_instead_of_sampling_hard_masks(
    tmp_path: Path,
) -> None:
    native = tmp_path / "native"
    canonical = tmp_path / "canonical"
    native.mkdir()
    foreground = np.asarray([[0.9, 0.9], [0.9, 0.0]], dtype=np.float32)
    probabilities = np.stack((1.0 - foreground, foreground), axis=0)[:, None]
    np.savez_compressed(native / "val_000000.npz", probabilities=probabilities)

    # Nearest-neighbor hard-mask projection samples the lower-right background
    # pixel and loses the foreground majority in this source-pixel block.
    hard = Image.fromarray(np.argmax(probabilities[:, 0], axis=0).astype(np.uint8))
    assert hard.resize((1, 1), Image.Resampling.NEAREST).getpixel((0, 0)) == 0

    case = _SemanticCase(
        case_id="val_000000",
        relative_path=Path("sample.jpg"),
        image_path=tmp_path / "sample.jpg",
        mask_path=tmp_path / "sample.png",
        width=1,
        height=1,
        image_sha256="image",
        mask_sha256="mask",
    )
    _canonicalize_predictions(
        native,
        canonical,
        [case],
        model_name="pooled",
        upscale_factor=2,
    )

    with Image.open(canonical / "val_000000.png") as projected:
        assert projected.getpixel((0, 0)) == 1


def test_comparison_case_selection_is_seeded_random_and_excludes_empty_masks(
    tmp_path: Path,
) -> None:
    cases: list[_SemanticCase] = []
    for index in range(12):
        image_path = tmp_path / f"image-{index}.png"
        mask_path = tmp_path / f"mask-{index}.png"
        Image.new("RGB", (4, 4), "black").save(image_path)
        mask = np.zeros((4, 4), dtype=np.uint8)
        if index >= 2:
            mask[index % 4, index % 4] = 1
        Image.fromarray(mask).save(mask_path)
        cases.append(
            _SemanticCase(
                case_id=f"case-{index}",
                relative_path=Path(f"image-{index}.png"),
                image_path=image_path,
                mask_path=mask_path,
                width=4,
                height=4,
                image_sha256=f"image-{index}",
                mask_sha256=f"mask-{index}",
            )
        )

    first = _select_visual_cases(cases, samples=8, include_empty=False, seed=42)
    second = _select_visual_cases(cases, samples=8, include_empty=False, seed=42)

    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert len(first) == 8
    assert all(int(case.case_id.removeprefix("case-")) >= 2 for case in first)


def test_semantic_comparison_reports_finite_dice_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "segments-with-empty",
        task="segment",
        names=["school"],
        train_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        val_rows=["0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7", ""],
        size=(20, 20),
    )
    exported = Dataset.open(source, task="segment", progress=False).export(
        destination=tmp_path / "semantic-with-empty",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )
    assert isinstance(exported, Dataset)
    assert exported.format == "semantic_masks"
    model = _nnunet_model(tmp_path / "perfect-empty-model")
    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.shutil.which",
        lambda command: f"/fake/{command}",
    )
    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.subprocess.run",
        _fake_nnunet_run,
    )
    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.environment_snapshot",
        lambda: {"test": True},
    )

    models = Model.load_many({"perfect": {"path": model, "workers": 1}})
    result = models.compare(
        exported,
        progress=False,
        destination=tmp_path / "finite-support-comparison",
        group_by=lambda path: path.stem,
    )

    row = result.ranking.iloc[0]
    assert row["cohort_cases"] == 2
    assert row["support_cases"] == 1
    assert row["undefined_cases"] == 1
    assert row["positive_cases"] == 1
    assert row["empty_cases"] == 1
    assert row["positive_case_dice"] == pytest.approx(1.0)
    assert row["empty_image_specificity"] == pytest.approx(1.0)
    assert row["raw_presence_precision"] == pytest.approx(1.0)
    assert row[
        "component_filtered_presence_precision"
    ] == pytest.approx(1.0)
    manifest = json.loads(
        (
            tmp_path
            / "finite-support-comparison"
            / "reports"
            / "result.json"
        ).read_text()
    )
    assert manifest["presence_analysis"]["threshold_source"] == (
        "held-out-reference-object-p10"
    )
    assert manifest["grouped_analysis"]["status"] == "complete"
    assert manifest["ranking"][0]["group_macro_dice"] == pytest.approx(1.0)
    assert manifest["ranking"][0]["group_macro_presence_f1"] == pytest.approx(1.0)
    assert manifest["grouped_analysis"]["presence"]["status"] == "complete"
    assert (
        tmp_path
        / "finite-support-comparison"
        / "reports"
        / "grouped-metric-breakdown.png"
    ).is_file()
    for metric in ("precision", "recall", "f1"):
        assert (
            tmp_path
            / "finite-support-comparison"
            / "reports"
            / f"grouped-presence-{metric}.png"
        ).is_file()


def test_binary_metric_breakdown_separates_positive_and_empty_images() -> None:
    rows = [
        {
            "dice": 0.8,
            "iou": 2 / 3,
            "tp": 8,
            "fp": 2,
            "fn": 2,
            "n_ref": 10,
            "n_pred": 10,
        },
        {
            "dice": 0.0,
            "iou": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 5,
            "n_ref": 5,
            "n_pred": 0,
        },
        {
            "dice": 0.0,
            "iou": 0.0,
            "tp": 0,
            "fp": 1,
            "fn": 0,
            "n_ref": 0,
            "n_pred": 1,
        },
        {
            "dice": float("nan"),
            "iou": float("nan"),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "n_ref": 0,
            "n_pred": 0,
        },
    ]

    metrics = _binary_metric_breakdown(rows)

    assert metrics["micro_dice"] == pytest.approx(16 / 26)
    assert metrics["micro_iou"] == pytest.approx(8 / 18)
    assert metrics["foreground_precision"] == pytest.approx(8 / 11)
    assert metrics["foreground_recall"] == pytest.approx(8 / 15)
    assert metrics["positive_case_dice"] == pytest.approx(0.4)
    assert metrics["positive_case_iou"] == pytest.approx(1 / 3)
    assert metrics["positive_micro_dice"] == pytest.approx(16 / 25)
    assert metrics["positive_micro_iou"] == pytest.approx(8 / 17)
    assert metrics["positive_image_recall"] == pytest.approx(0.5)
    assert metrics["empty_image_specificity"] == pytest.approx(0.5)
    assert metrics["empty_image_false_positive_rate"] == pytest.approx(0.5)
    assert metrics["empty_false_positive_pixels"] == 1
    assert metrics["empty_mean_false_positive_pixels"] == pytest.approx(0.5)


def test_mixed_yolo_seg_and_semantic_models_negotiate_binary_mask_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exported = _semantic_export(tmp_path)
    yolo_path = tmp_path / "yolo-seg.pt"
    yolo_path.write_bytes(b"synthetic-yolo-seg")
    yolo_semantic_path = tmp_path / "yolo-semantic.pt"
    yolo_semantic_path.write_bytes(b"synthetic-yolo-semantic")
    nnunet_path = _nnunet_model(tmp_path / "semantic-model")
    models = Model.load_many(
        {
            "yolo-seg": {
                "path": yolo_path,
                "task": "segment",
                "device": "cpu",
                "inference": "sahi",
                "sahi_slice_height": 24,
                "sahi_slice_width": 20,
                "sahi_overlap": 0.25,
            },
            "yolo-semantic": {
                "path": yolo_semantic_path,
                "task": "semantic",
                "device": "cpu",
                "inference": "sahi",
                "sahi_slice_height": 24,
                "sahi_slice_width": 20,
                "sahi_overlap": 0.25,
            },
            "nnunet": {
                "path": nnunet_path,
                "device": "cpu",
                "workers": 1,
                "nnunet_tta": True,
                "inference": "sahi",
                "sahi_slice_height": 24,
                "sahi_slice_width": 20,
                "sahi_overlap": 0.25,
            },
        }
    )

    prediction_options: list[dict[str, object]] = []

    def fake_predict(
        model: Model,
        source: object,
        **options: object,
    ) -> PredictionResult:
        prediction_options.append(options)
        records: list[ImagePrediction] = []
        for value in tuple(source):  # type: ignore[arg-type]
            assert value.mask_path is not None
            with Image.open(value.mask_path) as opened_mask:
                truth = np.asarray(opened_mask.convert("L")) > 0
            if model.task == "semantic_segment":
                records.append(
                    ImagePrediction(
                        image_id=value.image_id,
                        image_path=value.image_path,
                        relative_path=value.relative_path,
                        width=value.width,
                        height=value.height,
                        mask=truth,
                    )
                )
                task = "semantic_segment"
            else:
                bounds = Image.fromarray(truth).getbbox()
                objects: tuple[Prediction, ...] = ()
                if bounds is not None:
                    left, top, right, bottom = bounds
                    objects = (
                        Prediction(
                            class_id=0,
                            score=0.95,
                            polygon=[
                                (left, top),
                                (right - 1, top),
                                (right - 1, bottom - 1),
                                (left, bottom - 1),
                            ],
                        ),
                        Prediction(
                            class_id=0,
                            score=0.9,
                            polygon=[(left, top), (right - 1, bottom - 1)],
                        ),
                    )
                records.append(
                    ImagePrediction(
                        image_id=value.image_id,
                        image_path=value.image_path,
                        relative_path=value.relative_path,
                        width=value.width,
                        height=value.height,
                        objects=objects,
                    )
                )
                task = "segment"
        return PredictionResult(
            model_name=model.name,
            model_kind=model.kind,
            task=task,
            backend="synthetic",
            records=tuple(records),
            inference_seconds=0.1,
        )

    monkeypatch.setattr(Model, "predict", fake_predict)
    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.environment_snapshot",
        lambda: {"test": True},
    )
    destination = tmp_path / "mixed-comparison"
    result = models.compare(
        exported,
        split="val",
        progress=False,
        destination=destination,
    )

    assert isinstance(result, SemanticComparisonResult)
    assert isinstance(result.ranking, pd.DataFrame)
    assert isinstance(result.ranking.index, pd.RangeIndex)
    assert set(result.ranking["model"]) == {
        "yolo-seg",
        "yolo-semantic",
        "nnunet",
    }
    assert np.allclose(result.ranking["dice"].to_numpy(dtype=float), 1.0)
    by_name = result.ranking.set_index("model")
    assert by_name.loc["yolo-seg", "native_task"] == "segment"
    assert by_name.loc["yolo-seg", "projection"] == "polygon-foreground-union"
    assert by_name.loc["nnunet", "native_task"] == "semantic_segment"
    assert by_name.loc["nnunet", "projection"] == "native-semantic-mask"
    assert by_name.loc["yolo-semantic", "native_task"] == "semantic_segment"
    assert by_name.loc["yolo-semantic", "projection"] == "native-semantic-mask"
    assert by_name.loc["yolo-seg", "micro_dice"] == pytest.approx(1.0)
    assert by_name.loc["yolo-seg", "warning_count"] == 2
    manifest = json.loads((destination / "reports" / "result.json").read_text())
    assert manifest["backend"] == "common-semantic-mask"
    assert manifest["negotiated_comparison_space"] == "semantic"
    assert len(manifest["warnings"]) == 2
    assert all(warning["action"] == "skipped-object" for warning in manifest["warnings"])
    assert set(manifest["settings"]["sahi_models"]) == {
        "yolo-seg",
        "yolo-semantic",
        "nnunet",
    }
    assert all(options["inference"] == "sahi" for options in prediction_options)
    assert all(model.settings["sahi_slice_height"] == 24 for model in models)
    assert (destination / "reports" / "plots.png").is_file()
    assert (destination / "reports" / "metric-breakdown.png").is_file()
    assert (destination / "reports" / "object-size-breakdown.png").is_file()
    assert (destination / "reports" / "comparison.png").is_file()
    assert manifest["object_size_analysis"]["status"] == "complete"
    assert manifest["object_size_analysis"]["connectivity"] == 8
    assert manifest["object_size_analysis"]["matching_class_policy"] == (
        "binary-foreground"
    )
    assert all(
        row["small_object_dice"] == pytest.approx(1.0)
        for row in manifest["ranking"]
    )
    assert "positive_micro_iou" in manifest["ranking"][0]
    assert not list(destination.rglob("*.jsonl"))
    calls_after_first = len(prediction_options)

    # Releases before the stable logical cache identity included execution-only
    # settings and package versions in the directory hash. Simulate an archive
    # restored from that release and verify it is recognized before prediction.
    cache_root = default_cache_root(exported.location) / "semantic"
    legacy_dirs: list[Path] = []
    current_dirs: list[Path] = []
    current_payloads: list[dict[str, object]] = []
    for index, (model, options) in enumerate(zip(models, prediction_options)):
        current_payload = {
            "schema": 2,
            "space": "binary-semantic",
            "cohort": result.cohort_fingerprint,
            "model_hash": model.hash(),
            "kind": model.kind,
            "task": model.task,
            "folds": model.folds,
            "checkpoint": model.checkpoint,
            "upscale_factor": model.upscale_factor,
            "inference": model.inference,
            "resolution": model.resolution or 480,
            "settings": model.settings,
            **(
                {"nnunet_tta": model.nnunet_tta}
                if model.kind == "nnunet"
                else {}
            ),
        }
        current_payloads.append(current_payload)
        current_dir = cache_root / cache_key(current_payload)
        current_dirs.append(current_dir)
        metadata_path = current_dir / "evaluation.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        legacy_payload = {
            "schema": 2,
            "space": "binary-semantic",
            "cohort": result.cohort_fingerprint,
            "model_sha256": model.digest,
            "kind": model.kind,
            "task": model.task,
            "model_settings": model.settings,
            "folds": model.folds,
            "checkpoint": model.checkpoint,
            "upscale_factor": model.upscale_factor,
            "workers": model.workers,
            "batch_size": model.batch_size,
            "settings": {
                key: value for key, value in options.items() if key != "progress"
            },
            "versions": package_versions(),
        }
        if index == 0:
            # A portable archive may contain the correct stored identity under
            # a stale or otherwise noncanonical directory name.
            legacy_dir = cache_root / (f"restored-{index}-" + "0" * 54)
        else:
            # Older entries lacked cache_identity, so recognize their exact
            # historical key as well.
            metadata.pop("cache_identity", None)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            legacy_dir = cache_root / cache_key(legacy_payload)
        assert legacy_dir != current_dir
        current_dir.rename(legacy_dir)
        legacy_dirs.append(legacy_dir)

    restored = models.compare(
        exported,
        split="val",
        progress=False,
        destination=tmp_path / "mixed-comparison-from-cache",
    )
    assert len(prediction_options) == calls_after_first
    assert not any(path.exists() for path in legacy_dirs)
    assert restored.cohort_fingerprint == result.cohort_fingerprint

    # The dataset's absolute path is not part of the cache identity. A cache
    # copied with the dataset must remain valid under a different parent/home.
    relocated_root = tmp_path / "different-user-home" / "semantic-export"
    shutil.copytree(exported.location, relocated_root)
    relocated = Dataset.open(relocated_root, progress=False)
    relocated_result = models.compare(
        relocated,
        split="val",
        progress=False,
        destination=tmp_path / "mixed-comparison-relocated",
    )
    assert relocated_result.cohort_fingerprint == result.cohort_fingerprint
    assert len(prediction_options) == calls_after_first

    # An archive from a release that did not store any logical identity can be
    # retained through an explicit one-time trust decision. Its model name,
    # cohort paths, metadata, and complete mask set must still match. Promotion
    # records that decision and makes subsequent strict lookups reusable.
    unverified = cache_root / ("unverified-" + "1" * 53)
    unverified_metadata_path = current_dirs[0] / "evaluation.json"
    unverified_metadata = json.loads(
        unverified_metadata_path.read_text(encoding="utf-8")
    )
    unverified_metadata.pop("cache_identity")
    unverified_metadata_path.write_text(
        json.dumps(unverified_metadata),
        encoding="utf-8",
    )
    current_dirs[0].rename(unverified)
    calls_before_trusted_migration = len(prediction_options)
    capsys.readouterr()
    models.compare(
        exported,
        split="val",
        progress=True,
        destination=tmp_path / "mixed-comparison-trusted-legacy",
        trust_legacy_cache=True,
    )
    trusted_output = capsys.readouterr().out
    assert f"Cache hit: {models[0].name}" in trusted_output
    assert "user-trusted legacy migration" in trusted_output
    assert len(prediction_options) == calls_before_trusted_migration
    assert current_dirs[0].is_dir()
    assert not unverified.exists()
    migrated_metadata = json.loads(
        (current_dirs[0] / "evaluation.json").read_text(encoding="utf-8")
    )
    assert cache_key(migrated_metadata["cache_identity"]) == cache_key(
        current_payloads[0]
    )
    assert migrated_metadata["legacy_cache_migration"]["trusted"] is True

    models.compare(
        exported,
        split="val",
        progress=False,
        destination=tmp_path / "mixed-comparison-after-trusted-legacy",
    )
    assert len(prediction_options) == calls_before_trusted_migration

    # A completed-looking entry with a missing mask is invalid, must be
    # reported before inference, and must be replaced after inference finishes.
    broken_prediction = next((current_dirs[0] / "predictions").glob("*.png"))
    broken_prediction.unlink()
    capsys.readouterr()
    calls_before_repair = len(prediction_options)
    models.compare(
        exported,
        split="val",
        progress=True,
        destination=tmp_path / "mixed-comparison-repair-invalid-cache",
    )
    repair_output = capsys.readouterr().out
    assert f"Cache invalid: {models[0].name}" in repair_output
    assert "prediction set mismatch" in repair_output
    assert len(prediction_options) == calls_before_repair + 1

    models.compare(
        exported,
        split="val",
        progress=True,
        destination=tmp_path / "mixed-comparison-after-repair",
    )
    hit_output = capsys.readouterr().out
    assert f"Cache hit: {models[0].name}" in hit_output
    assert len(prediction_options) == calls_before_repair + 1

    # Even a staging directory that happens to contain every file is not a
    # published cache entry and must never be discovered as a hit.
    unpublished = current_dirs[1].with_name(
        f".{current_dirs[1].name}.building-interrupted"
    )
    current_dirs[1].rename(unpublished)
    calls_before_unpublished = len(prediction_options)
    models.compare(
        exported,
        split="val",
        progress=True,
        destination=tmp_path / "mixed-comparison-unpublished-cache",
    )
    unpublished_output = capsys.readouterr().out
    assert f"Cache miss: {models[1].name}" in unpublished_output
    assert len(prediction_options) == calls_before_unpublished + 1
    assert unpublished.is_dir()


def test_mixed_comparison_rejects_tasks_without_a_semantic_denominator(
    tmp_path: Path,
) -> None:
    exported = _semantic_export(tmp_path)
    detector_path = tmp_path / "detector.pt"
    detector_path.write_bytes(b"synthetic-detector")
    models = Model.load_many(
        {
            "detector": {"path": detector_path, "task": "detect"},
            "semantic": {"path": _nnunet_model(tmp_path / "semantic-model")},
        }
    )

    with pytest.raises(DatasetValidationError, match="No common semantic denominator"):
        models.compare(
            exported,
            progress=False,
            destination=tmp_path / "unsupported-comparison",
        )


def test_loaded_semantic_models_visualize_only_sampled_cases_with_shared_mask_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = _semantic_export(tmp_path)
    perfect = _nnunet_model(tmp_path / "perfect-model")
    weak = _nnunet_model(tmp_path / "weak-model")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _fake_nnunet_run(command, **kwargs)

    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.shutil.which",
        lambda command: f"/fake/{command}",
    )
    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.subprocess.run",
        fake_run,
    )
    loaded = Model.load_many(
        {
            "resenc-m-very-long-model-name-alpha": {
                "path": perfect,
                "upscale_factor": 2,
                "device": "mps",
                "workers": 1,
            },
            "resenc-m-very-long-model-name-beta": {
                "path": weak,
                "upscale_factor": 1,
                "device": "mps",
                "workers": 1,
            },
        }
    )

    assert isinstance(loaded, ModelCollection)
    assert loaded.names == (
        "resenc-m-very-long-model-name-alpha",
        "resenc-m-very-long-model-name-beta",
    )
    result = loaded.visualize(
        source=exported,
        samples=2,
        columns=1,
        panel_size=2.0,
        model_title_length=18,
        progress=False,
        destination=tmp_path / "quick-comparison.png",
    )

    assert (tmp_path / "quick-comparison.png").is_file()
    assert result is None
    assert len(commands) == 2
    assert all(command[0] == "nnUNetv2_predict_from_modelfolder" for command in commands)
    assert all(command[command.index("-device") + 1] == "mps" for command in commands)
    with Image.open(tmp_path / "quick-comparison.png") as rendered:
        assert rendered.width > rendered.height
        assert rendered.getbbox() is not None


def test_generic_model_predicts_semantic_export_and_saves_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = _semantic_export(tmp_path)
    folder = _nnunet_model(tmp_path / "semantic-model")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _fake_nnunet_run(command, **kwargs)

    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.shutil.which",
        lambda command: f"/fake/{command}",
    )
    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.subprocess.run",
        fake_run,
    )
    model = Model(
        folder,
        name="semantic",
        upscale_factor=2,
        device="mps",
        workers=1,
    )

    assert model.kind == "nnunet"
    assert model.task == "semantic_segment"
    assert model.folds == ("0",)
    result = model.predict(exported, split="val", progress=False)

    assert isinstance(result, PredictionResult)
    assert result.task == "semantic_segment"
    assert len(result) == 2
    assert all(record.mask is not None for record in result)
    assert all(record.mask.shape == (30, 40) for record in result)
    assert len(commands) == 1
    assert commands[0][0] == "nnUNetv2_predict_from_modelfolder"
    saved = result.save(tmp_path / "saved-semantic-predictions")
    assert len(list((saved / "masks").glob("*.png"))) == 2


def test_sampled_semantic_dataset_reuses_full_instance_cache_and_visualizes_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = _semantic_export(tmp_path)
    checkpoint = tmp_path / "sampled-instance.pt"
    checkpoint.write_bytes(b"sampled-instance")
    model = Model(checkpoint, task="segment", resolution=40)
    calls = 0

    def fake_predict_inputs(_model, inputs, **_kwargs):
        nonlocal calls
        calls += 1
        prediction = Prediction(
            class_id=0,
            score=0.9,
            bbox=(8.0, 6.0, 32.0, 24.0),
            polygon=[(8.0, 6.0), (32.0, 6.0), (32.0, 24.0), (8.0, 24.0)],
        )
        return {
            value.image_id: [prediction]
            for value in inputs
        }, "segment", {"synthetic": True}

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.predict_model_inputs",
        fake_predict_inputs,
    )

    full = model.predict(exported, split="val", progress=False)
    sampled = exported.sample(n=1, split="val", seed=42)
    reused = model.predict(sampled, progress=False)

    assert calls == 1
    assert full.cache_info["status"] == "fresh"
    assert reused.cache_info["status"] == "image-compatible-subset-hit"
    assert reused.task == "segment"
    assert len(reused) == 1
    assert reused.records[0].reference_mask_path is not None
    assert reused.records[0].objects[0].polygon is not None

    destination = tmp_path / "sampled-reference.png"
    rendered = reused.visualize(
        columns=1,
        panel_size=2.0,
        zoom=True,
        context_fraction=0.0,
        minimum_context=10,
        outline_width=0.8,
        destination=destination,
        show=False,
    )
    assert rendered is None
    with Image.open(destination) as preview:
        assert preview.width > preview.height
        assert preview.getbbox() is not None


def test_semantic_filter_and_add_retain_mask_metadata(tmp_path: Path) -> None:
    exported = _semantic_export(tmp_path)
    validation = [sample for sample in exported._samples if sample.split == "val"]
    empty_sample = validation[1]
    empty_mask = exported._mask_paths[empty_sample.image_path.resolve()]
    Image.new("L", (empty_sample.width, empty_sample.height), 0).save(empty_mask)
    exported._mask_statistics[empty_sample.image_path.resolve()] = {
        "foreground_pixels": 0,
        "total_pixels": empty_sample.width * empty_sample.height,
    }

    annotated = exported.filter(gt_annotated=True).sample(
        n=1,
        split="val",
        seed=1,
    )
    empty = exported.filter(gt_annotated=False).sample(
        n=1,
        split="val",
        seed=1,
    )
    combined = annotated.add(empty)

    assert [sample.relative_path for sample in combined._samples] == [
        validation[0].relative_path,
        validation[1].relative_path,
    ]
    assert set(combined._mask_paths) == {
        validation[0].image_path.resolve(),
        validation[1].image_path.resolve(),
    }
    assert combined._sample_has_ground_truth(combined._samples[0])
    assert not combined._sample_has_ground_truth(combined._samples[1])

    repeated = annotated.add(annotated)
    cases, _ = _freeze_cohort(repeated, "val", progress=False)
    assert len(cases) == 1
    assert cases[0].image_path == validation[0].image_path.resolve()


def test_prediction_visualization_letterboxes_wide_images_in_fixed_cards(
    tmp_path: Path,
) -> None:
    image = tmp_path / "wide-island.png"
    Image.new("RGB", (800, 80), (25, 45, 65)).save(image)
    mask = np.zeros((80, 800), dtype=bool)
    mask[30:50, 200:600] = True
    result = PredictionResult(
        model_name="wide-model",
        model_kind="ultralytics",
        task="semantic_segment",
        backend="native",
        records=(
            ImagePrediction(
                image_id="wide",
                image_path=image,
                relative_path="wide-aoi/wide-island.png",
                width=800,
                height=80,
                mask=mask,
            ),
        ),
        inference_seconds=0.0,
    )

    destination = tmp_path / "wide-letterboxed.png"
    assert result.visualize(columns=1, panel_size=3.0, destination=destination, show=False) is None
    with Image.open(destination) as rendered:
        assert rendered.width > rendered.height
        assert rendered.getbbox() is not None


def test_prediction_visualization_uses_shared_model_slugs_and_can_hide_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataset_fixer.model as model_module

    image = tmp_path / "slugged.png"
    Image.new("RGB", (32, 32), "black").save(image)
    result = PredictionResult(
        model_name="long-local-model-filename.pt",
        model_kind="ultralytics",
        task="semantic_segment",
        backend="native",
        records=(ImagePrediction(
            image_id="slugged",
            image_path=image,
            relative_path=image.name,
            width=32,
            height=32,
            mask=np.zeros((32, 32), dtype=bool),
        ),),
        inference_seconds=0.0,
        model_metadata={
            "model": "long-local-model-filename.pt",
            "canonical_name": "long-local-model-filename.pt",
            "hash": "i9xve33c",
            "model_identity": "both",
            "model_type": "yolo26m-seg",
            "upscale_factor": 2,
            "resolution": 256,
        },
    )
    specifications: list[str] = []
    monkeypatch.setattr(
        model_module,
        "finish_visualization",
        lambda chart, _options: specifications.append(str(chart.to_dict())),
    )

    result.visualize(show=False)
    result.visualize(show=False, show_model_slugs=False)

    assert all(
        value in specifications[0]
        for value in (
            "long-local-model-filename.pt",
            "i9xve33c",
            "yolo26m-seg",
            "2×",
            "256px",
            "#2563EB",
            "#B45309",
            "#475569",
        )
    )
    assert "yolo26m-seg" not in specifications[1]
    assert "i9xve33c" not in specifications[1]


def test_prediction_visualization_renders_many_rows_with_dedicated_headers(
    tmp_path: Path,
) -> None:
    image = tmp_path / "many-rows.png"
    Image.new("RGB", (320, 240), (25, 45, 65)).save(image)
    mask = np.zeros((240, 320), dtype=bool)
    records = tuple(
        ImagePrediction(
            image_id=f"row-{index}",
            image_path=image,
            relative_path=f"aoi-{index}/many-rows.png",
            width=320,
            height=240,
            mask=mask,
        )
        for index in range(10)
    )
    result = PredictionResult(
        model_name="many-row-model",
        model_kind="ultralytics",
        task="semantic_segment",
        backend="native",
        records=records,
        inference_seconds=0.0,
    )

    destination = tmp_path / "many-rows-rendered.png"
    assert result.visualize(columns=1, panel_size=2.0, destination=destination, show=False) is None
    with Image.open(destination) as rendered:
        assert rendered.height > rendered.width
        assert rendered.getbbox() is not None


def test_prediction_visualization_shortens_keys_in_the_middle(
    tmp_path: Path,
) -> None:
    from dataset_fixer.static_rendering import format_label

    image = tmp_path / "key-width.png"
    Image.new("RGB", (320, 240), (25, 45, 65)).save(image)
    mask = np.zeros((240, 320), dtype=bool)
    formerly_truncated = "aoi-" + "moderate-identifier-" * 5 + "/image.png"
    too_wide = "aoi-" + "extremely-long-identifier-" * 40 + "/image.png"
    result = PredictionResult(
        model_name="key-width-model",
        model_kind="ultralytics",
        task="semantic_segment",
        backend="native",
        records=tuple(
            ImagePrediction(
                image_id=str(index),
                image_path=image,
                relative_path=relative_path,
                width=320,
                height=240,
                mask=mask,
            )
            for index, relative_path in enumerate((formerly_truncated, too_wide))
        ),
        inference_seconds=0.0,
    )

    destination = tmp_path / "shortened-keys.png"
    assert result.visualize(columns=1, panel_size=3.0, destination=destination, show=False) is None
    first = format_label(formerly_truncated, mode="middle", maximum=125)
    second = format_label(too_wide, mode="middle", maximum=125)
    assert "…" not in first[0]
    assert "…" in second[0]
    assert len(second[0]) == 125
    assert destination.is_file()


def test_semantic_predict_and_compare_share_raw_prediction_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = _semantic_export(tmp_path)
    checkpoint = tmp_path / "shared-semantic.pt"
    checkpoint.write_bytes(b"shared-semantic")
    model = Model(checkpoint, task="semantic", resolution=64)
    calls = 0

    def fake_predict_inputs(_model, inputs, **_kwargs):
        nonlocal calls
        calls += 1
        values = {}
        for value in inputs:
            assert value.mask_path is not None
            with Image.open(value.mask_path) as opened:
                values[value.image_id] = np.asarray(opened.convert("L")) > 0
        return values, "semantic_segment", {"synthetic": True}

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.predict_model_inputs",
        fake_predict_inputs,
    )
    direct = model.predict(
        exported,
        split="val",
        progress=False,
        prediction_cache=True,
    )
    assert direct.cache_info["status"] == "fresh"
    assert calls == 1

    comparison = model.compare(
        exported,
        split="val",
        progress=False,
        destination=tmp_path / "shared-semantic-comparison",
    )
    assert comparison.ranking.iloc[0]["dice"] == pytest.approx(1.0)
    assert calls == 1
    initial_entry = Path(direct.cache_info["location"])
    shutil.rmtree(initial_entry / "raw-result")

    reused = Model(
        checkpoint,
        name="renamed-semantic",
        task="semantic",
        resolution=64,
    ).predict(
        exported,
        split="val",
        progress=False,
        prediction_cache=True,
    )
    assert reused.cache_info["status"] == "legacy-hit"
    assert calls == 1
    cache_entry = Path(reused.cache_info["location"])
    assert (cache_entry / "raw-result").is_dir()
    compatible_entry = cache_entry.parent / reused.cache_info["compatible_key"]
    assert (compatible_entry / "evaluation.json").is_file()


def test_completed_comparison_inference_is_durably_reused_across_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exported = _semantic_export(tmp_path)
    checkpoint = tmp_path / "durable-semantic.pt"
    checkpoint.write_bytes(b"durable-semantic")
    calls = 0

    def fake_predict_inputs(_model, inputs, **_kwargs):
        nonlocal calls
        calls += 1
        values = {}
        for value in inputs:
            probability = np.full(
                (value.height, value.width),
                0.8,
                dtype=np.float32,
            )
            values[value.image_id] = types.SimpleNamespace(
                class_map=(probability >= 0.5).astype(np.uint8),
                foreground_probability=probability,
                probability_source="model-probabilities",
            )
        return values, "semantic_segment", {"synthetic": True}

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.predict_model_inputs",
        fake_predict_inputs,
    )
    Model(
        checkpoint,
        task="semantic",
        resolution=64,
        prediction_threshold=0.6,
    ).compare(
        exported,
        split="val",
        progress=True,
        destination=tmp_path / "comparison-at-0p6",
    )
    first_output = capsys.readouterr().out
    assert "Publishing prediction cache:" in first_output
    assert "Prediction cache published:" in first_output
    assert calls == 1

    cache_root = default_cache_root(exported.location) / "semantic"
    published = []
    for manifest_path in cache_root.glob("*/raw-result/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("identity") or {}).get("model_hash") != Model(
            checkpoint,
            task="semantic",
            resolution=64,
        ).hash():
            continue
        published.append(manifest_path)
        assert len(manifest["records"]) == 2
        assert all(
            record.get("foreground_probability")
            for record in manifest["records"]
        )
        assert (manifest_path.parent / "complete.json").is_file()
    assert len(published) == 1

    Model(
        checkpoint,
        task="semantic",
        resolution=64,
        prediction_threshold=0.7,
    ).compare(
        exported,
        split="val",
        progress=True,
        destination=tmp_path / "comparison-at-0p7",
    )
    second_output = capsys.readouterr().out
    assert "Prediction cache hit:" in second_output
    assert "running inference" not in second_output
    assert calls == 1


def test_nnunet_without_a_device_uses_runtime_device_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = _semantic_export(tmp_path)
    folder = _nnunet_model(tmp_path / "automatic-device-model")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _fake_nnunet_run(command, **kwargs)

    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.shutil.which",
        lambda command: f"/fake/{command}",
    )
    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "dataset_fixer.model._default_nnunet_device",
        lambda: "mps",
    )

    model = Model(folder, workers=1)
    result = model.predict(exported, split="val", progress=False)

    assert model.device is None
    assert result.settings["device"] == "mps"
    assert commands[0][commands[0].index("-device") + 1] == "mps"


def test_nnunet_sahi_stitches_tile_probabilities_at_native_and_canonical_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sahi")
    exported = _semantic_export(tmp_path)
    folder = _nnunet_model(tmp_path / "semantic-sahi-model")
    session = _install_fake_session(monkeypatch)

    model = Model(folder, upscale_factor=2, workers=1)
    result = model.predict(
        exported,
        split="val",
        inference="sahi",
        sahi_slice_height=16,
        sahi_slice_width=16,
        sahi_overlap=0.25,
        progress=False,
        _keep_native=True,
    )

    assert result.backend == "sahi"
    assert all(record.mask.shape == (30, 40) for record in result.records)
    assert all(record.native_mask.shape == (60, 80) for record in result.records)
    assert all(np.all(record.mask == 1) for record in result.records)
    assert result.settings["sahi_stitching"] == "feathered-probabilities"
    assert session.released == 1


def test_nnunet_sahi_runs_in_process_without_the_prediction_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sahi")
    exported = _semantic_export(tmp_path)
    folder = _nnunet_model(tmp_path / "in-process-model")
    _install_fake_session(monkeypatch)

    def refuse(command: list[str], **_: object) -> None:
        raise AssertionError(f"sliced prediction must not shell out: {command}")

    monkeypatch.setattr("dataset_fixer.semantic_comparison.subprocess.run", refuse)
    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.shutil.which",
        lambda command: None,
    )

    result = Model(folder, upscale_factor=1, workers=1).predict(
        exported,
        split="val",
        inference="sahi",
        sahi_slice_height=16,
        sahi_slice_width=16,
        sahi_overlap=0.25,
        progress=False,
    )

    assert result.backend == "sahi"
    assert len(result.records) == 2
    assert result.settings["nnunet_tta"] is False


def test_nnunet_sahi_tta_can_be_enabled_from_model_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sahi")
    exported = _semantic_export(tmp_path)
    folder = _nnunet_model(tmp_path / "tta-model")
    session = _install_fake_session(monkeypatch)
    models = Model.load_many({"nnunet": {"path": folder, "upscale_factor": 1}})

    assert models["nnunet"].nnunet_tta is False
    with pytest.raises(ValueError, match="nnunet_tta must be a boolean"):
        models.configure({"nnunet": {"nnunet_tta": "false"}})
    configured = models.configure({"nnunet": {"nnunet_tta": True}})
    result = configured["nnunet"].predict(
        exported,
        split="val",
        inference="sahi",
        sahi_slice_height=16,
        sahi_slice_width=16,
        sahi_overlap=0.25,
        progress=False,
    )

    assert configured["nnunet"].nnunet_tta is True
    assert configured["nnunet"].describe()["nnunet_tta"] is True
    assert session.use_tta is True
    assert result.settings["nnunet_tta"] is True


def test_nnunet_sahi_groups_equally_shaped_tiles_into_real_minibatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sahi")
    exported = _semantic_export(tmp_path)
    folder = _nnunet_model(tmp_path / "batched-model")
    session = _install_fake_session(monkeypatch)

    result = Model(folder, upscale_factor=1, workers=1).predict(
        exported,
        split="val",
        inference="sahi",
        sahi_slice_height=16,
        sahi_slice_width=16,
        sahi_overlap=0.25,
        progress=False,
    )

    tiles = sum(record.metadata["tile_count"] for record in result.records)
    assert tiles == sum(session.batch_sizes)
    # Tiles are evaluated in real minibatches, not one network call per tile.
    assert max(session.batch_sizes) > 1
    assert len(session.batch_sizes) < tiles
    assert result.settings["nnunet_execution_engine"] == "in-process-minibatched"
    assert result.settings["nnunet_tiles"] == tiles
    assert result.settings["nnunet_sources"] == 2
    assert set(result.settings["nnunet_phase_seconds"]) == {
        "preprocess",
        "inference",
        "probability_conversion",
        "stitch",
    }


def test_nnunet_sahi_progress_counts_every_tile_and_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("sahi")
    exported = _semantic_export(tmp_path)
    folder = _nnunet_model(tmp_path / "progress-model")
    _install_fake_session(monkeypatch)
    bars: list[dict] = []

    class RecordingBar:
        def __init__(self, **kwargs: object) -> None:
            self.record = {"total": kwargs.get("total"), "unit": kwargs.get("unit"), "seen": 0}
            bars.append(self.record)

        def update(self, count: int = 1) -> None:
            self.record["seen"] += count

        def close(self) -> None:
            self.record["closed"] = True

    monkeypatch.setattr("dataset_fixer.semantic_comparison.tqdm", RecordingBar)
    result = Model(folder, upscale_factor=1, workers=1).predict(
        exported,
        split="val",
        inference="sahi",
        sahi_slice_height=16,
        sahi_slice_width=16,
        sahi_overlap=0.25,
        progress=True,
    )

    tiles = sum(record.metadata["tile_count"] for record in result.records)
    by_unit = {bar["unit"]: bar for bar in bars}
    assert by_unit["tile"]["total"] == tiles
    assert by_unit["tile"]["seen"] == tiles
    assert by_unit["image"]["total"] == 2
    assert by_unit["image"]["seen"] == 2
    assert all(bar.get("closed") for bar in bars)
    output = capsys.readouterr().out
    assert "nnU-Net SAHI inference: 0/" in output
    assert "Preparing first work group on CPU" in output
    assert "First work group preprocessed" in output
    assert "nnU-Net SAHI progress:" in output
    assert "active batch cap" in output
    assert "actual forward batches" in output


def test_nnunet_sahi_releases_the_session_when_prediction_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sahi")
    exported = _semantic_export(tmp_path)
    folder = _nnunet_model(tmp_path / "interrupted-model")
    session = _install_fake_session(monkeypatch)

    def cancel(prepared, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(session, "predict_logits", cancel)

    with pytest.raises(KeyboardInterrupt):
        Model(folder, upscale_factor=1, workers=1).predict(
            exported,
            split="val",
            inference="sahi",
            sahi_slice_height=16,
            sahi_slice_width=16,
            sahi_overlap=0.25,
            progress=False,
        )

    assert session.released == 1


def _install_fake_session(
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: object,
) -> FakeSession:
    session = FakeSession(**kwargs)

    def load_fake_session(**options: object) -> FakeSession:
        session.use_tta = bool(options.get("use_tta", False))
        return session

    monkeypatch.setattr(
        "dataset_fixer.nnunet_engine.load_session",
        load_fake_session,
    )
    return session


def test_semantic_export_compares_official_nnunet_model_folders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exported = _semantic_export(tmp_path)
    perfect = _nnunet_model(tmp_path / "perfect-model")
    weak = _nnunet_model(tmp_path / "weak-model")
    commands: list[list[str]] = []
    command_options: list[dict[str, object]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        command_options.append(kwargs)
        return _fake_nnunet_run(command, **kwargs)

    monkeypatch.setattr("dataset_fixer.semantic_comparison.shutil.which", lambda command: f"/fake/{command}")
    monkeypatch.setattr("dataset_fixer.semantic_comparison.subprocess.run", fake_run)
    monkeypatch.setattr("dataset_fixer.semantic_comparison.environment_snapshot", lambda: {"test": True})
    destination = tmp_path / "comparison"
    models = Model.load_many(
        {
            # Passing fold_0 mirrors the notebook's FOLD_OUTPUT value. The API
            # normalizes it to the complete trained-model folder automatically.
            "perfect": {
                "path": perfect / "fold_0",
                "upscale_factor": 2,
                "workers": 1,
            },
            "weak": {
                "path": weak,
                "upscale_factor": 1,
                "workers": 1,
                "nnunet_tta": True,
            },
        }
    )
    result = models.compare(
        exported,
        progress=True,
        destination=destination,
    )
    progress_output = capsys.readouterr().err

    assert isinstance(result, SemanticComparisonResult)
    assert "Finalizing comparison report" in progress_output
    assert "bootstrapping 1 model pair" in progress_output
    assert "Bootstrapping model pairs" in progress_output
    assert "Extracting reference components" in progress_output
    assert "Scoring object sizes" in progress_output
    assert "writing and publishing report" in progress_output
    assert result.cohort_verified
    assert result.ranking.iloc[0]["model"] == "perfect"
    assert result.ranking.iloc[0]["score"] == pytest.approx(1.0)
    assert result.ranking.iloc[1]["score"] == pytest.approx(0.0)
    assert result.ranking.iloc[0]["upscale_factor"] == 2
    assert result.ranking.iloc[1]["upscale_factor"] == 1
    assert len(result.ranking.iloc[0]["model_hash"]) == 8
    assert len(commands) == 6
    assert all(options["capture_output"] is False for options in command_options)
    predict = commands[0]
    assert predict[0] == "nnUNetv2_predict_from_modelfolder"
    assert predict[predict.index("-m") + 1] == str(perfect)
    assert predict[predict.index("-f") + 1] == "0"
    assert predict[predict.index("-chk") + 1] == "checkpoint_final.pth"
    assert predict[predict.index("-device") + 1] == models["perfect"]._resolved_device()
    assert "--save_probabilities" in predict
    assert "--disable_progress_bar" in predict
    assert "--disable_tta" in predict
    weak_predict = next(
        command
        for command in commands
        if command[0] == "nnUNetv2_predict_from_modelfolder"
        and command[command.index("-m") + 1] == str(weak)
    )
    assert "--disable_tta" not in weak_predict
    assert not (destination / "cohort").exists()
    assert not (destination / "predictions").exists()
    assert not list(destination.rglob("*.csv"))
    assert not list(destination.rglob("*.jsonl"))
    assert (destination / "reports" / "plots.png").is_file()
    assert (destination / "reports" / "metric-breakdown.png").is_file()
    assert (destination / "reports" / "comparison.png").is_file()
    assert not (destination / "figures").exists()
    assert not (destination / "qualitative").exists()
    manifest = json.loads((destination / "reports" / "result.json").read_text())
    assert manifest["backend"] == "native"
    assert manifest["adapter"] == "nnunetv2-official"
    assert manifest["schema"] == SEMANTIC_REPORT_SCHEMA
    assert manifest["cases"] == 2
    assert manifest["case_composition"] == {"positive": 2, "empty": 0, "total": 2}
    assert "micro_iou" in manifest["metric_definitions"]
    assert manifest["reports"]["metric_breakdown"] == "reports/metric-breakdown.png"
    assert manifest["reports"]["object_size_breakdown"] == (
        "reports/object-size-breakdown.png"
    )
    assert manifest["object_size_analysis"]["status"] == "complete"
    assert "small_object_dice" in manifest["metric_definitions"]
    assert "upscale_factor" not in manifest["settings"]
    assert [model["upscale_factor"] for model in manifest["settings"]["models"]] == [2, 1]
    assert result.ranking.iloc[0]["projection"] == "probability-area-pool-argmax"
    assert result.ranking.iloc[0]["cohort_cases"] == 2
    assert result.ranking.iloc[0]["support_cases"] == 2
    assert result.ranking.iloc[0]["native_dice"] == pytest.approx(1.0)
    assert manifest["worst_cases"]
    command_count = len(commands)
    manifest["schema"] = 12
    (destination / "reports" / "result.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    regenerated = models.compare(
        exported,
        progress=False,
        destination=destination,
    )
    assert len(commands) == command_count
    assert regenerated.ranking["cache"].eq("hit").all()
    assert json.loads(
        (destination / "reports" / "result.json").read_text()
    )["schema"] == SEMANTIC_REPORT_SCHEMA

    cached = models.compare(
        exported,
        progress=False,
        destination=tmp_path / "comparison-from-cache",
    )
    assert len(commands) == command_count
    assert cached.ranking["cache"].eq("hit").all()

    equal = models.compare(
        exported,
        progress=False,
        destination=tmp_path / "comparison-all-pairs",
    )
    equal_manifest = json.loads(
        (equal.location / "reports" / "result.json").read_text()
    )
    assert "baseline" not in equal_manifest
    assert "paired_comparisons" not in equal_manifest["settings"]
    assert equal_manifest["paired_statistics"]
    assert "baseline" not in equal_manifest["paired_statistics"][0]
    assert equal_manifest["paired_statistics"][0]["model_a"] == "perfect"
    assert equal_manifest["paired_statistics"][0]["model_b"] == "weak"
    assert len(commands) == command_count


def test_repeating_a_sahi_comparison_reuses_cached_predictions_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sahi")
    exported = _semantic_export(tmp_path)
    folder = _nnunet_model(tmp_path / "cached-sahi-model")
    session = _install_fake_session(monkeypatch)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object):
        commands.append(command)
        return _fake_nnunet_run(command, **kwargs)

    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.shutil.which",
        lambda command: f"/fake/{command}",
    )
    monkeypatch.setattr("dataset_fixer.semantic_comparison.subprocess.run", fake_run)
    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.environment_snapshot",
        lambda: {"test": True},
    )
    models = Model.load_many(
        {
            "sliced": {
                "path": folder,
                "upscale_factor": 1,
                "workers": 1,
                "inference": "sahi",
                "sahi_slice_height": 16,
                "sahi_slice_width": 16,
                "sahi_overlap": 0.25,
            }
        }
    )

    first = models.compare(exported, progress=False, destination=tmp_path / "sliced-a")
    assert first.ranking["cache"].eq("fresh").all()
    inference_calls = len(session.batch_sizes)
    evaluations = len(commands)
    assert inference_calls > 0
    assert evaluations > 0

    # Same models, same settings, same cohort: a different destination still
    # reuses the per-model prediction and metric cache under <dataset>/.cache.
    second = models.compare(exported, progress=False, destination=tmp_path / "sliced-b")

    assert second.ranking["cache"].eq("hit").all()
    assert len(session.batch_sizes) == inference_calls, "re-ran network inference"
    assert len(commands) == evaluations, "re-ran the official evaluator"
    assert second.ranking.iloc[0]["score"] == pytest.approx(
        first.ranking.iloc[0]["score"]
    )

    # Repeating into the same destination short-circuits before any per-model work.
    again = models.compare(exported, progress=False, destination=tmp_path / "sliced-b")
    assert len(session.batch_sizes) == inference_calls
    assert again.ranking.iloc[0]["score"] == pytest.approx(
        first.ranking.iloc[0]["score"]
    )

    # Device, worker count, and batching change execution only. They must not
    # create another prediction identity for the same model and cohort.
    changed_execution = models.configure(
        {"sliced": {"device": "cpu", "workers": 3, "batch_size": 1}}
    )
    execution_rerun = changed_execution.compare(
        exported,
        progress=False,
        destination=tmp_path / "sliced-execution-changed",
    )
    assert execution_rerun.ranking["cache"].eq("hit").all()
    assert len(session.batch_sizes) == inference_calls, "re-ran network inference"
    assert len(commands) == evaluations, "re-ran the official evaluator"

    # TTA changes model predictions and therefore owns a separate cache. Going
    # back to non-TTA must still recover the original completed cache.
    tta_models = models.configure({"sliced": {"nnunet_tta": True}})
    tta_result = tta_models.compare(
        exported,
        progress=False,
        # Reusing an explicit destination must still validate the complete
        # report's settings fingerprint before short-circuiting.
        destination=tmp_path / "sliced-b",
    )
    assert tta_result.ranking["cache"].eq("fresh").all()
    tta_inference_calls = len(session.batch_sizes)
    assert tta_inference_calls > inference_calls

    non_tta_again = models.compare(
        exported,
        progress=False,
        destination=tmp_path / "sliced-b",
    )
    assert non_tta_again.ranking["cache"].eq("hit").all()
    assert len(session.batch_sizes) == tta_inference_calls


def test_changing_a_sahi_setting_invalidates_the_cached_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sahi")
    exported = _semantic_export(tmp_path)
    folder = _nnunet_model(tmp_path / "invalidated-model")
    session = _install_fake_session(monkeypatch)
    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.shutil.which",
        lambda command: f"/fake/{command}",
    )
    monkeypatch.setattr("dataset_fixer.semantic_comparison.subprocess.run", _fake_nnunet_run)
    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.environment_snapshot",
        lambda: {"test": True},
    )

    def compare(overlap: float, destination: str):
        models = Model.load_many(
            {
                "sliced": {
                    "path": folder,
                    "upscale_factor": 1,
                    "workers": 1,
                    "inference": "sahi",
                    "sahi_slice_height": 16,
                    "sahi_slice_width": 16,
                    "sahi_overlap": overlap,
                }
            }
        )
        return models.compare(
            exported, progress=False, destination=tmp_path / destination
        )

    compare(0.25, "overlap-a")
    calls = len(session.batch_sizes)
    result = compare(0.5, "overlap-b")

    assert result.ranking["cache"].eq("fresh").all()
    assert len(session.batch_sizes) > calls


def test_semantic_comparison_failure_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = _semantic_export(tmp_path)
    model = _nnunet_model(tmp_path / "failing-model")
    monkeypatch.setattr("dataset_fixer.semantic_comparison.shutil.which", lambda command: f"/fake/{command}")

    def fail(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(2, command, stderr="synthetic failure")

    monkeypatch.setattr("dataset_fixer.semantic_comparison.subprocess.run", fail)
    destination = tmp_path / "failed-comparison"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        models = Model.load_many({"broken": {"path": model, "workers": 1}})
        models.compare(
            exported,
            progress=False,
            destination=destination,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".failed-comparison.building-*"))


def test_semantic_comparison_requires_binary_png_model_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = _semantic_export(tmp_path)
    model = _nnunet_model(tmp_path / "multiclass-model")
    dataset_json = json.loads((model / "dataset.json").read_text())
    dataset_json["labels"]["reef"] = 2
    (model / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    monkeypatch.setattr("dataset_fixer.semantic_comparison.shutil.which", lambda command: f"/fake/{command}")

    with pytest.raises(DatasetValidationError, match="requires binary nnU-Net labels"):
        models = Model.load_many({"multiclass": {"path": model, "workers": 1}})
        models.compare(
            exported,
            progress=False,
            destination=tmp_path / "invalid-comparison",
        )


def test_semantic_comparison_validates_model_specific_upscale_factor(tmp_path: Path) -> None:
    exported = _semantic_export(tmp_path)
    model = _nnunet_model(tmp_path / "bad-scale-model")

    with pytest.raises(DatasetValidationError, match="upscale_factor must be a positive integer"):
        models = Model.load_many(
            {"bad-scale": {"path": model, "upscale_factor": 0, "workers": 1}},
        )
        models.compare(
            exported,
            progress=False,
            destination=tmp_path / "invalid-scale-comparison",
        )


def test_prediction_plots_are_bounded_and_skip_empty_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the cases the report keeps are drawn, and empty ones are skipped."""

    from dataset_fixer.semantic_comparison import _render_semantic_prediction_grids

    root = tmp_path / "grids"
    predictions = root / "model-a"
    predictions.mkdir(parents=True)
    cases = []
    for index in range(4):
        case_id = f"val_{index:06d}"
        image = tmp_path / f"{case_id}.png"
        Image.new("RGB", (32, 32), (40, 90, 140)).save(image)
        mask_path = tmp_path / f"{case_id}_mask.png"
        # Case 0 has a reference, the rest are empty ground truth.
        truth = np.zeros((32, 32), dtype=np.uint8)
        if index == 0:
            truth[8:20, 8:20] = 255
        Image.fromarray(truth).save(mask_path)
        # Case 1 has no reference but a false-positive prediction.
        prediction = np.zeros((32, 32), dtype=np.uint8)
        if index in (0, 1):
            prediction[10:18, 10:18] = 1
        Image.fromarray(prediction).save(predictions / f"{case_id}.png")
        cases.append(
            _SemanticCase(
                case_id=case_id,
                relative_path=Path(f"{case_id}.png"),
                image_path=image,
                mask_path=mask_path,
                width=32,
                height=32,
                image_sha256="x",
                mask_sha256="y",
            )
        )
    rows = {
        "model-a": [
            {"case_id": case.case_id, "dice": 0.5, "iou": 0.4} for case in cases
        ]
    }

    # Ask for three cases; case 2 has neither reference nor prediction.
    rendered = _render_semantic_prediction_grids(
        root,
        cases,
        {"model-a": predictions},
        rows,
        case_ids=["val_000000", "val_000001", "val_000002"],
    )

    assert len(rendered) == 2
    written = {path.name for path in (root / "predictions").rglob("*.png")}
    assert written == {"val_000000.png", "val_000001.png"}
    # Case 3 was never requested, case 2 was requested but had nothing to show.
    assert "val_000003.png" not in written
    assert "val_000002.png" not in written


def test_comparison_renders_only_the_cases_it_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sahi")
    exported = _semantic_export(tmp_path)
    folder = _nnunet_model(tmp_path / "bounded-plots-model")
    _install_fake_session(monkeypatch)
    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.shutil.which",
        lambda command: f"/fake/{command}",
    )
    monkeypatch.setattr("dataset_fixer.semantic_comparison.subprocess.run", _fake_nnunet_run)
    monkeypatch.setattr(
        "dataset_fixer.semantic_comparison.environment_snapshot",
        lambda: {"test": True},
    )
    models = Model.load_many(
        {
            "sliced": {
                "path": folder,
                "upscale_factor": 1,
                "workers": 1,
                "inference": "sahi",
                "sahi_slice_height": 16,
                "sahi_slice_width": 16,
                "sahi_overlap": 0.25,
            }
        }
    )
    destination = tmp_path / "bounded-plots"

    models.compare(
        exported,
        progress=False,
        save_prediction_plots=True,
        destination=destination,
    )

    manifest = json.loads((destination / "reports" / "result.json").read_text())
    reported = {Path(p).name for p in manifest["reports"]["prediction_plots"]}
    written = {p.name for p in (destination / "predictions").rglob("*.png")}
    assert written == reported
    assert len(written) <= len(manifest["worst_cases"])
