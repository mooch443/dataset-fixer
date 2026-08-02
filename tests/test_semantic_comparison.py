from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset_fixer import (
    Dataset,
    DatasetValidationError,
    Model,
    PredictionResult,
    SemanticComparisonResult,
    SemanticMaskExport,
    SemanticModelCohort,
)
from dataset_fixer.semantic_comparison import _SemanticCase, _canonicalize_predictions
from conftest import make_yolo_dataset


def _semantic_export(tmp_path: Path) -> SemanticMaskExport:
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
    assert isinstance(exported, SemanticMaskExport)
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
    assert isinstance(exported, SemanticMaskExport)
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

    result = exported.compare_models(
        {"perfect": model},
        workers=1,
        bootstrap_resamples=10,
        visualize=False,
        progress=False,
        destination=tmp_path / "finite-support-comparison",
    )

    assert result.ranking[0]["cohort_cases"] == 2
    assert result.ranking[0]["support_cases"] == 1
    assert result.ranking[0]["undefined_cases"] == 1


def test_loaded_semantic_models_visualize_only_sampled_cases_with_shared_mask_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt

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
                "model_folder": perfect,
                "upscale_factor": 2,
            },
            "resenc-m-very-long-model-name-beta": {
                "model_folder": weak,
                "upscale_factor": 1,
            },
        },
        kind="nnunet",
        device="mps",
        workers=1,
    )

    assert isinstance(loaded, SemanticModelCohort)
    assert loaded.names == (
        "resenc-m-very-long-model-name-alpha",
        "resenc-m-very-long-model-name-beta",
    )
    figure = loaded.visualize(
        source=exported,
        samples=2,
        examples_per_row=1,
        panel_size=2.0,
        model_title_length=18,
        progress=False,
        destination=tmp_path / "quick-comparison.png",
    )

    assert (tmp_path / "quick-comparison.png").is_file()
    assert len(commands) == 2
    assert all(command[0] == "nnUNetv2_predict_from_modelfolder" for command in commands)
    assert all(command[command.index("-device") + 1] == "mps" for command in commands)
    assert len(figure.axes) == 11  # filenames, one shared heading row, and panels
    headings = [text.get_text() for text in figure.axes[1].texts]
    assert headings[:2] == ["Original", "GT"]
    assert "…" in headings[2]
    assert figure.axes[4].get_xlabel().startswith("Dice=")
    assert np.asarray(figure.axes[4].images[0].get_array()).ndim == 2
    assert figure.axes[6].texts[0].get_text().endswith(".jpg")
    plt.close(figure)


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


def test_semantic_export_compares_official_nnunet_model_folders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    result = exported.compare_models(
        {
            # Passing fold_0 mirrors the notebook's FOLD_OUTPUT value. The API
            # normalizes it to the complete trained-model folder automatically.
            "perfect": {
                "model_folder": perfect / "fold_0",
                "upscale_factor": 2,
            },
            "weak": {
                "model_folder": weak,
                "upscale_factor": 1,
            },
        },
        baseline="perfect",
        workers=1,
        bootstrap_resamples=20,
        seed=7,
        visualize=True,
        progress=True,
        destination=destination,
    )

    assert isinstance(result, SemanticComparisonResult)
    assert result.cohort_verified
    assert result.ranking[0]["model"] == "perfect"
    assert result.ranking[0]["score"] == pytest.approx(1.0)
    assert result.ranking[1]["score"] == pytest.approx(0.0)
    assert result.ranking[0]["upscale_factor"] == 2
    assert result.ranking[1]["upscale_factor"] == 1
    assert len(result.ranking[0]["model_sha256"]) == 64
    assert result.baseline == "perfect"
    assert len(commands) == 6
    assert all(options["capture_output"] is True for options in command_options)
    predict = commands[0]
    assert predict[0] == "nnUNetv2_predict_from_modelfolder"
    assert predict[predict.index("-m") + 1] == str(perfect)
    assert predict[predict.index("-f") + 1] == "0"
    assert predict[predict.index("-chk") + 1] == "checkpoint_final.pth"
    assert predict[predict.index("-device") + 1] == "cuda"
    assert "--save_probabilities" in predict
    cohort = [json.loads(line) for line in (destination / "evaluation-cohort.jsonl").read_text().splitlines()]
    assert len(cohort) == 2
    assert cohort[0]["width"] == 40
    assert cohort[0]["height"] == 30
    assert cohort[0]["projection"] == "probability-area-pool-argmax"
    assert cohort[0]["model_inputs"]["perfect"]["prepared_width"] == 80
    assert cohort[0]["model_inputs"]["weak"]["prepared_width"] == 40
    assert not (destination / "cohort").exists()
    assert len(list((destination / "predictions" / "perfect").glob("*.png"))) == 2
    with Image.open(destination / "predictions" / "perfect" / "val_000000.png") as prediction:
        assert prediction.size == (40, 30)
    assert (destination / "ranking.csv").is_file()
    assert (destination / "per-case.csv").is_file()
    assert (destination / "paired-statistics.csv").is_file()
    assert (destination / "ranking.png").is_file()
    assert (destination / "comparison.png").is_file()
    assert not (destination / "ranking.pdf").exists()
    assert not (destination / "ranking.svg").exists()
    assert not (destination / "figures").exists()
    assert not (destination / "qualitative").exists()
    assert not (destination / "reports").exists()
    manifest = json.loads((destination / "semantic-model-comparison.json").read_text())
    assert manifest["backend"] == "nnunetv2-official"
    assert manifest["schema"] == 3
    assert manifest["figures"] == ["ranking.png"]
    assert manifest["qualitative"] == ["comparison.png"]
    assert manifest["cases"] == 2
    assert "upscale_factor" not in manifest["settings"]
    assert [model["upscale_factor"] for model in manifest["settings"]["models"]] == [2, 1]
    assert result.ranking[0]["projection"] == "probability-area-pool-argmax"
    assert result.ranking[0]["cohort_cases"] == 2
    assert result.ranking[0]["support_cases"] == 2
    assert result.ranking[0]["native_dice"] == pytest.approx(1.0)
    official = json.loads((destination / "perfect-metrics.json").read_text())
    assert official["metric_per_case"][0]["reference_file"].startswith("dataset://val/masks/0/")
    assert official["metric_per_case"][0]["prediction_file"].startswith("predictions/perfect/")
    assert official["projection"] == "probability-area-pool-argmax"
    native = json.loads((destination / "perfect-native-metrics.json").read_text())
    assert native["foreground_mean"]["Dice"] == pytest.approx(1.0)
    assert native["metric_per_case"][0]["prediction_file"] is None
    assert native["projection"] == "none"


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
        exported.compare_models(
            {"broken": model},
            workers=1,
            bootstrap_resamples=10,
            visualize=False,
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
        exported.compare_models(
            {"multiclass": model},
            workers=1,
            bootstrap_resamples=10,
            visualize=False,
            progress=False,
            destination=tmp_path / "invalid-comparison",
        )


def test_semantic_comparison_validates_model_specific_upscale_factor(tmp_path: Path) -> None:
    exported = _semantic_export(tmp_path)
    model = _nnunet_model(tmp_path / "bad-scale-model")

    with pytest.raises(DatasetValidationError, match="upscale_factor must be a positive integer"):
        exported.compare_models(
            {"bad-scale": {"model_folder": model, "upscale_factor": 0}},
            workers=1,
            bootstrap_resamples=10,
            visualize=False,
            progress=False,
            destination=tmp_path / "invalid-scale-comparison",
        )
