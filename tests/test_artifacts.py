from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image

from dataset_fixer import Dataset, DatasetTrace, DatasetValidationError
from dataset_fixer.artifacts import (
    CANONICAL_REPORT_FILES,
    DATASET_INFO_SCHEMA,
    split_image_summary,
)
from dataset_fixer.dataset_report import render_dataset_report
from dataset_fixer.validation_audit import stage_load_validation_audit
from conftest import make_yolo_dataset


def test_compact_dataset_artifacts_and_exact_trace(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.25 0.25"],
        val_rows=["0 0.5 0.5 0.25 0.25"],
        size=(96, 80),
    )
    exported = (
        Dataset.open(source, task="detect", progress=False)
        .tile(tile_size=64, overlap=0.25, visualize=True)
        .export(destination=tmp_path / "tiles", visualize=True, progress=False)
    )

    reports = exported.location / "reports"
    assert (reports / "dataset-info.json").is_file()
    assert (reports / "source.json").is_file()
    assert (reports / "lineage.json.gz").is_file()
    assert (reports / "plots.png").is_file()
    assert not (reports / "source-operation.png").exists()
    assert not list(exported.location.rglob("*.jsonl"))
    assert not list(exported.location.rglob("*.csv"))
    assert not (exported.location / "dataset-fixer.json").exists()
    assert not (exported.location / "provenance.jsonl").exists()

    info = json.loads((reports / "dataset-info.json").read_text(encoding="utf-8"))
    source_info = json.loads((reports / "source.json").read_text(encoding="utf-8"))
    assert info["dataset_id"]
    assert set(info["split_summary"]) == {"train", "val"}
    for details in info["split_summary"].values():
        assert details["total_images"] == (
            details["labeled_images"] + details["background_images"]
        )
    assert source_info["id"]
    assert source_info["path"]
    with gzip.open(reports / "lineage.json.gz", "rt", encoding="utf-8") as handle:
        lineage = json.load(handle)
    assert len(lineage["records"]) == len(exported._samples)

    trace = exported.trace()
    assert isinstance(trace, DatasetTrace)
    assert trace.datasets[0].dataset_id == info["dataset_id"]
    assert trace.datasets[-1].path == source.resolve()
    assert trace.datasets[-1].present
    assert trace.tiles
    assert trace.tiles[0].resolved_original_image is not None
    assert trace.for_sample(trace.samples[0].output_image) == trace.samples[0]
    assert "tile(s)" in trace.summary()


def test_trace_resolves_moved_source_with_path_rewrite(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "old" / "source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.25 0.25"],
        val_rows=["0 0.5 0.5 0.25 0.25"],
    )
    exported = Dataset.open(source, task="detect", progress=False).export(
        destination=tmp_path / "derived",
        visualize=False,
        progress=False,
    )
    moved = tmp_path / "new" / "source"
    moved.parent.mkdir()
    shutil.move(str(source), moved)

    trace = exported.trace(path_rewrites={source: moved})
    assert trace.datasets[-1].path == moved.resolve()
    assert trace.datasets[-1].present
    assert all(
        sample.resolved_original_image is not None
        for sample in trace.samples
        if sample.original_image
    )


def test_generated_reports_hold_only_canonical_artifact_files(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.25 0.25", "0 0.4 0.4 0.2 0.2"],
        val_rows=["0 0.5 0.5 0.25 0.25"],
        size=(96, 80),
    )
    exported = (
        Dataset.open(source, task="detect", progress=False)
        .tile(tile_size=64, overlap=0.25, visualize=True)
        .export(destination=tmp_path / "tiles", visualize=True, progress=False)
    )

    reports = exported.location / "reports"
    assert {path.name for path in reports.iterdir()} == set(CANONICAL_REPORT_FILES)
    assert not any(path.is_dir() for path in reports.iterdir())
    assert not (exported.location / "coverage_summary").exists()
    # Operation audits survive the pruning as structured manifest data.
    assert exported.manifest["audits"]


def test_dataset_report_shows_split_composition_and_deterministic_examples(
    tmp_path: Path,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.25 0.25", "", "0 0.3 0.3 0.2 0.2", ""],
        val_rows=["0 0.5 0.5 0.25 0.25", ""],
    )
    exported = Dataset.open(source, task="detect", progress=False).export(
        destination=tmp_path / "derived",
        visualize=False,
        progress=False,
    )

    output = exported.location / "reports" / "plots.png"
    with Image.open(output) as report:
        assert report.width == 2400
        assert report.height > 900
        colors = set(report.convert("RGB").getdata())
    # One annotated/background composition bar per split, in report colors.
    assert (47, 158, 95) in colors
    assert (185, 194, 205) in colors

    rerendered = tmp_path / "again.png"
    render_dataset_report(
        exported.location,
        _lineage_records(exported.location),
        name=exported.name,
        task="detect",
        format_name="yolo",
        classes=exported.classes,
        output=rerendered,
    )
    assert rerendered.read_bytes() == output.read_bytes()


def test_report_captions_are_trimmed_to_fit_their_column() -> None:
    from dataset_fixer.dataset_report import _fit, _fit_middle, _font, _text_width

    font = _font(22)
    caption = (
        "val/images/0549e88a-741eaa7d-0cee-4225-87f1-58e5df816aa3"
        "_20220223_044202_89_2428_3B_Analyt_1vtlT3h.png  ·  annotated"
    )
    fitted = _fit_middle(caption, font, 540, keep_suffix=len("annotated") + 5)

    assert _text_width(fitted, font) <= 540
    # The distinguishing tail survives; the long shared prefix gives way.
    assert fitted.endswith("·  annotated")
    assert fitted.startswith("val/images/0549e88a")
    assert "…" in fitted

    short = "val/images/a.png  ·  annotated"
    assert _fit_middle(short, font, 540, keep_suffix=14) == short
    assert _text_width(_fit("x" * 400, font, 300), font) <= 300


def test_semantic_dataset_report_overlays_masks_in_red(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "polygons",
        task="segment",
        names=["fish"],
        train_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
    )
    masks = Dataset.open(source, task="segment", progress=False).export(
        destination=tmp_path / "masks",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )

    with Image.open(masks.location / "reports" / "plots.png") as report:
        pixels = report.convert("RGB").getdata()
    # The overlay blends toward pure red, so overlaid pixels are strongly warm.
    assert any(r > 150 and r > g + 60 and r > b + 60 for r, g, b in pixels)


def test_load_audit_does_not_embed_a_reopened_aggregate_plot(tmp_path: Path) -> None:
    aggregate = tmp_path / "source" / "reports" / "plots.png"
    aggregate.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "red").save(aggregate)
    reports = tmp_path / "derived" / "reports"

    detail, staged = stage_load_validation_audit(
        {
            "status": "passed_with_skips",
            "skipped_count": 1,
            "counts_by_category": {"invalid": 1},
        },
        aggregate,
        reports,
    )

    assert staged is None
    assert detail["visualization"] is None
    assert not (reports / "load_validation_examples.png").exists()


def test_split_image_summary_counts_labels_and_background_per_split() -> None:
    assert split_image_summary(
        [
            {"output_split": "train", "output_annotation_count": 2},
            {"output_split": "train", "output_annotation_count": 0},
            {
                "output_split": "val",
                "output_annotation_count": 3,
                "output_has_labels": False,
            },
        ]
    ) == {
        "train": {
            "total_images": 2,
            "labeled_images": 1,
            "background_images": 1,
        },
        "val": {
            "total_images": 1,
            "labeled_images": 0,
            "background_images": 1,
        },
    }


def test_dataset_update_in_place_and_to_string_destination(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "update-source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.25 0.25", ""],
        val_rows=["0 0.5 0.5 0.25 0.25"],
    )
    exported = Dataset.open(source, task="detect", progress=False).export(
        destination=tmp_path / "outdated",
        visualize=False,
        progress=False,
    )
    info_path = exported.location / "reports" / "dataset-info.json"
    old_info = json.loads(info_path.read_text(encoding="utf-8"))
    dataset_id = old_info["dataset_id"]
    old_info["schema_version"] = 2
    old_info.pop("split_summary")
    info_path.write_text(json.dumps(old_info), encoding="utf-8")
    (exported.location / "reports" / "legacy-audit.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    (exported.location / "dataset-fixer.json").write_text("{}", encoding="utf-8")
    (exported.location / "provenance.jsonl").write_text("{}\n", encoding="utf-8")
    payload_bytes = {
        path.relative_to(exported.location): path.read_bytes()
        for split in ("train", "val")
        for directory in ("images", "labels")
        for path in (exported.location / split / directory).rglob("*")
        if path.is_file()
    }

    updated = Dataset.open(exported.location, task="detect", progress=False).update()

    assert updated.location == exported.location
    assert updated.manifest["schema_version"] == DATASET_INFO_SCHEMA
    assert updated.manifest["dataset_id"] == dataset_id
    assert updated.manifest["split_summary"] == {
        "train": {
            "total_images": 2,
            "labeled_images": 1,
            "background_images": 1,
        },
        "val": {
            "total_images": 1,
            "labeled_images": 1,
            "background_images": 0,
        },
    }
    assert updated.manifest["audits"]["legacy-audit"] == {"status": "passed"}
    assert not (updated.location / "reports" / "legacy-audit.json").exists()
    assert not (updated.location / "dataset-fixer.json").exists()
    assert not (updated.location / "provenance.jsonl").exists()
    assert all((updated.location / path).read_bytes() == value for path, value in payload_bytes.items())

    destination = tmp_path / "updated-copy"
    copied = updated.update(dest=str(destination))

    assert copied.location == destination.resolve()
    assert copied.manifest["dataset_id"] == dataset_id
    assert copied.manifest["location"] == str(destination.resolve())
    assert "path" not in yaml.safe_load(copied.data_yaml.read_text(encoding="utf-8"))
    assert updated.location.is_dir()
    assert all((copied.location / path).read_bytes() == value for path, value in payload_bytes.items())


def test_dataset_update_removes_a_legacy_pinned_path_from_the_training_yaml(
    tmp_path: Path,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "legacy-source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.25 0.25"],
        val_rows=["0 0.5 0.5 0.25 0.25"],
    )
    exported = Dataset.open(source, task="detect", progress=False).export(
        destination=tmp_path / "legacy",
        visualize=False,
        progress=False,
    )
    legacy = yaml.safe_load(exported.data_yaml.read_text(encoding="utf-8"))
    exported.data_yaml.write_text(
        yaml.safe_dump({"path": str(exported.location), **legacy}, sort_keys=False),
        encoding="utf-8",
    )

    updated = Dataset.open(exported.location, task="detect", progress=False).update()

    assert "path" not in yaml.safe_load(updated.data_yaml.read_text(encoding="utf-8"))
    assert len(updated._samples) == len(exported._samples)


def test_dataset_update_progress_can_be_suppressed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = make_yolo_dataset(
        tmp_path / "quiet-source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.25 0.25"],
        val_rows=["0 0.5 0.5 0.25 0.25"],
    )
    exported = Dataset.open(source, task="detect", progress=False).export(
        destination=tmp_path / "quiet",
        visualize=False,
        progress=False,
    )
    payload_bytes = {
        path.relative_to(exported.location): path.read_bytes()
        for split in ("train", "val")
        for directory in ("images", "labels")
        for path in (exported.location / split / directory).rglob("*")
        if path.is_file()
    }

    capsys.readouterr()
    quiet = Dataset.open(exported.location, task="detect", progress=False).update(
        progress=False
    )
    silent = capsys.readouterr()
    assert silent.out == ""
    assert silent.err == ""

    loud = quiet.update(dest=tmp_path / "loud-copy", progress=True)
    noisy = capsys.readouterr()
    assert "Validating" in noisy.out or "Copying" in noisy.err

    for dataset in (quiet, loud):
        assert dataset.manifest["schema_version"] == DATASET_INFO_SCHEMA
        assert all(
            (dataset.location / path).read_bytes() == value
            for path, value in payload_bytes.items()
        )


def test_dataset_update_rejects_raw_and_virtual_datasets(tmp_path: Path) -> None:
    source = make_yolo_dataset(tmp_path / "raw", task="detect")
    dataset = Dataset.open(source, task="detect", progress=False)
    with pytest.raises(DatasetValidationError, match="requires a dataset-fixer manifest"):
        dataset.update()
    virtual = dataset.tile(tile_size=80, visualize=False, progress=False)
    with pytest.raises(DatasetValidationError, match="materialized dataset"):
        virtual.update()


def _lineage_records(root: Path) -> list[dict]:
    with gzip.open(root / "reports" / "lineage.json.gz", "rt", encoding="utf-8") as handle:
        return json.load(handle)["records"]


def test_yolo_and_semantic_exports_report_the_same_source_coverage(
    tmp_path: Path,
) -> None:
    """Both output formats must publish equivalent reports for one source.

    A mask export inherits its coverage statistic from the operation that
    produced the polygons, so it states the same numbers as the YOLO export
    rather than silently omitting the panel.
    """

    source = make_yolo_dataset(
        tmp_path / "source",
        task="segment",
        names=["school"],
        train_rows=[
            "0 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4",
            "0 0.6 0.6 0.8 0.6 0.8 0.8 0.6 0.8",
        ],
        val_rows=["0 0.3 0.3 0.6 0.3 0.6 0.6 0.3 0.6"],
        size=(600, 400),
    )
    plan = Dataset.open(source, task="segment", progress=False).tile(
        mode="coverage",
        tile_size=128,
        target_appearances_per_object=2,
        sparse_appearances_per_object=2,
        background_ratio=0,
        allow_lossy=True,
        visualize=False,
        progress=False,
    )
    vector = plan.export(
        destination=tmp_path / "vector", visualize=False, progress=False
    )
    masks = vector.export(
        destination=tmp_path / "masks",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )

    vector_coverage = vector.manifest["audits"]["coverage.source_coverage"]
    mask_coverage = masks.manifest["audits"]["coverage.source_coverage"]
    for key in (
        "source_labels",
        "source_labels_covered_at_least_once",
        "source_labels_never_covered",
        "source_label_coverage_percent",
        "source_image_space_coverage_percent",
    ):
        assert vector_coverage[key] == mask_coverage[key]

    # Both publish the same canonical files, and both plots carry the panel.
    assert {path.name for path in (vector.location / "reports").iterdir()} == {
        path.name for path in (masks.location / "reports").iterdir()
    }
    for dataset in (vector, masks):
        with Image.open(dataset.location / "reports" / "plots.png") as report:
            colors = set(report.convert("RGB").getdata())
        # The image-space coverage bar is drawn in the coverage panel only.
        assert (47, 111, 176) in colors
