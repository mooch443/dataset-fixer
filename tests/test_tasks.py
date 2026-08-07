from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml
from PIL import Image

from dataset_fixer import Dataset, DatasetValidationError, Task
from dataset_fixer import tiling as tiling_module
from dataset_fixer.models import Annotation, Sample
from dataset_fixer.visualization import _polygon_invalidity_details, save_coverage_annotated_original
from conftest import make_image, make_yolo_dataset


def _audit(dataset: Dataset, name: str):
    return dataset.manifest["audits"][name]


def test_segment_pose_and_polo_load(tmp_path: Path) -> None:
    segment = make_yolo_dataset(
        tmp_path / "segment",
        task="segment",
        names=["fruit"],
        train_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        val_rows=["0 0.25 0.25 0.75 0.25 0.75 0.75"],
    )
    pose = make_yolo_dataset(
        tmp_path / "pose",
        task="pose",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.5 0.5 0.4 0.4 2 0.6 0.6 1"],
        val_rows=["0 0.5 0.5 0.5 0.5 0.4 0.4 2 0.6 0.6 1"],
        extra={"kpt_shape": [2, 3], "flip_idx": [1, 0], "kpt_names": {0: ["left", "right"]}},
    )
    polo = make_yolo_dataset(
        tmp_path / "polo",
        task="polo",
        names=["fruit"],
        train_rows=["0 15 0.5 0.5"],
        val_rows=["0 15 0.4 0.4"],
        extra={"radii": {0: 15}},
    )
    assert Dataset.open(segment, task="segment", progress=False).task is Task.SEGMENT
    assert Dataset.open(pose, task="pose", progress=False).task is Task.POSE
    assert Dataset.open(polo, task="polo", progress=False).task is Task.POLO


def test_invalid_segmentation_can_be_skipped_virtually(tmp_path: Path) -> None:
    invalid_row = "0 0.2 0.2 0.8 0.8 0.8 0.2 0.2 0.8"
    valid_row = "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"
    source = make_yolo_dataset(
        tmp_path / "segment_with_invalid_polygon",
        task="segment",
        names=["fruit"],
        train_rows=[f"{invalid_row}\n{valid_row}"],
        val_rows=[valid_row],
    )
    source_label = next((source / "train" / "labels").rglob("*.txt"))
    original_label = source_label.read_text(encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="Invalid or self-intersecting polygon"):
        Dataset.open(source, task="segment", progress=False)

    dataset = Dataset.open(
        source,
        task="segment",
        errors="skip",
        progress=False,
    )

    assert sum(len(sample.annotations) for sample in dataset._samples) == 2
    assert any("Skipped invalid annotation" in warning for warning in dataset.warnings)
    assert any("Invalid or self-intersecting polygon" in warning for warning in dataset.warnings)
    assert dataset.validation_audit["skipped_count"] == 1
    assert dataset.validation_audit["visualized_count"] == 1
    assert Path(dataset.validation_audit["visualization"]).is_file()
    assert source_label.read_text(encoding="utf-8") == original_label

    exported = dataset.export(
        destination=tmp_path / "segment_without_invalid_polygon",
        visualize=False,
        progress=False,
    )
    exported_train_label = next((exported.location / "train" / "labels").rglob("*.txt"))
    assert len(exported_train_label.read_text(encoding="utf-8").splitlines()) == 1
    manifest = exported.manifest
    assert any("Skipped invalid annotation" in warning for warning in manifest["warnings"])
    assert manifest["validation"]["load_validation"]["skipped_count"] == 1
    assert "load_validation_audit" in manifest["audits"]
    assert (exported.location / "reports" / "plots.png").is_file()
    reopened = Dataset.open(exported.location, task="segment", progress=False)
    assert reopened.validation_audit["skipped_count"] == 1
    assert reopened._validation_audit_visualization == (
        exported.location / "reports" / "plots.png"
    )

    semantic = dataset.export(
        destination=tmp_path / "semantic_without_invalid_polygon",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )
    assert semantic.manifest["validation"]["load_validation"]["skipped_count"] == 1
    assert "load_validation_audit" in semantic.manifest["audits"]
    assert (semantic.location / "reports" / "plots.png").is_file()


def test_skip_audit_counts_all_failures_and_visualizes_at_most_four(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from matplotlib import pyplot as plt

    open_figures = set(plt.get_fignums())
    invalid_row = "0 0.2 0.2 0.8 0.8 0.8 0.2 0.2 0.8"
    valid_row = "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"
    source = make_yolo_dataset(
        tmp_path / "many_invalid_polygons",
        task="segment",
        names=["fruit"],
        train_rows=["\n".join([invalid_row] * 5 + [valid_row])],
        val_rows=[valid_row],
    )

    dataset = Dataset.open(
        source,
        task="segment",
        errors="skip",
        progress=False,
    )

    audit = dataset.validation_audit
    assert audit["status"] == "passed_with_skips"
    assert audit["skipped_count"] == 5
    assert audit["visualized_count"] == 4
    assert audit["max_visualized_examples"] == 4
    assert audit["counts_by_category"] == {"Skipped invalid annotation": 5}
    visualization = Path(audit["visualization"])
    assert visualization.is_file()
    with Image.open(visualization) as image:
        assert image.width > 0 and image.height > 0
    output = capsys.readouterr().out
    assert "Validation skip audit: 5 failed item(s)" in output
    assert "showing 4 example(s)" in output
    assert set(plt.get_fignums()) == open_figures


def test_notebook_display_closes_figure_after_immediate_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import IPython
    import IPython.display
    from matplotlib import pyplot as plt

    from dataset_fixer.visualization import _display_or_print

    figure = plt.figure()
    displayed: list[object] = []
    monkeypatch.setattr(IPython, "get_ipython", lambda: object())
    monkeypatch.setattr(IPython.display, "display", displayed.append)

    _display_or_print(figure, None)

    assert displayed == [figure]
    assert figure.number not in plt.get_fignums()


def test_invalid_polygon_visualization_identifies_exact_defects() -> None:
    reasons, markers = _polygon_invalidity_details(
        [(20.0, 20.0), (80.0, 80.0), (80.0, 20.0), (20.0, 80.0)],
        width=100,
        height=100,
    )
    assert "self-intersection" in reasons
    assert markers == [(50.0, 50.0, "self-intersection")]

    reasons, markers = _polygon_invalidity_details(
        [(-5.0, 20.0), (80.0, 20.0), (80.0, 80.0)],
        width=100,
        height=100,
    )
    assert "vertex 0 lies outside the image" in reasons
    assert markers[0] == (0.0, 20.0, "vertex 0 outside image\n(-5.0, 20.0)")


def test_grid_tiling_transforms_segment_and_pose(tmp_path: Path) -> None:
    segment = make_yolo_dataset(
        tmp_path / "segment_grid",
        task="segment",
        names=["fruit"],
        train_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        size=(160, 120),
    )
    pose = make_yolo_dataset(
        tmp_path / "pose_grid",
        task="pose",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.5 0.5 0.4 0.4 2 0.6 0.6 1"],
        val_rows=["0 0.5 0.5 0.5 0.5 0.4 0.4 2 0.6 0.6 1"],
        size=(160, 120),
        extra={"kpt_shape": [2, 3], "flip_idx": [1, 0], "kpt_names": {0: ["left", "right"]}},
    )
    segment_tiles = Dataset.open(segment, task="segment", progress=False).tile(
        tile_size=100,
        overlap=0.2,
        visualize=False,
        progress=False,
    ).export(destination=tmp_path / "segment_tiles", visualize=False, progress=False)
    pose_tiles = Dataset.open(pose, task="pose", progress=False).tile(
        tile_size=100,
        overlap=0.2,
        visualize=False,
        progress=False,
    ).export(destination=tmp_path / "pose_tiles", visualize=False, progress=False)
    assert any(annotation.polygon for sample in segment_tiles._samples for annotation in sample.annotations)
    assert any(annotation.keypoints for sample in pose_tiles._samples for annotation in sample.annotations)
    assert all(
        0 <= x <= sample.width and 0 <= y <= sample.height
        for sample in pose_tiles._samples
        for annotation in sample.annotations
        for x, y, _ in annotation.keypoints or []
    )


def test_tiling_geometry_errors_are_diagnostic_and_skippable(tmp_path: Path) -> None:
    # This valid source polygon intersects the [8, 8, 16, 16] grid window as
    # one Polygon plus one disconnected LineString: a Shapely GeometryCollection
    # that one YOLO segmentation row cannot represent.
    coordinates = [
        (11, 9), (6, 9), (6, 10), (6, 11), (6, 16), (6, 17),
        (7, 17), (16, 17), (16, 18), (18, 18), (18, 16), (17, 16),
        (16, 16), (7, 16), (7, 11), (11, 11),
    ]
    row = "0 " + " ".join(
        f"{value / 32:.8f}" for point in coordinates for value in point
    )
    source = make_yolo_dataset(
        tmp_path / "mixed_crop_geometry",
        task="segment",
        names=["fruit"],
        train_rows=[row],
        val_rows=[row],
        size=(32, 32),
    )
    dataset = Dataset.open(source, task="segment", progress=False)
    strict_destination = tmp_path / "strict_geometry_tiles"

    with pytest.raises(DatasetValidationError) as caught:
        dataset.tile(
            mode="grid",
            splits=["train"],
            tile_size=8,
            overlap=0,
            min_area_ratio=0,
            negative_tiles="all",
            errors="raise",
            visualize=False,
            progress=False,
        ).export(
            destination=strict_destination,
            visualize=False,
            progress=False,
        )

    message = str(caught.value)
    assert "unsupported mixed geometry" in message
    assert "GeometryCollection" in message
    assert "LineString" in message
    assert "annotation_index" in message
    assert "crop_xyxy" in message
    assert str(next((source / "train" / "images").rglob("*.jpg"))) in message
    assert not strict_destination.exists()

    skipped = dataset.tile(
        mode="grid",
        splits=["train"],
        tile_size=8,
        overlap=0,
        min_area_ratio=0,
        negative_tiles="all",
        errors="skip",
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "skipped_geometry_tiles",
        visualize=False,
        progress=False,
    )

    report = _audit(skipped, "tiling_skips")
    assert report["errors"] == "skip"
    assert report["skipped_candidates"] >= 1
    mixed = next(
        item
        for item in report["items"]
        if item["details"].get("result_geometry", {}).get("type")
        == "GeometryCollection"
    )
    assert "LineString" in mixed["details"]["result_geometry"]["component_types"]
    assert not list(
        (skipped.location / "train" / "images").rglob(
            "train_0__x8_y8_w8_h8.jpg"
        )
    )
    assert any("dataset-info.json" in warning for warning in skipped.warnings)


def test_polo_coverage_reports_and_visual_audit(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "polo_source",
        task="polo",
        names=["fruit"],
        train_rows=["0 4 0.4 0.4\n0 9 0.6 0.6"],
        val_rows=["0 7 0.5 0.5"],
        size=(300, 260),
        extra={"radii": {0: 15}},
    )
    dataset = Dataset.open(source, task="polo", progress=False)
    tiled = dataset.tile(
        mode="coverage",
        tile_size=100,
        large_image_threshold=200,
        scale_range=(1.0, 1.0),
        polo_radius_px=10,
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        max_tiles_per_source_image=5,
        background_ratio=0.5,
        seed=3,
        visualize=True,
        progress=False,
    ).export(destination=tmp_path / "coverage", visualize=False, progress=False)
    assert (tiled.location / "reports" / "plots.png").is_file()
    for name in (
        "coverage.label_coverage",
        "coverage.label_hit_summary",
        "coverage.class_coverage_summary",
        "coverage.tile_summary",
    ):
        assert name in tiled.manifest["audits"]
    rows = _audit(tiled, "coverage.label_coverage")
    assert rows and all(float(row["actual_coverages"]) >= 1 for row in rows)
    assert all(row["coverage_type"] == "sparse" for row in rows)
    assert all(annotation.radius == 10 for sample in tiled._samples for annotation in sample.annotations)


def test_coverage_resamples_tiles_that_cut_annotations(tmp_path: Path) -> None:
    rows = ["0 0.15 0.5 0.1 0.2\n0 0.65 0.5 0.1 0.2"]
    source = make_yolo_dataset(
        tmp_path / "boundary_source",
        task="detect",
        names=["fruit"],
        train_rows=rows,
        val_rows=rows,
        size=(200, 100),
    )
    dataset = Dataset.open(source, task="detect", progress=False)
    tiled = dataset.tile(
        mode="coverage",
        tile_size=100,
        large_image_threshold=50,
        scale_range=(1.0, 1.0),
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        max_attempts_per_target=100,
        background_ratio=0,
        seed=9,
        visualize=False,
        progress=False,
    ).export(destination=tmp_path / "boundary-safe", visualize=False, progress=False)

    sources = {str(sample.image_path): sample for sample in dataset._samples}
    assert len(tiled._samples) == 4
    for sample in tiled._samples:
        crop = sample.provenance["crop"]
        source_sample = sources[sample.provenance["original_image"]]
        left, top, right, bottom = crop
        for annotation in source_sample.annotations:
            x1, y1, x2, y2 = annotation.bbox
            intersects = min(x2, right) > max(x1, left) and min(y2, bottom) > max(y1, top)
            contained = left <= x1 and top <= y1 and x2 <= right and y2 <= bottom
            assert not intersects or contained


def test_coverage_writes_source_pixel_and_label_jpg_audits(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "source_pixel_coverage",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.25 0.5 0.1 0.2"],
        val_rows=["0 0.25 0.5 0.1 0.2"],
        size=(200, 100),
    )
    tiled = Dataset.open(source, task="detect", progress=False).tile(
        mode="coverage",
        tile_size=100,
        large_image_threshold=20,
        scale_range=(1.0, 1.0),
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        background_ratio=0,
        allow_lossy=True,
        seed=8,
        visualize=True,
        progress=False,
    ).export(
        destination=tmp_path / "source_pixel_coverage_tiles",
        visualize=True,
        progress=False,
    )

    assert (tiled.location / "reports" / "plots.png").is_file()
    coverage = _audit(tiled, "coverage.source_pixel_coverage")
    rows = coverage["rows"]
    assert {row["split"] for row in rows} == {"train", "val"}
    assert all(row["coverage_status"] == "exact" for row in rows)
    assert all(int(row["output_tiles"]) == 1 for row in rows)
    assert all(float(row["source_pixel_coverage_percent"]) == pytest.approx(50.0) for row in rows)
    aggregate = coverage["summary"]
    assert aggregate["splits"]["train"]["pixel_weighted_coverage_percent"] == pytest.approx(50.0)
    assert aggregate["splits"]["val"]["pixel_weighted_coverage_percent"] == pytest.approx(50.0)
    assert tiled._manifest["visuals"] == ["reports/plots.png"]


def test_lossless_coverage_fails_atomically_when_no_complete_crop_exists(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "uncroppable_source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.8 0.2"],
        val_rows=["0 0.5 0.5 0.8 0.2"],
        size=(200, 100),
    )
    destination = tmp_path / "uncroppable"

    with pytest.raises(DatasetValidationError, match="could not replace boundary-cut candidates"):
        Dataset.open(source, task="detect", progress=False).tile(
            mode="coverage",
            tile_size=100,
            large_image_threshold=50,
            scale_range=(1.0, 1.0),
            target_appearances_per_object=1,
            sparse_appearances_per_object=1,
            max_attempts_per_target=3,
            background_ratio=0,
            allow_lossy=False,
            visualize=False,
            progress=False,
        ).export(destination=destination, visualize=False, progress=False)

    assert not destination.exists()


def test_coverage_visual_keeps_legend_off_image_and_boxes_segmentations(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "blue.jpg"
    make_image(image_path, size=(120, 100), color=(20, 90, 150))
    polygon = [(30.0, 20.0), (90.0, 50.0), (45.0, 80.0)]
    sample = Sample(
        image_path=image_path,
        relative_path=Path("blue.jpg"),
        split="val",
        width=120,
        height=100,
        annotations=[
            Annotation(
                class_id=0,
                polygon=polygon,
                bbox=(30.0, 20.0, 90.0, 80.0),
            )
        ],
    )
    output = tmp_path / "coverage_visual.jpg"
    save_coverage_annotated_original(
        sample,
        {0: 3},
        {0: 5},
        {0: "dense"},
        [],
        output,
        {
            "target_appearances_per_object": 5,
            "sparse_appearances_per_object": 1,
            "max_tiles_per_source_image": 100,
            "min_nearby_objects_for_full_coverage": 5,
            "dense_neighbor_radius_px": 50.0,
            "background_ratio": 0.1,
            "polo_radius_px": 15.0,
            "jpeg_quality": 100,
        },
    )

    with Image.open(output) as preview:
        assert preview.size[0] >= 900
        scale = preview.size[0] / 120
        displayed_image_height = round(100 * scale)
        assert preview.size[1] > displayed_image_height
        pixels = preview.convert("RGB")
        # The triangle does not pass through its top-right bounding-box corner;
        # a magenta pixel there therefore comes from the new segmentation box.
        corner_x, corner_y = round(90 * scale), round(20 * scale)
        nearby = [
            pixels.getpixel((x, y))
            for x in range(corner_x - 8, corner_x + 9)
            for y in range(corner_y - 8, corner_y + 9)
        ]
        assert any(red > 170 and blue > 130 and green < 140 for red, green, blue in nearby)


def test_coverage_background_ratio_applies_to_complete_output(tmp_path: Path) -> None:
    positive = "0 0.5 0.5 0.2 0.2"
    rows = [positive, positive, positive, *([""] * 7)]
    source = make_yolo_dataset(
        tmp_path / "coverage_with_existing_backgrounds",
        task="detect",
        names=["fruit"],
        train_rows=rows,
        val_rows=rows,
        size=(160, 120),
    )

    tiled = (
        Dataset.open(source, task="detect", progress=False)
        .tile(
            mode="coverage",
            tile_size=100,
            large_image_threshold=500,
            background_ratio=0.25,
            visualize=False,
            progress=False,
        )
        .export(
            destination=tmp_path / "coverage_with_final_background_ratio",
            visualize=False,
            progress=False,
        )
    )

    for split in ("train", "val"):
        output = [sample for sample in tiled._samples if sample.split == split]
        assert len(output) == 4
        assert sum(not sample.annotations for sample in output) == 1
        assert sum(not sample.annotations for sample in output) / len(output) == 0.25

    summary = {
        row["split"]: row
        for row in _audit(tiled, "coverage.tile_summary")
    }
    for split in ("train", "val", "all"):
        assert float(summary[split]["background_fraction"]) == 0.25
    assert int(summary["train"]["candidate_background_source_images"]) == 7
    assert int(summary["train"]["dropped_background_source_images"]) == 6
    assert int(summary["train"]["target_background_images"]) == 1
    assert int(summary["train"]["actual_background_images"]) == 1
    class_counts = _audit(tiled, "class_counts")
    assert class_counts["operation"] == "tile-coverage"
    assert class_counts["result"]["background"] == 2
    assert class_counts["image_composition"]["result"] == {
        "annotated": 6,
        "background": 2,
        "total": 8,
        "background_fraction": 0.25,
    }
    assert class_counts["annotation_counts"]["result"]["0"] == 6


def test_coverage_balances_background_source_types(tmp_path: Path) -> None:
    positive = "0 0.5 0.5 0.1 0.1"
    source = make_yolo_dataset(
        tmp_path / "coverage_background_source_mix",
        task="detect",
        names=["fruit"],
        train_rows=[positive, positive, "", ""],
        val_rows=[positive, positive, "", ""],
        size=(300, 260),
    )

    tiled = (
        Dataset.open(source, task="detect", progress=False)
        .tile(
            mode="coverage",
            tile_size=100,
            large_image_threshold=200,
            scale_range=(1.0, 1.0),
            target_appearances_per_object=1,
            sparse_appearances_per_object=1,
            background_ratio=0.5,
            seed=13,
            visualize=False,
            progress=False,
        )
        .export(
            destination=tmp_path / "coverage_balanced_background_sources",
            visualize=False,
            progress=False,
        )
    )

    sampling = _audit(tiled, "coverage.background_sampling")
    for split in ("train", "val"):
        details = sampling["splits"][split]
        assert details["status"] == "target and equal source mix met"
        assert details["target_background_images"] == 2
        assert details["actual_background_images"] == 2
        assert details["actual_from_empty_source_images"] == 1
        assert details["actual_from_populated_image_space"] == 1
        assert details["actual_background_fraction"] == 0.5

    background_sources = Counter(
        record.get("background_source")
        for record in tiled.provenance.values()
        if record.get("output_annotation_count") == 0
    )
    assert background_sources == {
        "empty_source_image": 2,
        "populated_image_empty_space": 2,
    }


def test_coverage_background_filter_discards_black_crop_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "coverage_background_filter",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.1 0.1"],
        val_rows=["0 0.5 0.5 0.1 0.1"],
        size=(200, 100),
    )
    train_image = next((source / "train" / "images").rglob("*.jpg"))
    pixels = Image.new("RGB", (200, 100), (0, 0, 0))
    pixels.paste((80, 120, 160), (100, 0, 200, 100))
    pixels.save(train_image, format="PNG")

    crops = iter([(0, 0, 50, 50), (150, 0, 200, 50)])
    monkeypatch.setattr(
        tiling_module,
        "_random_crop",
        lambda width, height, cfg, rng: next(crops),
    )

    def keep_non_black(candidate: Image.Image) -> bool:
        return candidate.getbbox() is not None

    tiled = Dataset.open(source, task="detect", progress=False).tile(
        mode="coverage",
        splits=["train"],
        tile_size=50,
        large_image_threshold=20,
        scale_range=(1.0, 1.0),
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        background_ratio=0.5,
        max_background_attempts_per_tile=3,
        background_filter=keep_non_black,
        seed=5,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "coverage_filtered_backgrounds",
        visualize=False,
        progress=False,
    )

    backgrounds = [sample for sample in tiled._samples if not sample.annotations]
    assert len(backgrounds) == 1
    with Image.open(backgrounds[0].image_path) as candidate:
        assert candidate.convert("RGB").getbbox() is not None
    assert backgrounds[0].provenance["background_filter_result"] == "accepted"
    report = _audit(tiled, "background_filter")
    assert report["evaluated_candidates"] == 2
    assert report["accepted_candidates"] == 1
    assert report["rejected_candidates"] == 1
    assert report["accepted_percentage"] == 50.0
    assert report["rejected_percentage"] == 50.0
    assert report["by_origin"]["coverage-populated_image_empty_space"] == {
        "accepted": 1,
        "accepted_percentage": 50.0,
        "evaluated": 2,
        "rejected": 1,
        "rejected_percentage": 50.0,
    }


def test_coverage_background_filter_applies_to_copied_empty_sources(
    tmp_path: Path,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "copied_background_filter",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.1 0.1", "", "", ""],
        val_rows=["0 0.5 0.5 0.1 0.1"],
        size=(100, 100),
    )
    empty_images = sorted((source / "train" / "images").rglob("*.jpg"))[1:]
    for image_path in empty_images[:2]:
        Image.new("RGB", (100, 100), (0, 0, 0)).save(image_path, format="PNG")

    tiled = Dataset.open(source, task="detect", progress=False).tile(
        mode="coverage",
        splits=["train"],
        tile_size=100,
        large_image_threshold=200,
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        background_ratio=0.5,
        background_filter=lambda candidate: candidate.getbbox() is not None,
        seed=4,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "copied_filtered_backgrounds",
        visualize=False,
        progress=False,
    )

    background = next(sample for sample in tiled._samples if not sample.annotations)
    with Image.open(background.image_path) as candidate:
        assert candidate.convert("RGB").getbbox() is not None
    assert background.provenance["tile_mode"] == "coverage-background-copy"
    report = _audit(tiled, "background_filter")
    assert report["by_origin"]["coverage-background-copy"]["accepted"] == 1


def test_grid_background_filter_discards_black_windows(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "grid_background_filter",
        task="detect",
        names=["fruit"],
        train_rows=["", "0 0.5 0.5 0.1 0.1"],
        val_rows=["0 0.5 0.5 0.1 0.1"],
        size=(200, 100),
    )
    train_image = next((source / "train" / "images").rglob("*.jpg"))
    pixels = Image.new("RGB", (200, 100), (0, 0, 0))
    pixels.paste((80, 120, 160), (100, 0, 200, 100))
    pixels.save(train_image, format="PNG")

    tiled = Dataset.open(source, task="detect", progress=False).tile(
        mode="grid",
        splits=["train"],
        tile_size=100,
        overlap=0,
        negative_tiles="all",
        background_filter=lambda candidate: candidate.getbbox() is not None,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "grid_filtered_backgrounds",
        visualize=False,
        progress=False,
    )

    backgrounds = [sample for sample in tiled._samples if not sample.annotations]
    assert len(backgrounds) == 1
    assert backgrounds[0].relative_path.name.endswith("__x100_y0_w100_h100.jpg")
    report = _audit(tiled, "background_filter")
    assert report["evaluated_candidates"] == 2
    assert report["accepted_candidates"] == 1
    assert report["rejected_candidates"] == 1


def test_background_filter_exception_is_diagnostic_and_atomic(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "broken_background_filter",
        task="detect",
        names=["fruit"],
        train_rows=["", "0 0.5 0.5 0.1 0.1"],
        val_rows=["0 0.5 0.5 0.1 0.1"],
        size=(100, 100),
    )
    destination = tmp_path / "broken_background_filter_output"

    def broken_filter(candidate: Image.Image) -> bool:
        raise RuntimeError(f"cannot inspect {candidate.size}")

    with pytest.raises(DatasetValidationError) as caught:
        Dataset.open(source, task="detect", progress=False).tile(
            mode="grid",
            splits=["train"],
            tile_size=100,
            negative_tiles="all",
            background_filter=broken_filter,
            visualize=False,
            progress=False,
        ).export(
            destination=destination,
            visualize=False,
            progress=False,
        )

    message = str(caught.value)
    assert "Background filter raised an exception" in message
    assert "RuntimeError: cannot inspect (100, 100)" in message
    assert "crop_xyxy" in message
    assert not destination.exists()


def test_grid_negative_fraction_applies_to_complete_output(tmp_path: Path) -> None:
    positive = "0 0.5 0.5 0.2 0.2"
    rows = [positive, positive, positive, *([""] * 7)]
    source = make_yolo_dataset(
        tmp_path / "grid_with_existing_backgrounds",
        task="detect",
        names=["fruit"],
        train_rows=rows,
        val_rows=rows,
        size=(160, 120),
    )

    tiled = (
        Dataset.open(source, task="detect", progress=False)
        .tile(
            mode="grid",
            tile_size=200,
            negative_tiles=0.25,
            visualize=False,
            progress=False,
        )
        .export(
            destination=tmp_path / "grid_with_final_background_ratio",
            visualize=False,
            progress=False,
        )
    )

    for split in ("train", "val"):
        output = [sample for sample in tiled._samples if sample.split == split]
        assert len(output) == 4
        assert sum(not sample.annotations for sample in output) == 1
        assert sum(not sample.annotations for sample in output) / len(output) == 0.25
    class_counts = _audit(tiled, "class_counts")
    assert class_counts["operation"] == "tile-grid"
    assert class_counts["result"]["background"] == 2


@pytest.mark.parametrize("task", ["detect", "segment", "pose", "polo"])
def test_coverage_tiling_supports_every_task(task: str, tmp_path: Path) -> None:
    rows = {
        "detect": "0 0.5 0.5 0.2 0.2",
        "segment": "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6",
        "pose": "0 0.5 0.5 0.2 0.2 0.47 0.47 2 0.53 0.53 2",
        "polo": "0 10 0.5 0.5",
    }
    extra = {
        "pose": {"kpt_shape": [2, 3], "flip_idx": [1, 0], "kpt_names": {0: ["left", "right"]}},
        "polo": {"radii": {0: 10}},
    }.get(task, {})
    source = make_yolo_dataset(
        tmp_path / f"{task}_coverage_source",
        task=task,
        names=["fruit"],
        train_rows=[rows[task]],
        val_rows=[rows[task]],
        size=(300, 260),
        extra=extra,
    )

    tiled = (
        Dataset.open(source, task=task, progress=False)
        .tile(
            mode="coverage",
            tile_size=100,
            large_image_threshold=200,
            scale_range=(1.0, 1.0),
            target_appearances_per_object=2,
            sparse_appearances_per_object=2,
            background_ratio=0,
            max_tiles_per_source_image=10,
            seed=7,
            visualize=False,
            progress=False,
        )
        .export(
            destination=tmp_path / f"{task}_coverage",
            visualize=False,
            progress=False,
        )
    )

    rows_out = _audit(tiled, "coverage.label_coverage")
    assert rows_out
    assert all(int(row["requested_coverages"]) == 2 for row in rows_out)
    assert all(int(row["actual_coverages"]) == 2 for row in rows_out)
    assert all(sample.width == 100 and sample.height == 100 for sample in tiled._samples)
    assert tiled.task.value == task


def test_coverage_object_appearance_override_uses_source_id(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "coverage_override_source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.2 0.2"],
        val_rows=["0 0.5 0.5 0.2 0.2"],
        size=(300, 260),
    )
    dataset = Dataset.open(source, task="detect", progress=False)
    source_id = dataset._samples[0].annotations[0].source_id

    tiled = dataset.tile(
        mode="coverage",
        tile_size=100,
        large_image_threshold=200,
        scale_range=(1.0, 1.0),
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        object_appearance_overrides={str(source_id): 3},
        background_ratio=0,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "coverage_override",
        visualize=False,
        progress=False,
    )

    rows = _audit(tiled, "coverage.label_coverage")
    overridden = next(row for row in rows if row["source_id"] == str(source_id))
    assert int(overridden["requested_coverages"]) == 3
    assert int(overridden["actual_coverages"]) == 3
    with pytest.raises(ValueError, match="Unknown object_appearance_overrides"):
        dataset.tile(mode="coverage", object_appearance_overrides={"missing": 2})


def test_coco_detection_export_compacts_categories(tmp_path: Path) -> None:
    root = tmp_path / "coco"
    make_image(root / "images" / "train" / "one.jpg", size=(100, 80))
    make_image(root / "images" / "val" / "two.jpg", size=(100, 80))
    categories = [{"id": 4, "name": "apple"}, {"id": 9, "name": "pear"}]
    train = {
        "info": {"name": "coco-fruit"},
        "images": [{"id": 1, "file_name": "one.jpg", "width": 100, "height": 80}],
        "categories": categories,
        "annotations": [{"id": 11, "image_id": 1, "category_id": 9, "bbox": [10, 10, 20, 30], "area": 600}],
    }
    val = {
        "images": [{"id": 2, "file_name": "two.jpg", "width": 100, "height": 80}],
        "categories": categories,
        "annotations": [{"id": 12, "image_id": 2, "category_id": 4, "bbox": [20, 10, 30, 30], "area": 900}],
    }
    (root / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (root / "val.json").write_text(json.dumps(val), encoding="utf-8")
    dataset = Dataset.open(root, task="detect", progress=False)
    assert dataset.classes == {0: "apple", 1: "pear"}
    exported = dataset.export(destination=tmp_path / "coco_yolo", visualize=False, progress=False)
    assert exported.classes == {0: "apple", 1: "pear"}
    data = yaml.safe_load(exported.data_yaml.read_text(encoding="utf-8"))
    assert data["train"] == "train/images" and data["val"] == "val/images"
    labels = "\n".join(path.read_text(encoding="utf-8") for path in exported.location.rglob("*.txt") if "labels" in path.parts)
    assert labels.startswith(("0 ", "1 "))


def test_coco_nested_images_path_round_trips_without_annotation_loss(tmp_path: Path) -> None:
    root = tmp_path / "nested_coco"
    image_path = root / "train" / "images" / "one.jpg"
    make_image(image_path, size=(100, 80))
    source = {
        "info": {"name": "nested-coco"},
        "images": [
            {
                "id": 1,
                "file_name": "train/images/one.jpg",
                "width": 100,
                "height": 80,
            }
        ],
        "categories": [{"id": 4, "name": "school"}],
        "annotations": [
            {
                "id": 11,
                "image_id": 1,
                "category_id": 4,
                "bbox": [10, 15, 20, 25],
                "area": 500,
            }
        ],
    }
    (root / "annotations.json").write_text(json.dumps(source), encoding="utf-8")
    dataset = Dataset.open(root, task="detect", progress=False)
    original = dataset._samples[0]

    assert original.relative_path == Path("one.jpg")
    assert len(original.annotations) == 1

    exported = dataset.export(
        destination=tmp_path / "nested_yolo",
        visualize=False,
        progress=False,
    )

    expected_image = exported.location / "train" / "images" / original.relative_path
    expected_label = exported.location / "train" / "labels" / original.relative_path.with_suffix(".txt")
    assert expected_image.is_file()
    assert expected_label.is_file()
    assert len(exported._samples) == 1
    assert len(exported._samples[0].annotations) == 1

    reopened = Dataset.open(exported.location, progress=False)

    assert reopened.task is Task.DETECT
    assert reopened._samples[0].relative_path == original.relative_path
    assert len(reopened._samples) == len(dataset._samples)
    assert len(reopened._samples[0].annotations) == len(original.annotations)
    assert reopened._samples[0].annotations[0].class_id == original.annotations[0].class_id
    assert reopened._samples[0].annotations[0].bbox == pytest.approx(original.annotations[0].bbox)


def test_coco_canonical_path_collisions_fail_instead_of_overwriting(tmp_path: Path) -> None:
    root = tmp_path / "colliding_coco"
    first = root / "train" / "images" / "one.jpg"
    second = root / "images" / "train" / "one.jpg"
    make_image(first, size=(100, 80))
    make_image(second, size=(100, 80), color=(10, 20, 30))
    source = {
        "images": [
            {"id": 1, "file_name": "train/images/one.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "images/train/one.jpg", "width": 100, "height": 80},
        ],
        "categories": [{"id": 4, "name": "school"}],
        "annotations": [
            {"id": 11, "image_id": 1, "category_id": 4, "bbox": [10, 15, 20, 25]},
            {"id": 12, "image_id": 2, "category_id": 4, "bbox": [20, 15, 20, 25]},
        ],
    }
    (root / "annotations.json").write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="same canonical output path"):
        Dataset.open(root, task="detect", errors="skip", progress=False)


def test_yaml_less_split_first_yolo_does_not_duplicate_layout(tmp_path: Path) -> None:
    root = tmp_path / "flat_split_yolo"
    image_path = root / "train" / "images" / "nested" / "one.jpg"
    label_path = root / "train" / "labels" / "nested" / "one.txt"
    make_image(image_path, size=(100, 80))
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.2 0.25\n", encoding="utf-8")

    dataset = Dataset.open(root, task="detect", progress=False)

    assert dataset.splits == ("train",)
    assert dataset._samples[0].relative_path == Path("nested/one.jpg")
    assert len(dataset._samples[0].annotations) == 1

    exported = dataset.export(
        destination=tmp_path / "canonical_yolo",
        visualize=False,
        progress=False,
    )

    assert (exported.location / "train" / "images" / "nested" / "one.jpg").is_file()
    assert (exported.location / "train" / "labels" / "nested" / "one.txt").is_file()
    assert not (exported.location / "train" / "images" / "train" / "images").exists()

    reopened = Dataset.open(exported.location, progress=False)
    assert reopened._samples[0].relative_path == Path("nested/one.jpg")
    assert len(reopened._samples[0].annotations) == 1


def test_coco_errors_skip_filters_bad_records_across_loading_stages(tmp_path: Path) -> None:
    root = tmp_path / "coco_with_bad_records"
    make_image(root / "images" / "one.jpg", size=(100, 80))
    data = {
        "images": [
            {"id": 1, "file_name": "one.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "missing.jpg", "width": 100, "height": 80},
        ],
        "categories": [{"id": 4, "name": "apple"}],
        "annotations": [
            {"id": 11, "image_id": 1, "category_id": 4, "bbox": [10, 10, 20, 30]},
            {"id": 12, "image_id": 1, "category_id": 4, "bbox": ["bad", 10, 20, 30]},
            {"id": 13, "image_id": 99, "category_id": 4, "bbox": [10, 10, 20, 30]},
            {"id": 11, "image_id": 1, "category_id": 4, "bbox": [10, 10, 20, 30]},
        ],
    }
    (root / "annotations.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "annotations.json").write_text(json.dumps(data), encoding="utf-8")

    dataset = Dataset.open(root, task="detect", errors="skip", progress=False)

    assert len(dataset._samples) == 1
    assert len(dataset._samples[0].annotations) == 1
    assert any("unknown image" in warning for warning in dataset.warnings)
    assert any("IDs must be unique" in warning for warning in dataset.warnings)
    assert any("Malformed COCO annotation" in warning for warning in dataset.warnings)
    assert any("COCO image file not found" in warning for warning in dataset.warnings)
    assert dataset.validation_audit["skipped_count"] == 4
    assert dataset.validation_audit["visualized_count"] == 4
    assert Path(dataset.validation_audit["visualization"]).is_file()
