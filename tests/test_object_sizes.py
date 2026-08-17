from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset_fixer.comparison.object_sizes import (
    ObjectComponent,
    ObjectSizeModelResult,
    binary_mask_components,
    component_dice,
    evaluate_object_size_model,
    match_components,
    object_size_report_artifacts_exist,
    polygon_components,
    prepare_object_size_reference,
    render_grouped_metric_breakdown,
    render_grouped_presence_metric_breakdown,
    render_large_object_examples,
    render_object_size_breakdown,
    render_segmentation_metric_breakdown,
    select_large_examples,
)
import dataset_fixer.comparison.object_sizes as object_sizes_module


def _component(
    component_id: str,
    mask: np.ndarray,
    *,
    bbox: tuple[int, int, int, int] | None = None,
    class_id: int = 0,
    image_id: str = "image",
    image_path: Path | None = None,
) -> ObjectComponent:
    mask = np.asarray(mask, dtype=bool)
    if bbox is None:
        bbox = (0, 0, mask.shape[1], mask.shape[0])
    return ObjectComponent(
        component_id=component_id,
        image_id=image_id,
        relative_path=f"{image_id}.png",
        image_path=image_path or Path(f"/{image_id}.png"),
        class_id=class_id,
        bbox=bbox,
        mask=mask,
        area=int(np.sum(mask)),
    )


def _capture_chart(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}
    original = object_sizes_module.save_chart

    def save(chart: object, path: Path, **kwargs: object) -> Path:
        captured["spec"] = chart.to_dict()  # type: ignore[attr-defined]
        return original(chart, path, **kwargs)

    monkeypatch.setattr(object_sizes_module, "save_chart", save)
    return captured


def _inline_rows(
    value: object,
    datasets: dict[str, list[dict[str, object]]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if isinstance(value, dict):
        if datasets is None:
            candidate = value.get("datasets")
            datasets = candidate if isinstance(candidate, dict) else {}
        data = value.get("data")
        if isinstance(data, dict) and isinstance(data.get("values"), list):
            rows.extend(data["values"])
        elif isinstance(data, dict) and isinstance(data.get("name"), str):
            rows.extend(datasets.get(data["name"], []))
        for child in value.values():
            rows.extend(_inline_rows(child, datasets))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_inline_rows(child, datasets))
    return rows


def test_semantic_components_use_eight_connectivity_and_foreground_area() -> None:
    components = binary_mask_components(
        np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
        image_id="image",
        relative_path="image.png",
        image_path=Path("/image.png"),
        prefix="reference",
    )

    assert len(components) == 1
    assert components[0].area == 2
    assert components[0].bbox == (0, 0, 2, 2)


def test_instance_polygon_components_measure_native_foreground_pixels(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "black").save(image_path)

    components = polygon_components(
        8,
        8,
        [
            {
                "class_id": 3,
                "polygon": [(1, 1), (3, 1), (3, 3), (1, 3)],
            }
        ],
        image_id="image",
        relative_path="image.png",
        image_path=image_path,
        prefix="reference",
        strict=True,
    )

    assert len(components) == 1
    assert components[0].class_id == 3
    assert components[0].area == 9
    assert components[0].bbox == (1, 1, 4, 4)


def test_percentile_groups_are_exclusive_and_ties_use_small_precedence() -> None:
    components = {
        "image": tuple(
            _component(f"object-{index}", np.ones((2, 2), dtype=bool))
            for index in range(3)
        )
    }

    reference = prepare_object_size_reference(components)

    assert reference.p10_area == pytest.approx(4)
    assert reference.p90_area == pytest.approx(4)
    assert [reference.group(component.area) for component in reference.all_components] == [
        "small",
        "small",
        "small",
    ]
    assert reference.metadata()["reference_support"] == {
        "small": 3,
        "medium": 0,
        "large": 0,
    }


def test_empty_reference_cohort_skips_object_size_analysis() -> None:
    reference = prepare_object_size_reference({"image": ()})

    assert reference.status == "skipped"
    assert reference.metadata()["reference_support"] == {
        "small": 0,
        "medium": 0,
        "large": 0,
    }


def test_legacy_tiled_skip_report_is_forced_to_regenerate(tmp_path: Path) -> None:
    manifest = {
        "kind": "semantic-mask-model-comparison",
        "dataset": {"task": "segment"},
        "object_size_analysis": {
            "status": "skipped",
            "reason": "object-size analysis is unavailable for tiled evaluation datasets",
        },
        "reports": {
            "metric_breakdown": "reports/metric-breakdown.png",
            "object_size_breakdown": None,
            "large_object_examples": [],
        },
    }

    assert not object_size_report_artifacts_exist(tmp_path, manifest)


def test_object_size_scoring_covers_matches_misses_false_positives_and_classes() -> None:
    small = _component("small", np.ones((1, 1), dtype=bool), class_id=0)
    medium = _component("medium", np.ones((2, 2), dtype=bool), class_id=0)
    large = _component("large", np.ones((3, 3), dtype=bool), class_id=1)
    reference = prepare_object_size_reference({"image": (small, medium, large)})
    predictions = {
        "image": (
            _component("small-pred", np.ones((1, 1), dtype=bool), class_id=0),
            _component("medium-wrong-class", np.ones((2, 2), dtype=bool), class_id=1),
            _component("large-pred", np.ones((3, 3), dtype=bool), class_id=1),
        )
    }

    result = evaluate_object_size_model(reference, predictions).summary

    assert result["small_object_dice"] == pytest.approx(1.0)
    assert result["medium_object_dice"] == pytest.approx(0.0)
    assert result["large_object_dice"] == pytest.approx(1.0)
    assert result["medium_object_reference_count"] == 1
    assert result["medium_object_prediction_count"] == 1
    assert result["medium_object_match_count"] == 0
    assert result["medium_object_scoring_count"] == 2


def test_hungarian_matching_scores_splits_and_merges_once() -> None:
    truth = _component("truth", np.ones((4, 4), dtype=bool))
    top = _component(
        "top",
        np.ones((2, 4), dtype=bool),
        bbox=(0, 0, 4, 2),
    )
    bottom = _component(
        "bottom",
        np.ones((2, 4), dtype=bool),
        bbox=(0, 2, 4, 4),
    )

    matches, unmatched_truth, unmatched_prediction = match_components(
        (truth,), (top, bottom)
    )

    assert len(matches) == 1
    assert matches[0][2] == pytest.approx(2 / 3)
    assert unmatched_truth == ()
    assert len(unmatched_prediction) == 1

    left = _component("left", np.ones((2, 2), dtype=bool), bbox=(0, 0, 2, 2))
    right = _component("right", np.ones((2, 2), dtype=bool), bbox=(3, 0, 5, 2))
    merged_mask = np.zeros((2, 5), dtype=bool)
    merged_mask[:, :2] = True
    merged_mask[:, 3:] = True
    merged = _component("merged", merged_mask, bbox=(0, 0, 5, 2))
    matches, unmatched_truth, unmatched_prediction = match_components(
        (left, right), (merged,)
    )

    assert len(matches) == 1
    assert matches[0][2] == pytest.approx(2 / 3)
    assert len(unmatched_truth) == 1
    assert unmatched_prediction == ()
    assert component_dice(left, _component("wrong", left.mask, class_id=1)) == 0


def test_unsupported_reference_group_reports_nan_even_with_false_positive() -> None:
    truth = _component("truth", np.ones((2, 2), dtype=bool))
    reference = prepare_object_size_reference({"image": (truth,)})
    false_positive = _component("false-positive", np.ones((4, 4), dtype=bool))

    summary = evaluate_object_size_model(
        reference, {"image": (truth, false_positive)}
    ).summary

    assert summary["small_object_dice"] == pytest.approx(1.0)
    assert math.isnan(summary["large_object_dice"])
    assert summary["large_object_prediction_count"] == 1
    assert summary["large_object_scoring_count"] == 1


def test_large_example_selection_is_deterministic_and_has_no_duplicates() -> None:
    components = []
    for index, area in enumerate([1, 2, 3, 4, 5, 6, 10, 10, 10, 10]):
        components.append(
            _component(
                f"object-{index}",
                np.ones((1, area), dtype=bool),
            )
        )
    reference = prepare_object_size_reference({"image": tuple(components)})
    scores = {
        component.component_id: (0.0 if component.component_id in {"object-8", "object-9"} else 1.0)
        for component in components
    }
    result = ObjectSizeModelResult(
        summary={},
        reference_scores=scores,
        matched_prediction_ids={},
    )

    selected = select_large_examples(reference, {"model": result})

    assert len(selected) == 4
    assert len({row["component"].component_id for row in selected}) == 4
    assert [row["selection_reason"] for row in selected[:2]] == [
        "largest-reference-area",
        "largest-reference-area",
    ]
    assert [row["component"].component_id for row in selected[2:]] == [
        "object-8",
        "object-9",
    ]


def test_four_selected_large_object_crop_reports_are_rendered(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "gray").save(image_path)
    components = tuple(
        _component(
            f"object-{index}",
            np.ones((1, area), dtype=bool),
            image_path=image_path,
        )
        for index, area in enumerate([1, 2, 3, 4, 5, 6, 10, 10, 10, 10])
    )
    reference = prepare_object_size_reference({"image": components})
    result = ObjectSizeModelResult(
        summary={},
        reference_scores={component.component_id: 1.0 for component in components},
        matched_prediction_ids={},
    )
    reports = tmp_path / "reports"

    rendered = render_large_object_examples(
        reports,
        select_large_examples(reference, {"model": result}),
        {"model": {"image": components}},
        {"model": result},
        {"model": "sahi"},
    )

    assert len(rendered) == 4
    assert all((tmp_path / row["path"]).is_file() for row in rendered)


@pytest.mark.parametrize(
    ("model_type", "expected_slug"),
    [("yolo26m-seg", "yolo26m-seg"), ("nnunet-m", "nnunet-m")],
)
def test_metric_breakdown_includes_model_type_and_presence_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    expected_slug: str,
) -> None:
    ranking = [
        {
            "model": "model",
            "model_type": model_type,
            "upscale_factor": 4,
            "effective_prediction_resolution": "512px",
            "dice": 0.1,
            "micro_dice": 0.2,
            "foreground_precision": 0.3,
            "foreground_recall": 0.4,
            "raw_presence_precision": 0.45,
            "component_filtered_presence_precision": 0.55,
            "raw_positive_image_recall": 0.5,
            "component_filtered_positive_image_recall": 0.4,
            "raw_empty_image_specificity": 0.6,
            "component_filtered_empty_image_specificity": 0.8,
            "micro_iou": 0.7,
            "positive_case_dice": 0.8,
        }
    ]
    captured = _capture_chart(monkeypatch)

    path = render_segmentation_metric_breakdown(
        tmp_path,
        ranking,
        minimum_component_area=12,
    )
    rows = _inline_rows(captured["spec"])
    column_labels = list(dict.fromkeys(str(row["column"]) for row in rows if "column" in row))
    identity_labels = [
        str(row["label"])
        for row in rows
        if "row" in row and "label" in row and "model" not in row
    ]
    badges = {str(row["text"]): str(row["color"]) for row in rows if "color" in row and "text" in row}

    assert path.is_file()
    assert identity_labels == ["model"]
    assert expected_slug in badges
    assert badges["4×"] == "#B45309"
    assert badges["512px"] == "#475569"
    assert column_labels == [
        "Mean Dice",
        "Pooled foreground\nDice",
        "Pixel\nprecision",
        "Pixel\nrecall",
        "Presence precision\nraw",
        "Presence precision\narea-filtered",
        "Positive-image recall\nraw",
        "Positive-image recall\narea-filtered",
        "Empty specificity\nraw",
        "Empty specificity\narea-filtered",
    ]


def test_object_size_breakdown_includes_colored_model_type_badges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = {
        "image": (
            _component("small", np.ones((1, 1), dtype=bool)),
            _component("medium", np.ones((2, 2), dtype=bool)),
            _component("large", np.ones((3, 3), dtype=bool)),
        )
    }
    analysis = prepare_object_size_reference(components)
    ranking = [
        {
            "model": "semantic-model",
            "model_type": "yolo26m-sem",
            "small_object_dice": 1.0,
            "medium_object_dice": 0.5,
            "large_object_dice": 0.0,
        },
        {
            "model": "instance-model",
            "model_type": "yolo26m-seg",
            "small_object_dice": 1.0,
            "medium_object_dice": 0.5,
            "large_object_dice": 0.0,
        },
    ]
    captured = _capture_chart(monkeypatch)

    path = render_object_size_breakdown(tmp_path, ranking, analysis)
    rows = _inline_rows(captured["spec"])
    labels = [
        str(row["label"])
        for row in rows
        if "row" in row and "label" in row and "model" not in row
    ]
    badges = {str(row["text"]): str(row["color"]) for row in rows if "color" in row and "text" in row}

    assert path is not None and path.is_file()
    assert labels == [
        "semantic-model",
        "instance-model",
    ]
    assert badges["yolo26m-sem"] == "#0F766E"
    assert badges["yolo26m-seg"] == "#2563EB"
    specification = str(captured["spec"])
    assert "Object Dice by foreground area (small 1, medium 1, large 1)" in specification
    assert "reference n=" not in specification


def test_grouped_metric_breakdown_sorts_by_macro_and_shows_defined_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranking = [
        {"model": "lower", "model_type": "yolo26m-seg"},
        {"model": "higher", "model_type": "nnunet-m"},
    ]
    grouped_by_model = {
        "lower": {
            "group_macro_dice": 0.25,
            "group_defined_dice_count": 1,
            "group_count": 2,
            "per_group": [
                {"group": "aoi-a", "dice": 0.25},
                {"group": "aoi-b", "dice": math.nan},
            ],
        },
        "higher": {
            "group_macro_dice": 0.75,
            "group_defined_dice_count": 2,
            "group_count": 2,
            "per_group": [
                {"group": "aoi-a", "dice": 0.5},
                {"group": "aoi-b", "dice": 1.0},
            ],
        },
    }
    captured = _capture_chart(monkeypatch)

    path = render_grouped_metric_breakdown(
        tmp_path,
        ranking,
        grouped_by_model,
        group_splits={"aoi-a": ("train",), "aoi-b": ("val",)},
    )
    rows = _inline_rows(captured["spec"])
    row_labels = [
        str(row["label"])
        for row in rows
        if "row" in row and "label" in row and "model" not in row
    ]
    column_labels = list(dict.fromkeys(str(row["column"]) for row in rows if "column" in row))
    cell_labels = [
        str(row["label"]) for row in rows if "label" in row and "model" in row
    ]

    assert path.is_file()
    assert row_labels == ["higher", "lower"]
    assert column_labels == ["Macro", "aoi-a\n[train]", "aoi-b\n[val]"]
    assert cell_labels[0] == "0.750\n(2/2)"
    assert cell_labels[3] == "0.250\n(1/2)"
    assert cell_labels[5] == "TN"
    assert next(row["display_value"] for row in rows if row.get("model") == "lower" and row.get("column") == "aoi-b\n[val]") == 1.0


@pytest.mark.parametrize("metric", ["precision", "recall", "f1"])
def test_grouped_presence_breakdown_sorts_by_macro_f1_and_marks_edge_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
) -> None:
    ranking = [
        {"model": "lower", "model_type": "yolo26m-seg"},
        {"model": "higher", "model_type": "nnunet-m"},
    ]
    grouped_by_model = {
        "lower": {
            "group_count": 3,
            "group_macro_presence_precision": 0.5,
            "group_macro_presence_recall": 0.25,
            "group_macro_presence_f1": 0.3,
            "group_defined_presence_precision_count": 1,
            "group_defined_presence_recall_count": 1,
            "group_defined_presence_f1_count": 2,
            "per_group": [
                {
                    "group": "aoi-fp",
                    "positive_cases": 0,
                    "presence_tp": 0,
                    "presence_fp": 1,
                    "presence_precision": 0.0,
                    "presence_recall": math.nan,
                    "presence_f1": 0.0,
                },
                {
                    "group": "aoi-miss",
                    "positive_cases": 1,
                    "presence_tp": 0,
                    "presence_fp": 0,
                    "presence_precision": math.nan,
                    "presence_recall": 0.0,
                    "presence_f1": 0.0,
                },
                {
                    "group": "aoi-tn",
                    "positive_cases": 0,
                    "presence_tp": 0,
                    "presence_fp": 0,
                    "presence_precision": math.nan,
                    "presence_recall": math.nan,
                    "presence_f1": math.nan,
                },
            ],
        },
        "higher": {
            "group_count": 3,
            "group_macro_presence_precision": 1.0,
            "group_macro_presence_recall": 1.0,
            "group_macro_presence_f1": 1.0,
            "group_defined_presence_precision_count": 1,
            "group_defined_presence_recall_count": 1,
            "group_defined_presence_f1_count": 1,
            "per_group": [
                {
                    "group": group,
                    "positive_cases": int(group == "aoi-miss"),
                    "presence_tp": int(group == "aoi-miss"),
                    "presence_fp": 0,
                    "presence_precision": 1.0 if group == "aoi-miss" else math.nan,
                    "presence_recall": 1.0 if group == "aoi-miss" else math.nan,
                    "presence_f1": 1.0 if group == "aoi-miss" else math.nan,
                }
                for group in ["aoi-fp", "aoi-miss", "aoi-tn"]
            ],
        },
    }
    captured = _capture_chart(monkeypatch)

    path = render_grouped_presence_metric_breakdown(
        tmp_path,
        ranking,
        grouped_by_model,
        metric=metric,
        group_splits={
            "aoi-fp": ("train",),
            "aoi-miss": ("val",),
            "aoi-tn": ("train", "val"),
        },
    )
    rows = _inline_rows(captured["spec"])
    row_labels = [
        str(row["label"])
        for row in rows
        if "row" in row and "label" in row and "model" not in row
    ]
    column_labels = list(dict.fromkeys(str(row["column"]) for row in rows if "column" in row))
    cell_labels = [
        str(row["label"]) for row in rows if "label" in row and "model" in row
    ]

    assert path.name == f"grouped-presence-{metric}.png"
    assert path.is_file()
    assert row_labels == ["higher", "lower"]
    assert column_labels == [
        f"Macro {metric.upper()}",
        "aoi-fp\n[train]",
        "aoi-miss\n[val]",
        "aoi-tn\n[train/val]",
    ]
    assert cell_labels[0].startswith("1.000")
    assert "TN" in cell_labels
    if metric == "precision":
        assert "MISS" in cell_labels
    if metric == "recall":
        assert "FP" in cell_labels
