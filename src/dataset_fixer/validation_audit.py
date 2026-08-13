from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import Annotation, DatasetMetadata, Sample, Task
from .utils import to_jsonable

MAX_VISUALIZED_FAILURES = 4
REPORT_NAME = "load_validation_audit.json"
VISUALIZATION_NAME = "load_validation_examples.png"


@dataclass(frozen=True)
class ValidationFailureExample:
    """One load-time validation failure that may be rendered for inspection."""

    warning: str
    summary: str | None = None
    image_path: Path | None = None
    relative_path: Path | None = None
    split: str | None = None
    width: int | None = None
    height: int | None = None
    annotation: Annotation | None = None


def add_failure_example(
    examples: list[ValidationFailureExample] | None,
    example: ValidationFailureExample,
) -> None:
    if examples is not None and len(examples) < MAX_VISUALIZED_FAILURES:
        examples.append(example)


def build_load_validation_audit(
    warnings: Iterable[str],
    structured_examples: list[ValidationFailureExample],
    samples: list[Sample],
    task: Task,
    metadata: DatasetMetadata,
    *,
    dataset_name: str,
) -> tuple[dict[str, Any], Path | None]:
    """Count skipped validation failures and render at most four examples."""

    failure_warnings = [warning for warning in warnings if _is_skip_failure(warning)]
    if not failure_warnings:
        return {
            "status": "passed",
            "skipped_count": 0,
            "counts_by_category": {},
            "visualized_count": 0,
            "max_visualized_examples": MAX_VISUALIZED_FAILURES,
            "report": None,
            "visualization": None,
        }, None

    categories = Counter(_failure_category(warning) for warning in failure_warnings)
    examples = list(structured_examples[:MAX_VISUALIZED_FAILURES])
    remaining = list(failure_warnings)
    for example in examples:
        try:
            remaining.remove(example.warning)
        except ValueError:
            pass
    for warning in remaining:
        if len(examples) == MAX_VISUALIZED_FAILURES:
            break
        sample = _sample_for_warning(warning, samples)
        examples.append(
            ValidationFailureExample(
                warning=warning,
                summary=_concise_warning(warning),
                image_path=sample.image_path if sample else None,
                relative_path=sample.relative_path if sample else None,
                split=sample.split if sample else None,
                width=sample.width if sample else None,
                height=sample.height if sample else None,
            )
        )

    audit_dir = Path(tempfile.mkdtemp(prefix="dataset-fixer-validation-audit-"))
    visualization = audit_dir / VISUALIZATION_NAME
    print(
        f"Validation skip audit: {len(failure_warnings)} failed item(s); "
        f"showing {len(examples)} example(s) (maximum {MAX_VISUALIZED_FAILURES})."
    )
    from .visualization import visualize_validation_failures

    visualize_validation_failures(
        examples,
        task,
        metadata,
        total_count=len(failure_warnings),
        dataset_name=dataset_name,
        save_to=visualization,
        show=True,
    )
    return {
        "status": "passed_with_skips",
        "skipped_count": len(failure_warnings),
        "counts_by_category": dict(sorted(categories.items())),
        "visualized_count": len(examples),
        "max_visualized_examples": MAX_VISUALIZED_FAILURES,
        "report": None,
        "visualization": str(visualization),
    }, visualization


def stage_load_validation_audit(
    audit: dict[str, Any],
    visualization: Path | None,
    reports_dir: Path,
) -> tuple[dict[str, Any], Path | None]:
    """Persist a load audit inside an output staging tree."""

    detail = dict(audit)
    report_path = reports_dir / REPORT_NAME
    visualization_path = reports_dir / VISUALIZATION_NAME
    if int(detail.get("skipped_count", 0)) <= 0:
        report_path.unlink(missing_ok=True)
        visualization_path.unlink(missing_ok=True)
        detail["report"] = None
        detail["visualization"] = None
        return detail, None

    reports_dir.mkdir(parents=True, exist_ok=True)
    staged_visualization: Path | None = None
    if (
        visualization is not None
        and visualization.is_file()
        and visualization.name not in {"plots.png", "comparison.png"}
    ):
        staged_visualization = visualization_path
        shutil.copy2(visualization, staged_visualization)
        detail["visualization"] = f"reports/{VISUALIZATION_NAME}"
    else:
        visualization_path.unlink(missing_ok=True)
        detail["visualization"] = None
    detail["report"] = f"reports/{REPORT_NAME}"
    report_path.write_text(
        json.dumps(to_jsonable(detail), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return detail, staged_visualization


def _is_skip_failure(warning: str) -> bool:
    return warning.startswith("Skipped ") or warning.startswith("Ignored ")


def _concise_warning(warning: str) -> str:
    without_fix = warning.split("; fix:", 1)[0]
    return without_fix.rsplit(": ", 1)[-1]


def _failure_category(warning: str) -> str:
    prefixes = (
        "Skipped incompatible annotation file",
        "Skipped invalid annotation file",
        "Skipped invalid annotation",
        "Skipped invalid image record",
        "Skipped invalid image",
        "Skipped invalid label file",
        "Skipped invalid split entry",
        "Skipped invalid provenance record",
        "Ignored incomplete provenance",
        "Ignored orphan label",
        "Ignored invalid manifest",
    )
    return next((prefix for prefix in prefixes if warning.startswith(prefix)), "other")


def _sample_for_warning(warning: str, samples: list[Sample]) -> Sample | None:
    for sample in samples:
        if str(sample.image_path) in warning or str(_label_path(sample.image_path)) in warning:
            return sample
    return None


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    indices = [index for index, part in enumerate(parts) if part == "images"]
    if indices:
        parts[indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
