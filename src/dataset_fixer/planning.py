from __future__ import annotations

import inspect
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Mapping

from .errors import DatasetValidationError, ValidationIssue
from .models import Annotation, DatasetMetadata, Sample
from .utils import normalize_split, settings_fingerprint, slugify, to_jsonable


@dataclass(frozen=True)
class PlannedOperation:
    """One immutable, in-memory transformation waiting for ``export()``."""

    kind: str
    kwargs: dict[str, Any]
    settings: dict[str, Any]

    def public_record(self) -> dict[str, Any]:
        return {
            "operation": self.kind,
            "settings": to_jsonable(self.settings),
            "settings_fingerprint": settings_fingerprint(self.settings),
            "status": "pending export",
        }


def clone_sample(
    sample: Sample,
    *,
    split: str | None = None,
    annotations: list[Annotation] | None = None,
) -> Sample:
    return Sample(
        image_path=sample.image_path,
        relative_path=sample.relative_path,
        split=split or sample.split,
        width=sample.width,
        height=sample.height,
        annotations=list(sample.annotations) if annotations is None else annotations,
        source_sha256=sample.source_sha256,
        provenance=dict(sample.provenance),
    )


def plan_split(
    samples: list[Sample],
    ratios: Mapping[str, float],
    *,
    source_splits: Iterable[str] | None,
    group_by: Callable[[Path], Hashable] | None,
    assign: Callable[[Path], str | None] | None,
    seed: int,
) -> tuple[list[Sample], dict[str, Any], dict[str, str]]:
    normalized = {normalize_split(key): float(value) for key, value in ratios.items()}
    if any(value < 0 for value in normalized.values()) or not math.isclose(
        sum(normalized.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError(f"Split ratios must be non-negative and sum to 1.0, got {normalized}")
    selected = {normalize_split(value) for value in source_splits} if source_splits else {
        sample.split for sample in samples
    }
    chosen = [sample for sample in samples if sample.split in selected]
    if not chosen:
        raise DatasetValidationError("No images selected for splitting")

    groups: dict[Hashable, list[Sample]] = defaultdict(list)
    locked: dict[Hashable, str] = {}
    assignments: dict[str, str] = {}
    for sample in chosen:
        group = group_by(sample.image_path) if group_by else str(sample.image_path)
        groups[group].append(sample)
        target = assign(sample.image_path) if assign else None
        if target is None:
            continue
        target = normalize_split(target)
        if target not in normalized:
            raise ValueError(f"Assignment callback returned {target!r}, which is absent from ratios")
        if group in locked and locked[group] != target:
            raise DatasetValidationError(
                ValidationIssue(
                    "One physical group received conflicting explicit split assignments",
                    value={"group": repr(group), "first": locked[group], "second": target},
                )
            )
        locked[group] = target

    counts: Counter[str] = Counter()
    for group, target in locked.items():
        counts[target] += len(groups[group])
        assignments.update({str(sample.image_path): target for sample in groups[group]})
    remaining = [group for group in groups if group not in locked]
    random.Random(seed).shuffle(remaining)
    targets = {split: ratio * len(chosen) for split, ratio in normalized.items()}
    for group in remaining:
        eligible = [split for split, ratio in normalized.items() if ratio > 0]
        target = max(eligible, key=lambda split: (targets[split] - counts[split]) / max(targets[split], 1))
        counts[target] += len(groups[group])
        assignments.update({str(sample.image_path): target for sample in groups[group]})

    resolved_groups = {
        repr(group): {
            "size": len(group_samples),
            "split": assignments[str(group_samples[0].image_path)],
            "images": [str(sample.image_path) for sample in group_samples],
        }
        for group, group_samples in groups.items()
    }
    settings = {
        "ratios": normalized,
        "source_splits": sorted(selected),
        "seed": seed,
        "group_by": callback_description(group_by),
        "assign": callback_description(assign),
        "resolved_groups": resolved_groups,
        "resolved_assignments": assignments,
    }
    projected = [clone_sample(sample, split=assignments[str(sample.image_path)]) for sample in chosen]
    return projected, settings, assignments


def resolve_removed_classes(
    metadata: DatasetMetadata, selectors: Iterable[str | int]
) -> tuple[set[int], dict[int, int], DatasetMetadata]:
    reverse: dict[str, list[int]] = defaultdict(list)
    for class_id, class_name in metadata.names.items():
        reverse[class_name].append(class_id)
    removed: set[int] = set()
    for selector in selectors:
        if isinstance(selector, int):
            if selector not in metadata.names:
                raise ValueError(f"Unknown class ID {selector}; available IDs are {sorted(metadata.names)}")
            removed.add(selector)
        else:
            matches = reverse.get(selector, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Class name {selector!r} matched {len(matches)} classes; available names are {list(reverse)}"
                )
            removed.add(matches[0])
    if not removed:
        raise ValueError("At least one class must be removed")
    remaining = [class_id for class_id in sorted(metadata.names) if class_id not in removed]
    if not remaining:
        raise DatasetValidationError("Removing these classes would leave the dataset with no classes")
    mapping = {old: new for new, old in enumerate(remaining)}
    projected = metadata.copy()
    projected.names = {mapping[old]: metadata.names[old] for old in mapping}
    projected.radii = {mapping[old]: metadata.radii[old] for old in mapping if old in metadata.radii}
    projected.kpt_names = {
        mapping[old]: metadata.kpt_names[old] for old in mapping if old in metadata.kpt_names
    }
    return removed, mapping, projected


def project_remove_classes(
    samples: list[Sample],
    *,
    selected_splits: set[str],
    mapping: dict[int, int],
    drop_empty_images: bool,
) -> list[Sample]:
    projected: list[Sample] = []
    for sample in samples:
        if sample.split not in selected_splits:
            continue
        annotations = [
            annotation.clone(class_id=mapping[annotation.class_id])
            for annotation in sample.annotations
            if annotation.class_id in mapping
        ]
        if drop_empty_images and not annotations:
            continue
        projected.append(clone_sample(sample, annotations=annotations))
    return projected


def select_empty_images(
    samples: list[Sample],
    *,
    max_empty_fraction: float,
    selected_splits: set[str],
    seed: int,
) -> tuple[list[Sample], dict[str, dict[str, int | float]]]:
    if not 0 <= max_empty_fraction < 1:
        raise ValueError("max_empty_fraction must be in [0, 1)")
    keep_ids: set[int] = set()
    summary: dict[str, dict[str, int | float]] = {}
    for split in sorted({sample.split for sample in samples}):
        rows = [sample for sample in samples if sample.split == split]
        positives = [sample for sample in rows if sample.annotations]
        empties = [sample for sample in rows if not sample.annotations]
        if split not in selected_splits:
            kept_empty = empties
        elif not positives:
            kept_empty = []
        else:
            allowed = int(math.floor(len(positives) * max_empty_fraction / (1 - max_empty_fraction)))
            allowed = min(allowed, len(empties))
            shuffled = list(empties)
            random.Random(f"{seed}:{split}").shuffle(shuffled)
            kept_empty = shuffled[:allowed]
        kept = positives + kept_empty
        keep_ids.update(id(sample) for sample in kept)
        summary[split] = {
            "positive_images": len(positives),
            "empty_images_before": len(empties),
            "empty_images_after": len(kept_empty),
            "empty_fraction_after": len(kept_empty) / len(kept) if kept else 0.0,
        }
    return [clone_sample(sample) for sample in samples if id(sample) in keep_ids], summary


def derived_name(current: str, operation: str, settings: dict[str, Any]) -> str:
    detail = operation
    if operation == "split":
        detail = "split-" + "-".join(
            f"{key}{int(value * 100)}" for key, value in settings["ratios"].items()
        )
    elif operation == "remove-classes":
        detail = "remove-" + "-".join(
            slugify(value) for value in list(settings["removed_classes"].values())[:3]
        )
    elif operation == "rebalance-empty":
        detail = f"empty-max{int(float(settings['max_empty_fraction']) * 100)}"
    elif operation == "tile":
        detail = f"tile-{settings['mode']}-{settings['tile_size']}"
    return slugify(f"{current}__{detail}__{settings_fingerprint(settings)}")


def callback_description(callback: Callable | None) -> dict[str, Any] | None:
    if callback is None:
        return None
    try:
        source = inspect.getsource(callback).strip()
    except (OSError, TypeError):
        source = None
    return {
        "module": getattr(callback, "__module__", None),
        "qualname": getattr(callback, "__qualname__", None),
        "source": source,
    }
