from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping
from pathlib import Path
from typing import Any


_SPLIT_ALIASES = {"valid": "val", "validation": "val"}


def _group_label(
    value: Hashable,
    image_path: str | Path,
    seen: dict[str, Any],
) -> str:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            "group_by must return a hashable group label; "
            f"received {type(value).__name__} for {image_path}"
        ) from exc
    label = str(value)
    if label in seen and seen[label] != value:
        raise ValueError(
            "group_by returned distinct values with the same string label "
            f"{label!r}; return unambiguous labels"
        )
    seen[label] = value
    return label


def resolve_evaluation_groups(
    cases: Iterable[tuple[str, str | Path]],
    group_by: Callable[[Path], Hashable] | None,
) -> dict[str, str] | None:
    """Resolve stable, human-readable group labels for evaluation cases.

    The callback receives the evaluation image path, matching the public
    ``Dataset.split(group_by=...)`` convention. Group labels are stored as
    strings in reports; ambiguous string representations are rejected rather
    than silently merging distinct callback results.
    """

    if group_by is None:
        return None

    resolved: dict[str, str] = {}
    seen: dict[str, Any] = {}
    for case_id, image_path in cases:
        value = group_by(Path(image_path))
        label = _group_label(value, image_path, seen)
        resolved[str(case_id)] = label

    if not resolved:
        raise ValueError("group_by cannot be evaluated on an empty cohort")
    return resolved


def resolve_group_splits(
    cases: Iterable[tuple[str, str | Path]],
    group_by: Callable[[Path], Hashable] | None,
) -> dict[str, tuple[str, ...]] | None:
    """Resolve the dataset splits in which every group occurs."""

    if group_by is None:
        return None

    resolved: defaultdict[str, set[str]] = defaultdict(set)
    seen: dict[str, Any] = {}
    for split, image_path in cases:
        value = group_by(Path(image_path))
        label = _group_label(value, image_path, seen)
        canonical_split = _SPLIT_ALIASES.get(str(split).lower(), str(split).lower())
        resolved[label].add(canonical_split)
    return {
        group: tuple(
            sorted(
                splits,
                key=lambda value: (
                    {"train": 0, "val": 1, "test": 2}.get(value, 3),
                    value,
                ),
            )
        )
        for group, splits in sorted(resolved.items())
    }


def annotate_group_splits(
    result: Mapping[str, Any],
    group_splits: Mapping[str, Iterable[str]] | None,
) -> None:
    """Attach dataset split membership to each mutable per-group metric row."""

    if not group_splits:
        return
    for row in result.get("per_group", []):
        group = str(row["group"])
        row["dataset_splits"] = list(group_splits.get(group, ()))
