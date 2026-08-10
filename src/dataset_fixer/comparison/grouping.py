from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import Any


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
        resolved[str(case_id)] = label

    if not resolved:
        raise ValueError("group_by cannot be evaluated on an empty cohort")
    return resolved
