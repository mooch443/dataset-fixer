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
        identity_id=sample.identity_id, clip_id=sample.clip_id,
        frame_index=sample.frame_index, local_id=sample.local_id, image_ref=sample.image_ref,
    )


def plan_split(
    samples: list[Sample],
    ratios: Mapping[str, float],
    *,
    source_splits: Iterable[str] | None,
    group_by: Callable[[Path], Hashable] | None,
    assign: Callable[[Path], str | None] | None,
    seed: int,
    classes_must_appear_in: tuple[str, ...] = (),
) -> tuple[list[Sample], dict[str, Any], dict[str, str]]:
    normalized = normalize_split_ratios(ratios)
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

    group_classes = {key: Counter(label for sample in values for label in {a.class_id for a in sample.annotations})
                     for key, values in groups.items()} if classes_must_appear_in else None
    allocated = allocate_groups({key: len(value) for key, value in groups.items()}, normalized, locked, seed,
                                group_classes=group_classes, classes_must_appear_in=classes_must_appear_in)
    for group, target in allocated.items():
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
    if classes_must_appear_in:
        settings["classes_must_appear_in"] = classes_must_appear_in
    projected = [clone_sample(sample, split=assignments[str(sample.image_path)]) for sample in chosen]
    return projected, settings, assignments


def allocate_groups(sizes, ratios, locked, seed, *, group_classes=None, classes_must_appear_in=()):
    """Assign indivisible groups using the shared weighted-deficit policy."""
    if classes_must_appear_in:
        return allocate_class_groups(sizes, ratios, locked, seed, group_classes, classes_must_appear_in)
    counts: Counter[str] = Counter()
    assigned = dict(locked)
    for group, target in locked.items():
        counts[target] += sizes[group]
    remaining = [group for group in sizes if group not in locked]
    random.Random(seed).shuffle(remaining)
    targets = {split: ratio * sum(sizes.values()) for split, ratio in ratios.items()}
    eligible = [split for split, ratio in ratios.items() if ratio > 0]
    for group in remaining:
        target = max(eligible, key=lambda split: (targets[split] - counts[split]) / max(targets[split], 1))
        counts[target] += sizes[group]
        assigned[group] = target
    return assigned


def allocate_class_groups(sizes, ratios, locked, seed, group_classes, required):
    """Assign whole groups: mandatory coverage, sample ratios, then broader coverage.

    SciPy solves two sparse integer programs with the same constraints. Groups
    with identical sizes and per-class counts are interchangeable, so individual
    crops compress to one group type per identity rather than millions of variables.
    """
    if isinstance(required, str):
        raise TypeError("classes_must_appear_in must be a sequence of split names, e.g. ('train',)")
    required = tuple(dict.fromkeys(normalize_split(split) for split in required))
    if any(ratios.get(split, 0) <= 0 for split in required):
        raise ValueError("classes_must_appear_in must name splits with positive ratios")
    group_classes = {group: Counter(labels) for group, labels in group_classes.items()}
    classes = sorted({label for labels in group_classes.values() for label in labels}, key=repr)
    covered = {split: set() for split in required}
    for group, split in locked.items():
        if split in covered:
            covered[split].update(group_classes.get(group, ()))
    availability = Counter(label for group in sizes if group not in locked for label in group_classes.get(group, ()))
    issues = []
    for label in classes:
        needed = [split for split in required if label not in covered[split]]
        if len(needed) > availability[label]:
            issues.append(ValidationIssue(
                f"classes_must_appear_in cannot cover class {label!r} in {needed}: "
                f"needs {len(needed)} distinct unassigned groups, but only {availability[label]} remain. "
                "Change group_by, explicit assignments, or the required splits; groups are never divided."
            ))
    if issues:
        raise DatasetValidationError(issues)
    splits = [split for split, ratio in ratios.items() if ratio > 0]
    empty_splits = set(splits) - set(locked.values())
    if len(sizes) - len(locked) < len(empty_splits):
        raise DatasetValidationError(
            f"Positive split ratios require nonempty splits {sorted(empty_splits)}, but only "
            f"{len(sizes)-len(locked)} unassigned groups remain. Change ratios, group_by, or explicit assignments."
        )
    if len(locked) == len(sizes):
        return dict(locked)

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix, vstack

    patterns = defaultdict(list)
    for group, size in sizes.items():
        if group not in locked:
            signature = (size, tuple(sorted(group_classes.get(group, {}).items(), key=lambda item: repr(item[0]))))
            patterns[signature].append(group)
    patterns = list(patterns.items())
    ns = len(splits)
    nx, nz = len(patterns) * ns, len(classes) * ns
    # Allocation counts, binary class-presence flags, and absolute fraction errors.
    nv = nx + nz + (len(classes) + 1) * ns
    lower, upper = np.zeros(nv), np.full(nv, np.inf)
    integrality = np.zeros(nv)
    integrality[:nx+nz] = 1
    upper[nx:nx+nz] = 1
    rows, columns, values, lo, hi = [], [], [], [], []

    def constraint(entries, minimum, maximum):
        for column, value in entries:
            if value:
                rows.append(len(lo)); columns.append(column); values.append(value)
        lo.append(minimum); hi.append(maximum)

    by_class = defaultdict(list)
    for pattern, ((size, counts), groups) in enumerate(patterns):
        upper[pattern*ns:(pattern+1)*ns] = len(groups)
        constraint([(pattern*ns+s, 1) for s in range(ns)], len(groups), len(groups))
        for label, count in counts:
            by_class[label].append((pattern, count))
    for s, split in enumerate(splits):
        constraint([(p*ns+s, 1) for p in range(len(patterns))], int(split in empty_splits), np.inf)
    fixed = {split: Counter() for split in ratios}
    for group, split in locked.items():
        fixed[split].update(group_classes.get(group, ()))
    for c, label in enumerate(classes):
        for s, split in enumerate(splits):
            z = nx + c*ns + s
            lower[z] = int(split in required)
            constraint([(p*ns+s, 1) for p, _ in by_class[label]] + [(z, -1)],
                       -int(fixed[split][label] > 0), np.inf)

    ratio_cost = np.zeros(nv)
    # Equal weight to overall image fractions and the mean per-class fractions;
    # large identities must not hide a badly distributed smaller identity.
    totals = Counter()
    for counts in group_classes.values():
        totals.update(counts)
    for metric, label in enumerate([None, *classes]):
        total = sum(sizes.values()) if label is None else totals[label]
        entries = [(p, signature[0]) for p, (signature, _) in enumerate(patterns)] if label is None else by_class[label]
        for s, split in enumerate(splits):
            d = nx + nz + metric*ns + s
            fixed_count = sum(sizes[g] for g, target in locked.items() if target == split) if label is None else fixed[split][label]
            target = ratios[split] - fixed_count / total
            fractions = [(p*ns+s, count/total) for p, count in entries]
            constraint(fractions + [(d, -1)], -np.inf, target)
            constraint(fractions + [(d, 1)], target, np.inf)
            ratio_cost[d] = 1 if label is None else 1/len(classes)
    matrix = coo_matrix((values, (rows, columns)), shape=(len(lo), nv)).tocsc()
    bounds = Bounds(lower, upper)

    def solve(cost, matrix, lo, hi):
        result = milp(cost, integrality=integrality, bounds=bounds,
                      constraints=LinearConstraint(matrix, lo, hi), options={"mip_rel_gap": 0})
        if result.status == 2:
            raise DatasetValidationError(
                f"classes_must_appear_in={required!r} is incompatible with whole-group membership "
                "and explicit assignments. Class overlaps prevent simultaneous coverage "
                "or leave a positive-ratio split empty; change group_by, assignments, or required splits."
            )
        if not result.success:
            raise RuntimeError(f"Class-aware allocation failed: {result.message}")
        return result.x

    coverage_cost = np.zeros(nv)
    coverage_cost[nx:nx+nz] = -1
    first = solve(ratio_cost, matrix, lo, hi)
    error = float(ratio_cost @ first)
    # Prefer broader coverage among equally good proportional allocations.
    # Coverage must not silently turn e.g. 70/15/15 into 48/21/31.
    nonzero = np.flatnonzero(ratio_cost)
    error_row = coo_matrix((ratio_cost[nonzero], (np.zeros(len(nonzero), dtype=int), nonzero)), shape=(1, nv)).tocsc()
    solution = solve(coverage_cost, vstack([matrix, error_row], format="csc"), [*lo, -np.inf], [*hi, error+1e-8])
    assigned = dict(locked)
    rng = random.Random(seed)
    for p, (_, groups) in enumerate(patterns):
        rng.shuffle(groups)
        for s, split in enumerate(splits):
            for _ in range(round(solution[p*ns+s])):
                assigned[groups.pop()] = split
    return assigned


def normalize_split_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    """Normalize non-negative split weights into fractions that sum to one."""

    parsed = {normalize_split(key): float(value) for key, value in ratios.items()}
    total = sum(parsed.values())
    if (
        not parsed
        or any(not math.isfinite(value) or value < 0 for value in parsed.values())
        or not math.isfinite(total)
        or total <= 0
    ):
        raise ValueError(
            f"Split ratios must contain finite non-negative weights with a positive total, got {parsed}"
        )
    return {split: value / total for split, value in parsed.items()}


def resolve_removed_classes(
    metadata: DatasetMetadata,
    selectors: Iterable[str | int],
    *,
    merge_into: str | int | None = None,
) -> tuple[set[int], dict[int, int], DatasetMetadata]:
    removed = resolve_class_selectors(
        metadata,
        selectors,
        empty_message="At least one class must be removed",
    )
    reverse: dict[str, list[int]] = defaultdict(list)
    for class_id, class_name in metadata.names.items():
        reverse[class_name].append(class_id)
    merge_target: int | None = None
    if merge_into is not None:
        if isinstance(merge_into, int):
            if merge_into not in metadata.names:
                raise ValueError(
                    f"Unknown merge target class ID {merge_into}; available IDs are {sorted(metadata.names)}"
                )
            merge_target = merge_into
        else:
            matches = reverse.get(merge_into, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Merge target class name {merge_into!r} matched {len(matches)} classes; "
                    f"available names are {list(reverse)}"
                )
            merge_target = matches[0]
        if merge_target in removed:
            raise ValueError("merge_into must select a class that is not being removed")
    remaining = [class_id for class_id in sorted(metadata.names) if class_id not in removed]
    if not remaining:
        raise DatasetValidationError("Removing these classes would leave the dataset with no classes")
    mapping = {old: new for new, old in enumerate(remaining)}
    if merge_target is not None:
        mapping.update({class_id: mapping[merge_target] for class_id in removed})
    projected = metadata.copy()
    projected.names = {mapping[old]: metadata.names[old] for old in remaining}
    projected.radii = {
        mapping[old]: metadata.radii[old] for old in remaining if old in metadata.radii
    }
    projected.kpt_names = {
        mapping[old]: metadata.kpt_names[old]
        for old in remaining
        if old in metadata.kpt_names
    }
    return removed, mapping, projected


def resolve_class_selectors(
    metadata: DatasetMetadata,
    selectors: Iterable[str | int],
    *,
    empty_message: str = "At least one class must be selected",
) -> set[int]:
    """Resolve class names or IDs without changing the class schema."""

    reverse: dict[str, list[int]] = defaultdict(list)
    for class_id, class_name in metadata.names.items():
        reverse[class_name].append(class_id)
    resolved: set[int] = set()
    for selector in selectors:
        if isinstance(selector, int):
            if selector not in metadata.names:
                raise ValueError(f"Unknown class ID {selector}; available IDs are {sorted(metadata.names)}")
            resolved.add(selector)
        else:
            matches = reverse.get(selector, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Class name {selector!r} matched {len(matches)} classes; available names are {list(reverse)}"
                )
            resolved.add(matches[0])
    if not resolved:
        raise ValueError(empty_message)
    return resolved


def resolve_renamed_classes(
    metadata: DatasetMetadata,
    renames: Mapping[str | int, str],
) -> tuple[dict[int, dict[str, str]], DatasetMetadata]:
    """Resolve class selectors and validate the final name schema."""

    if not renames:
        raise ValueError("At least one class must be renamed")
    reverse: dict[str, list[int]] = defaultdict(list)
    for class_id, class_name in metadata.names.items():
        reverse[class_name].append(class_id)
    resolved: dict[int, str] = {}
    for selector, new_name in renames.items():
        if not isinstance(new_name, str) or not new_name.strip():
            raise ValueError(f"New class name for {selector!r} must be a non-empty string")
        if isinstance(selector, int):
            if selector not in metadata.names:
                raise ValueError(f"Unknown class ID {selector}; available IDs are {sorted(metadata.names)}")
            class_id = selector
        else:
            matches = reverse.get(selector, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Class name {selector!r} matched {len(matches)} classes; available names are {list(reverse)}"
                )
            class_id = matches[0]
        if class_id in resolved and resolved[class_id] != new_name:
            raise ValueError(f"Class {class_id} was assigned conflicting new names")
        resolved[class_id] = new_name

    final_names = {class_id: resolved.get(class_id, class_name) for class_id, class_name in metadata.names.items()}
    duplicates = sorted({name for name in final_names.values() if list(final_names.values()).count(name) > 1})
    if duplicates:
        raise ValueError(f"Renaming would create duplicate class names: {duplicates}")
    renamed = {
        class_id: {"from": metadata.names[class_id], "to": final_names[class_id]}
        for class_id in sorted(resolved)
        if metadata.names[class_id] != final_names[class_id]
    }
    if not renamed:
        raise ValueError("The requested class renames do not change any names")
    projected = metadata.copy()
    projected.names = final_names
    return renamed, projected


def project_remove_classes(
    samples: list[Sample],
    *,
    selected_splits: set[str],
    removed_classes: set[int],
    mapping: dict[int, int],
    drop_empty_images: bool,
    drop_containing_images: bool,
) -> list[Sample]:
    projected: list[Sample] = []
    for sample in samples:
        if sample.split not in selected_splits:
            continue
        if drop_containing_images and any(
            annotation.class_id in removed_classes
            for annotation in sample.annotations
        ):
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


def project_move_images_with_classes(
    samples: list[Sample],
    *,
    selected_splits: set[str],
    class_ids: set[int],
    to_split: str,
    group_by: Callable[[Path], Hashable] | None,
) -> tuple[list[Sample], dict[str, Any], dict[str, dict[str, Any]]]:
    """Move matching images and every member of a matching physical group."""

    before = Counter(sample.split for sample in samples)
    groups, group_for_path = _group_samples(samples, group_by)
    matched_paths = {
        str(sample.image_path)
        for sample in samples
        if sample.split in selected_splits
        and any(annotation.class_id in class_ids for annotation in sample.annotations)
    }
    triggered_groups = {
        group_for_path[path]
        for path in matched_paths
    }
    matched = len(matched_paths)
    if not matched:
        raise ValueError(
            "No images containing the selected classes were found in source_splits"
        )

    projected: list[Sample] = []
    details: dict[str, dict[str, Any]] = {}
    for sample in samples:
        path = str(sample.image_path)
        group = group_for_path[path]
        selected_by_group = group in triggered_groups
        target = to_split if selected_by_group else sample.split
        projected.append(clone_sample(sample, split=target))
        details[path] = {
            "group": repr(group),
            "direct_class_match": path in matched_paths,
            "selected_by_group": selected_by_group,
            "from_split": sample.split,
            "to_split": target,
            "moved": sample.split != target,
        }
    moved = sum(detail["moved"] for detail in details.values())
    if not moved:
        raise ValueError(
            f"All triggered groups are already in split {to_split!r}"
        )
    after = Counter(sample.split for sample in projected)
    selected_group_images = sum(len(groups[group]) for group in triggered_groups)
    return projected, {
        "matched_images": matched,
        "matched_groups": len(triggered_groups),
        "selected_group_images": selected_group_images,
        "group_expansion_images": selected_group_images - matched,
        "moved_images": moved,
        "already_in_target_images": selected_group_images - moved,
        "distribution_before": dict(sorted(before.items())),
        "distribution_after": dict(sorted(after.items())),
    }, details


def project_move_n_groups(
    samples: list[Sample],
    *,
    n: int,
    from_split: str,
    to_split: str,
    group_by: Callable[[Path], Hashable],
    seed: int,
) -> tuple[list[Sample], dict[str, Any], dict[str, dict[str, Any]]]:
    """Select source groups deterministically and move every group member."""

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if from_split == to_split:
        raise ValueError("from_split and to_split must be different")
    groups, group_for_path = _group_samples(samples, group_by)
    eligible = [
        group
        for group, group_samples in groups.items()
        if any(sample.split == from_split for sample in group_samples)
    ]
    if n > len(eligible):
        raise ValueError(
            f"Requested {n} groups from {from_split!r}, but only {len(eligible)} are available"
        )
    selected_groups = set(random.Random(seed).sample(eligible, n))
    before = Counter(sample.split for sample in samples)
    projected: list[Sample] = []
    details: dict[str, dict[str, Any]] = {}
    for sample in samples:
        path = str(sample.image_path)
        group = group_for_path[path]
        selected = group in selected_groups
        target = to_split if selected else sample.split
        projected.append(clone_sample(sample, split=target))
        details[path] = {
            "group": repr(group),
            "selected": selected,
            "from_split": sample.split,
            "to_split": target,
            "moved": sample.split != target,
        }
    after = Counter(sample.split for sample in projected)
    selected_images = sum(len(groups[group]) for group in selected_groups)
    moved_images = sum(detail["moved"] for detail in details.values())
    return projected, {
        "requested_groups": n,
        "eligible_groups": len(eligible),
        "selected_groups": n,
        "selected_group_images": selected_images,
        "moved_images": moved_images,
        "already_in_target_images": selected_images - moved_images,
        "distribution_before": dict(sorted(before.items())),
        "distribution_after": dict(sorted(after.items())),
    }, details


def _group_samples(
    samples: list[Sample],
    group_by: Callable[[Path], Hashable] | None,
) -> tuple[dict[Hashable, list[Sample]], dict[str, Hashable]]:
    groups: dict[Hashable, list[Sample]] = defaultdict(list)
    group_for_path: dict[str, Hashable] = {}
    for sample in samples:
        group = group_by(sample.image_path) if group_by else str(sample.image_path)
        try:
            hash(group)
        except TypeError as exc:
            raise TypeError(
                f"group_by must return a hashable value, got {type(group).__name__}"
            ) from exc
        groups[group].append(sample)
        group_for_path[str(sample.image_path)] = group
    return groups, group_for_path


def select_empty_images(
    samples: list[Sample],
    *,
    max_empty_fraction: float,
    selected_splits: set[str],
    seed: int,
) -> tuple[list[Sample], dict[str, dict[str, Any]]]:
    if not 0 <= max_empty_fraction < 1:
        raise ValueError("max_empty_fraction must be in [0, 1)")
    keep_ids: set[int] = set()
    summary: dict[str, dict[str, Any]] = {}
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
            "before": {"annotated": len(positives), "background": len(empties)},
            "after": {"annotated": len(positives), "background": len(kept_empty)},
            "background_fraction_after": len(kept_empty) / len(kept) if kept else 0.0,
        }
    return [clone_sample(sample) for sample in samples if id(sample) in keep_ids], summary


def derived_name(current: str, operation: str, settings: dict[str, Any]) -> str:
    detail = operation
    if operation == "split":
        detail = "split-" + "-".join(
            f"{key}{int(value * 100)}" for key, value in settings["ratios"].items()
        )
    elif operation == "move-images-with-classes":
        selected = "-".join(
            slugify(value)
            for value in list(settings["selected_classes"].values())[:3]
        )
        detail = f"move-{selected}-to-{settings['to_split']}"
    elif operation == "move-n-groups":
        detail = (
            f"move-{settings['selected_groups']}-groups-"
            f"{settings['from_split']}-to-{settings['to_split']}"
        )
    elif operation == "remove-classes":
        removed = "-".join(
            slugify(value) for value in list(settings["removed_classes"].values())[:3]
        )
        if settings.get("merge_into"):
            detail = f"merge-{removed}-into-{slugify(settings['merge_into']['output_class_name'])}"
        else:
            detail = f"remove-{removed}"
    elif operation == "rename-classes":
        detail = "rename-" + "-".join(
            slugify(value["to"]) for value in list(settings["renamed_classes"].values())[:3]
        )
    elif operation == "rebalance-empty":
        detail = f"empty-max{int(float(settings['max_empty_fraction']) * 100)}"
    elif operation == "tile":
        detail = f"tile-{settings['mode']}-{settings['tile_size']}"
    elif operation == "augment":
        detail = f"augment-{settings['copies']}x"
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
