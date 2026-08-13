from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_MODEL_TYPE_COLORS = {
    "semantic": "#0F766E",
    "instance": "#2563EB",
    "yolox": "#C2410C",
    "nnunet": "#7C3AED",
    "other": "#4B5563",
}
_UPSCALE_COLOR = "#B45309"
_RESOLUTION_COLOR = "#475569"
_SHORT_RUN_PATTERN = re.compile(r"^[a-z0-9]{8}(?:_20\d{6}_\d{6})?$")
_COMPACT_TIMESTAMP_PATTERN = re.compile(
    r"(?<!\d)(20\d{6})[_-](\d{6})(?!\d)"
)
_ISO_TIMESTAMP_PATTERN = re.compile(
    r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})[_-](\d{2})[-_](\d{2})"
    r"(?:[-_](\d{2}))?(?!\d)"
)
_TRAINING_DATASET_PREFIX = re.compile(
    r"^(?:islands-)?(?:128-)?\d{2}\.\d{2}\.\d{4}-merged-1class_"
    r"(?:masks|yolo)[_-]*",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ModelBadge:
    """One compact presentation badge for a model row or panel.

    text:
        Visible badge text.
    color:
        CSS-compatible presentation color.
    """

    text: str
    color: str


def model_type_color(model_type: str) -> str:
    """Return the report color assigned to a model architecture family."""

    normalized = model_type.lower()
    if normalized.startswith("nnunet"):
        return _MODEL_TYPE_COLORS["nnunet"]
    if normalized.startswith("yolox"):
        return _MODEL_TYPE_COLORS["yolox"]
    if normalized.endswith("-sem") or "semantic" in normalized:
        return _MODEL_TYPE_COLORS["semantic"]
    if normalized.endswith("-seg") or "instance" in normalized:
        return _MODEL_TYPE_COLORS["instance"]
    return _MODEL_TYPE_COLORS["other"]


def model_label(model: Any) -> str:
    """Return a uniform checkpoint-provenance label for a model.

    The visible identity is the checkpoint/run creation time plus an optional
    checkpoint digest. Architecture, upscale factor, and input resolution are
    rendered separately as badges. Dataset-version dates in training-export
    prefixes are deliberately stripped before filename timestamps are parsed.

    model:
        Model or model metadata mapping.
    """

    row = _model_metadata(model)
    name = str(row.get("model") or row.get("name") or "model")
    source = str(row.get("model_source") or row.get("source_key") or "")
    lowered_name = name.lower()
    candidate: str

    if "sweep" in lowered_name:
        candidate = name
    elif source.startswith("wandb:"):
        source_token = source.rsplit("/", 1)[-1]
        if _SHORT_RUN_PATTERN.fullmatch(source_token):
            candidate = source_token.split("_", 1)[0]
        else:
            candidate = source_token
    elif source:
        candidate = Path(source).stem
    else:
        parts = name.split("__")
        candidate = parts[1] if len(parts) >= 2 else name

    candidate = _strip_dataset_prefix(candidate, row)
    candidate = _strip_badge_prefix(candidate, row)
    timestamp = _provenance_timestamp(row)
    if timestamp is None:
        timestamp = _timestamp_in_text(candidate)
    if timestamp is None:
        timestamp = _checkpoint_file_timestamp(row, source)
    candidate = timestamp or _shorten_middle(candidate or name, 58)
    digest = _checkpoint_digest(row)
    if digest:
        return f"{candidate} · {digest}"
    return candidate


def model_badges(model: Any) -> tuple[ModelBadge, ...]:
    """Return architecture, upscale, and model-input-resolution badges.

    model:
        Model or model metadata mapping.
    """

    row = _model_metadata(model)
    badges: list[ModelBadge] = []
    model_type = str(row.get("model_type") or "").strip()
    if model_type:
        badges.append(ModelBadge(model_type, model_type_color(model_type)))

    upscale = row.get("upscale_factor")
    try:
        parsed_upscale = float(upscale)
    except (TypeError, ValueError):
        parsed_upscale = math.nan
    if math.isfinite(parsed_upscale) and parsed_upscale > 0:
        label = (
            f"{int(parsed_upscale)}×"
            if parsed_upscale.is_integer()
            else f"{parsed_upscale:g}×"
        )
        badges.append(ModelBadge(label, _UPSCALE_COLOR))

    resolution = _resolution_label(row)
    if resolution:
        badges.append(ModelBadge(resolution, _RESOLUTION_COLOR))
    return tuple(badges)


def model_badge_text(model: Any) -> str:
    """Return a plain-text form for plots that cannot host colored badges."""

    return " · ".join(badge.text for badge in model_badges(model))


def model_full_label(model: Any) -> str:
    """Return the normalized name followed by its plain-text identity badges.

    model:
        Model or model metadata mapping.
    """

    values = (model_label(model), model_badge_text(model))
    return " · ".join(value for value in values if value)


def _model_metadata(model: Any) -> Mapping[str, Any]:
    if isinstance(model, Mapping):
        return model
    describe = getattr(model, "describe", None)
    if not callable(describe):
        raise TypeError("model must be a Model or model metadata mapping")
    value = describe()
    if not isinstance(value, Mapping):
        raise TypeError("model.describe() must return a metadata mapping")
    return value


def _resolution_label(row: Mapping[str, Any]) -> str | None:
    value = (
        row.get("native_resolution")
        or row.get("effective_prediction_resolution")
        or row.get("resolution")
    )
    if value is None:
        size = row.get("effective_prediction_size") or row.get("input_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            value = f"{size[0]}px" if size[0] == size[1] else f"{size[1]}×{size[0]}px"
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{int(value)}px"
    label = str(value).strip().replace(" ", "")
    if label.lower() in {"unknown", "none", "n/a", "na"}:
        return None
    return label or None


def _strip_dataset_prefix(value: str, row: Mapping[str, Any]) -> str:
    candidate = value.strip().strip("_- ")
    model_name = str(row.get("model") or row.get("name") or "")
    canonical_parts = model_name.split("__")
    prefixes = [canonical_parts[0]] if len(canonical_parts) >= 2 else []
    source_dataset = str(row.get("source_dataset_zip") or "")
    if source_dataset:
        prefixes.append(Path(source_dataset).stem)
    for prefix in sorted(set(prefixes), key=len, reverse=True):
        if candidate.lower().startswith(prefix.lower()):
            candidate = candidate[len(prefix) :].strip("_- ")
            break
    return _TRAINING_DATASET_PREFIX.sub("", candidate).strip("_- ")


def _checkpoint_digest(row: Mapping[str, Any]) -> str | None:
    for key in (
        "checkpoint_sha256_short",
        "model_sha256_short",
        "checkpoint_sha256",
        "model_sha256",
        "digest",
    ):
        value = str(row.get(key) or "").strip().lower()
        if len(value) >= 7 and all(
            character in "0123456789abcdef" for character in value
        ):
            return value[:8]
    return None


def _strip_badge_prefix(value: str, row: Mapping[str, Any]) -> str:
    """Remove fields already rendered as model-identity badges."""

    candidate = value.strip().strip("_- ")
    prefixes: list[str] = []
    model_type = str(row.get("model_type") or "").strip()
    if model_type:
        prefixes.append(model_type)
    resolution = _resolution_label(row)
    if resolution:
        prefixes.append(resolution)
    upscale = row.get("upscale_factor")
    try:
        parsed_upscale = float(upscale)
    except (TypeError, ValueError):
        parsed_upscale = math.nan
    if math.isfinite(parsed_upscale) and parsed_upscale > 0:
        value = f"{parsed_upscale:g}"
        prefixes.extend((f"{value}x", f"{value}×"))

    changed = True
    while candidate and changed:
        changed = False
        for prefix in prefixes:
            match = re.match(
                rf"^{re.escape(prefix)}(?:[-_ ]+|$)",
                candidate,
                flags=re.IGNORECASE,
            )
            if match is None:
                continue
            candidate = candidate[match.end() :].strip("_- ")
            changed = True
            break
    return candidate


def _timestamp_in_text(value: str) -> str | None:
    """Return the last model-specific timestamp in already de-prefixed text."""

    compact = list(_COMPACT_TIMESTAMP_PATTERN.finditer(value))
    iso = list(_ISO_TIMESTAMP_PATTERN.finditer(value))
    if compact:
        date, time = compact[-1].groups()
        timestamp = (
            f"{date[:4]}-{date[4:6]}-{date[6:]} "
            f"{time[:2]}:{time[2:4]}:{time[4:]}"
        )
    elif iso:
        year, month, day, hour, minute, second = iso[-1].groups()
        timestamp = f"{year}-{month}-{day} {hour}:{minute}"
        if second:
            timestamp += f":{second}"
    else:
        return None
    return timestamp


def _provenance_timestamp(row: Mapping[str, Any]) -> str | None:
    for key in (
        "source_created_at",
        "checkpoint_created_at",
        "run_created_at",
        "created_at",
    ):
        if timestamp := _format_timestamp(row.get(key)):
            return timestamp
    for section_key in ("provenance", "run", "manifest"):
        section = row.get(section_key)
        if not isinstance(section, Mapping):
            continue
        for key in (
            "checkpoint_created_at",
            "created_at",
            "finished_at",
            "started_at",
        ):
            if timestamp := _format_timestamp(section.get(key)):
                return timestamp
    return None


def _format_timestamp(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(float(value)).astimezone()
        except (OverflowError, OSError, ValueError):
            return None
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return _timestamp_in_text(text)
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _checkpoint_file_timestamp(
    row: Mapping[str, Any], source: str
) -> str | None:
    candidates = (row.get("path"), source)
    path = next(
        (
            Path(str(value)).expanduser()
            for value in candidates
            if value and not str(value).startswith("wandb:")
        ),
        None,
    )
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    created = float(getattr(stat, "st_birthtime", stat.st_mtime))
    return datetime.fromtimestamp(created).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _shorten_middle(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    left = (maximum - 1) // 2
    right = maximum - 1 - left
    return f"{value[:left]}…{value[-right:]}"
