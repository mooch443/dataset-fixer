from __future__ import annotations

import math
import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import altair as alt

from ..tabular import chart_data
from ..utils import shorten_middle as _shorten_middle

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


@dataclass(frozen=True)
class ModelPresentation:
    """Canonical display identity shared by every model-bearing figure.

    key:
        Stable model key used to associate plotted data with this identity.
    label:
        Length-limited model name/filename and optional second-line hash.
    badges:
        Coloured architecture, upscale, and resolution presentation slugs.
    """

    key: str
    label: str
    badges: tuple[ModelBadge, ...]

    @property
    def badge_pairs(self) -> tuple[tuple[str, str], ...]:
        """Return renderer-neutral ``(text, colour)`` badge values."""

        return tuple((badge.text, badge.color) for badge in self.badges)


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
    compact = f"{candidate} · {digest}" if digest else candidate
    identity = row.get("model_identity")
    if identity not in {"name", "both"}:
        return compact
    if identity == "name":
        return candidate
    return f"{candidate}\n{digest}" if digest else candidate


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


def model_presentation(model: Any, *, key: str | None = None) -> ModelPresentation:
    """Resolve one model once for axes, legends, and image-panel headings.

    model:
        Model or model metadata mapping.
    key:
        Optional plotted-data key; defaults to the model/name field.
    """

    row = _model_metadata(model)
    return ModelPresentation(
        key=str(key if key is not None else row.get("model") or row.get("name") or "model"),
        label=model_label(row),
        badges=model_badges(row),
    )


def model_full_label(model: Any) -> str:
    """Return the normalized name followed by its plain-text identity badges.

    model:
        Model or model metadata mapping.
    """

    values = (model_label(model), model_badge_text(model))
    return " · ".join(value for value in values if value)


def model_identity_chart(
    models: Sequence[Mapping[str, Any]],
    *,
    width: int = 430,
    row_height: int = 54,
) -> alt.TopLevelMixin:
    """Render comparison identities with their colored configuration badges."""

    identities, badges = [], []
    max_lines = 1
    for index, model in enumerate(models):
        presentation = model_presentation(model, key=str(index))
        label = presentation.label
        max_lines = max(max_lines, label.count("\n") + 1)
        identities.append({"row": index, "label": label})
        values = presentation.badges
        widths = [max(45, len(value.text) * 8 + 14) for value in values]
        cursor = width - sum(widths) - max(0, len(widths) - 1) * 6
        for value, badge_width in zip(values, widths):
            badges.append({
                "row": index, "text": value.text, "color": value.color,
                "x": cursor, "x2": cursor + badge_width,
                "center": cursor + badge_width / 2,
            })
            cursor += badge_width + 6
    row_height = max(row_height, 55 + 15 * max_lines)
    label_center = 10 + 7.5 * max_lines
    badge_top = 25 + 15 * max_lines
    for value in identities:
        value["y"] = value["row"] * row_height + label_center
    for value in badges:
        value["y"] = value["row"] * row_height + badge_top
        value["y2"] = value["y"] + 20
        value["middle"] = value["y"] + 10
    y_scale = alt.Scale(domain=[row_height * len(models), 0])
    labels = alt.Chart(chart_data(identities)).mark_text(
        align="right", baseline="middle", lineBreak="\n", lineHeight=15
    ).encode(
        x=alt.value(width - 4),
        y=alt.Y("y:Q", scale=y_scale, axis=None),
        text="label:N",
    )
    badge_data = alt.Chart(chart_data(badges))
    boxes = badge_data.mark_rect(cornerRadius=3).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0, width]), axis=None),
        x2="x2:Q",
        y=alt.Y("y:Q", scale=y_scale, axis=None),
        y2="y2:Q",
        color=alt.Color("color:N", scale=None, legend=None),
    )
    text = badge_data.mark_text(color="white", baseline="middle").encode(
        x=alt.X("center:Q", scale=alt.Scale(domain=[0, width]), axis=None),
        y=alt.Y("middle:Q", scale=y_scale, axis=None),
        text="text:N",
    )
    return (labels + boxes + text).properties(
        width=width, height=row_height * len(models)
    )


def model_identity_card(
    model: Any,
    *,
    width: int,
    maximum: int = 44,
    series_color: str | None = None,
) -> alt.TopLevelMixin:
    """Render a model identifier with its coloured slug row underneath."""

    presentation = model_presentation(model)
    lines = [
        wrapped
        for source_line in presentation.label.splitlines()
        for wrapped in (
            textwrap.wrap(
                source_line,
                width=max(8, maximum),
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [source_line]
        )
    ] or [presentation.label]
    text_rows = [
        {"text": value, "y": 8 + 15 * index}
        for index, value in enumerate(lines)
    ]
    label_height = max(18, 15 * len(lines) + 2)
    label = alt.Chart(chart_data(text_rows)).mark_text(
        align="center", baseline="middle", fontSize=11, fontWeight="bold"
    ).encode(
        x=alt.value(width / 2),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[label_height, 0]), axis=None),
        text="text:N",
    ).properties(width=width, height=label_height)
    if not presentation.badges and series_color is None:
        return label
    values = (
        ((ModelBadge("", series_color),) if series_color else ())
        + presentation.badges
    )
    badge_widths = [
        20 if not value.text else max(42, len(value.text) * 8 + 14)
        for value in values
    ]
    cursor = (width - sum(badge_widths) - 6 * (len(values) - 1)) / 2
    badge_rows = []
    for value, badge_width in zip(values, badge_widths):
        badge_rows.append({
            "text": value.text,
            "color": value.color,
            "x": cursor,
            "x2": cursor + badge_width,
            "center": cursor + badge_width / 2,
        })
        cursor += badge_width + 6
    data = alt.Chart(chart_data(badge_rows))
    boxes = data.mark_rect(cornerRadius=3).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0, width]), axis=None),
        x2="x2:Q",
        y=alt.value(1),
        y2=alt.value(21),
        color=alt.Color("color:N", scale=None, legend=None),
    )
    badge_text = data.mark_text(color="white", baseline="middle", fontSize=11).encode(
        x=alt.X("center:Q", scale=alt.Scale(domain=[0, width]), axis=None),
        y=alt.value(11),
        text="text:N",
    )
    return alt.vconcat(label, (boxes + badge_text).properties(width=width, height=22), spacing=4)


def with_model_identities(
    chart: alt.TopLevelMixin,
    models: Sequence[Mapping[str, Any]],
    *,
    columns: int = 2,
    width: int = 360,
    series_colors: Sequence[str] | None = None,
) -> alt.TopLevelMixin:
    """Place the canonical model identity cards above a model-bearing plot."""

    if not models:
        return chart
    cards = [
        model_identity_card(
            model,
            width=width,
            series_color=(
                series_colors[index % len(series_colors)]
                if series_colors
                else None
            ),
        )
        for index, model in enumerate(models)
    ]
    rows = [
        alt.hconcat(*cards[index : index + columns], spacing=16)
        for index in range(0, len(cards), columns)
    ]
    identities = alt.vconcat(*rows, spacing=8).properties(
        title=alt.TitleParams(text="Models", anchor="start", fontSize=13)
    )
    return alt.vconcat(identities, chart, spacing=14)


def model_identity_row_height(models: Sequence[Mapping[str, Any]]) -> int:
    """Return the shared plot-row height needed by identity and metric panels."""

    lines = max((model_label(row).count("\n") + 1 for row in models), default=1)
    return max(54, 55 + 15 * lines)


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
        "hash",
        "model_hash_short",
        "model_hash",
        "checkpoint_sha256_short",
        "model_sha256_short",
        "checkpoint_sha256",
        "model_sha256",
        "digest",
    ):
        value = str(row.get(key) or "").strip().lower()
        if len(value) >= 7 and value.isalnum():
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
            parsed = datetime.fromtimestamp(float(value), timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromtimestamp(float(text), timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except (OverflowError, OSError, ValueError):
        pass
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
