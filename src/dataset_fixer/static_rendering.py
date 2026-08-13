"""One static Altair renderer for image cards and declarative report charts."""

from __future__ import annotations

import base64
import io
import math
import textwrap
from pathlib import Path
from typing import Any, Literal, Sequence

import altair as alt
import numpy as np
from PIL import Image, ImageColor, ImageOps


LabelMode = Literal["middle", "wrap"]
SUPPORTED_STATIC_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".pdf", ".svg"})


def image_data_url(image: Image.Image | np.ndarray) -> str:
    """Encode an image for a self-contained Vega-Lite image mark."""

    resolved = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    buffer = io.BytesIO()
    resolved.convert("RGB").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def letterbox_image(
    image: Image.Image | np.ndarray,
    *,
    width: int,
    height: int,
    background: str = "#111318",
) -> Image.Image:
    """Fit an image into a fixed viewport without changing its aspect ratio."""

    source = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    source = ImageOps.exif_transpose(source).convert("RGB")
    fitted = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), ImageColor.getrgb(background))
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return canvas


def format_label(
    value: str,
    *,
    mode: LabelMode,
    maximum: int,
    wrap_width: int = 42,
    maximum_lines: int = 2,
) -> list[str]:
    """Lay out a bounded label without allowing text to enter image regions."""

    if mode not in {"middle", "wrap"}:
        raise ValueError("label_mode must be 'middle' or 'wrap'")
    source_lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    if not source_lines:
        return []
    if mode == "middle":
        shortened: list[str] = []
        for line in source_lines[:maximum_lines]:
            if len(line) <= maximum:
                shortened.append(line)
                continue
            left = (maximum - 1) // 2
            right = maximum - 1 - left
            shortened.append(f"{line[:left]}…{line[-right:]}")
        return shortened
    lines = [
        wrapped
        for source_line in source_lines
        for wrapped in textwrap.wrap(
            source_line,
            width=max(8, wrap_width),
            break_long_words=True,
            break_on_hyphens=False,
        )
    ]
    if len(lines) <= maximum_lines:
        return lines
    retained = lines[:maximum_lines]
    retained[-1] = textwrap.shorten(retained[-1] + " …", width=wrap_width, placeholder="…")
    return retained


def text_region(
    lines: Sequence[str],
    *,
    width: int,
    font_size: int = 12,
    color: str = "#252A34",
    height_per_line: int = 18,
    font_weight: str = "normal",
) -> alt.Chart:
    """Create a fixed text-only layout region with one record per line."""

    values = [
        {"line": line, "y": (index + 0.5) * height_per_line}
        for index, line in enumerate(lines)
    ]
    height = max(2, len(values) * height_per_line)
    if not values:
        values = [{"line": "", "y": 1}]
    return (
        alt.Chart(alt.Data(values=values))
        .mark_text(
            align="center",
            baseline="middle",
            color=color,
            fontSize=font_size,
            fontWeight=font_weight,
        )
        .encode(
            x=alt.value(width / 2),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[0, height], reverse=True), axis=None),
            text=alt.Text("line:N"),
        )
        .properties(width=width, height=height)
    )


def image_region(
    image: Image.Image | np.ndarray,
    *,
    width: int,
    height: int,
    background: str = "#111318",
) -> alt.Chart:
    """Create a fixed-size self-contained static image region."""

    boxed = letterbox_image(image, width=width, height=height, background=background)
    return (
        alt.Chart(alt.Data(values=[{"url": image_data_url(boxed)}]))
        .mark_image(width=width, height=height)
        .encode(
            x=alt.value(width / 2),
            y=alt.value(height / 2),
            url=alt.Url("url:N"),
        )
        .properties(width=width, height=height)
    )


def card(
    image: Image.Image | np.ndarray,
    *,
    width: int,
    height: int,
    heading: Sequence[str] = (),
    footer: Sequence[str] = (),
) -> alt.VConcatChart:
    """Stack heading, raster, and footer in non-overlapping regions."""

    regions: list[alt.Chart] = []
    if heading:
        regions.append(text_region(heading, width=width, font_size=12, font_weight="bold"))
    regions.append(image_region(image, width=width, height=height))
    if footer:
        regions.append(text_region(footer, width=width, font_size=11, color="#5B6572"))
    return alt.vconcat(*regions, spacing=4)


def save_chart(
    chart: alt.TopLevelMixin,
    destination: str | Path,
    *,
    scale_factor: float = 2.0,
    overwrite: bool = True,
) -> Path:
    """Save one Altair chart in the requested static format."""

    output = Path(destination).expanduser().resolve()
    suffix = output.suffix.lower()
    if suffix not in SUPPORTED_STATIC_SUFFIXES:
        raise ValueError("visualization destination must be PNG, JPEG, PDF, or SVG")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Visualization already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".jpg", ".jpeg"}:
        raster = Image.open(io.BytesIO(chart_png(chart, scale_factor=scale_factor))).convert("RGB")
        raster.save(output, quality=95)
    elif suffix == ".pdf":
        chart.save(output)
    else:
        chart.save(output, scale_factor=scale_factor)
    return output


def chart_png(chart: alt.TopLevelMixin, *, scale_factor: float = 1.5) -> bytes:
    buffer = io.BytesIO()
    chart.save(buffer, format="png", scale_factor=scale_factor)
    return buffer.getvalue()


def display_chart(chart: alt.TopLevelMixin, destination: Path | None = None) -> None:
    """Display a per-call static PNG without mutating Altair's renderer."""

    try:
        from IPython import get_ipython
        from IPython.display import Image as DisplayImage
        from IPython.display import display

        if get_ipython() is not None:
            display(DisplayImage(data=chart_png(chart)))
            return
    except Exception:
        pass
    if destination is not None:
        print(f"Visualization: {destination}")


def display_image(path: str | Path) -> None:
    """Display an existing raster directly, with a terminal fallback."""

    resolved = Path(path).expanduser().resolve()
    try:
        from IPython import get_ipython
        from IPython.display import Image as DisplayImage
        from IPython.display import display

        if get_ipython() is not None:
            display(DisplayImage(filename=str(resolved)))
            return
    except Exception:
        pass
    print(f"Visualization: {resolved}")


def finish_chart(
    chart: alt.TopLevelMixin,
    *,
    destination: Path | None,
    show: bool,
    overwrite: bool = False,
) -> None:
    """Apply the shared save/show policy; public visualization returns nothing."""

    if destination is not None:
        save_chart(chart, destination, overwrite=overwrite)
    if show:
        display_chart(chart, destination)


def concat_grid(
    charts: Sequence[alt.TopLevelMixin],
    *,
    columns: int,
    spacing: int = 18,
    title: str | None = None,
) -> alt.TopLevelMixin:
    """Build a rectangular grid while retaining incomplete final rows."""

    if columns < 1:
        raise ValueError("columns must be a positive integer")
    if not charts:
        raise ValueError("At least one chart is required")
    rows = [alt.hconcat(*charts[index : index + columns], spacing=spacing) for index in range(0, len(charts), columns)]
    result = alt.vconcat(*rows, spacing=spacing)
    if title:
        result = result.properties(
            title=alt.TitleParams(text=title, anchor="middle", fontSize=16, offset=14)
        )
    return result.configure_view(stroke=None).configure_axis(labelFontSize=11, titleFontSize=12)


def finite_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace non-finite floats because Vega-Lite JSON cannot represent them."""

    cleaned: list[dict[str, Any]] = []
    for row in rows:
        cleaned.append(
            {
                key: (
                    None
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                )
                for key, value in row.items()
            }
        )
    return cleaned
