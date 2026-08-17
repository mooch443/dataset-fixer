from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

from dataset_fixer.comparison.plot_labels import model_identity_card
from dataset_fixer.static_rendering import (
    format_label,
    letterbox_image,
    save_chart,
)
from dataset_fixer.visualization import (
    VisualizationItem,
    VisualizationOptions,
    VisualizationPanel,
    draw_mask_outline,
    visualize_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "dataset_fixer"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_framework_has_no_matplotlib_import_or_direct_dependency() -> None:
    offenders = {
        str(path.relative_to(PROJECT_ROOT)): sorted(
            name for name in _imports(path) if name == "matplotlib" or name.startswith("matplotlib.")
        )
        for path in PACKAGE_ROOT.rglob("*.py")
        if any(name == "matplotlib" or name.startswith("matplotlib.") for name in _imports(path))
    }
    assert offenders == {}
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    assert not any(str(value).lower().startswith("matplotlib") for value in dependencies)
    assert "altair[save]>=6.2,<7" in dependencies


def test_labels_support_middle_shortening_and_bounded_wrapping() -> None:
    value = "prefix/" + "very-long-segment-" * 12 + "/suffix.png"
    shortened = format_label(value, mode="middle", maximum=45)
    wrapped = format_label(value, mode="wrap", maximum=45, wrap_width=24, maximum_lines=2)

    assert len(shortened) == 1 and len(shortened[0]) == 45
    assert shortened[0].startswith("prefix/") and shortened[0].endswith("suffix.png")
    assert "…" in shortened[0]
    assert 1 <= len(wrapped) <= 2
    assert all(len(line) <= 24 for line in wrapped)


def test_letterbox_preserves_aspect_and_mask_outline_handles_empty_masks() -> None:
    wide = Image.new("RGB", (240, 40), "#335577")
    boxed = letterbox_image(wide, width=160, height=160, background="#101010")
    assert boxed.size == (160, 160)
    assert boxed.getpixel((80, 80)) == (51, 85, 119)
    assert boxed.getpixel((80, 5)) == (16, 16, 16)

    source = np.asarray(wide)
    empty = np.zeros((40, 240), dtype=bool)
    assert np.array_equal(
        draw_mask_outline(source, empty, color="#ff0000", line_width=1, outline_width=2, alpha=1),
        source,
    )
    mask = empty.copy()
    mask[10:30, 80:160] = True
    outlined = Image.fromarray(
        draw_mask_outline(source, mask, color="#ff0000", line_width=2, outline_width=4, alpha=1)
    )
    assert ImageChops.difference(wide, outlined).getbbox() is not None


def test_visualization_grid_is_deterministic_and_keeps_incomplete_rows() -> None:
    records = list(range(5))

    def prepare(index: int) -> VisualizationItem:
        image = np.full((40 + index * 3, 90 + index * 7, 3), 30 + index * 20, dtype=np.uint8)
        foreground = np.zeros(image.shape[:2], dtype=bool)
        if index % 2:
            foreground[10:20, 20:35] = True
        return VisualizationItem(
            image_path=Path(f"/tmp/{index}-{'long-' * 20}image.png"),
            label=f"item-{index}\nmetadata-{index}",
            panels=(VisualizationPanel(title="Annotation", image=image),),
            foreground=foreground,
        )

    options = VisualizationOptions(
        samples=4,
        columns=3,
        seed=7,
        panel_size=1.5,
        zoom=True,
        label_mode="wrap",
        show=False,
    )
    first = visualize_records(records, options=options, prepare=prepare, title="Mixed aspects")
    second = visualize_records(records, options=options, prepare=prepare, title="Mixed aspects")
    first_spec = first.to_dict()
    assert first_spec == second.to_dict()
    assert len(first_spec["vconcat"]) == 2
    assert len(first_spec["vconcat"][0]["hconcat"]) == 3
    assert len(first_spec["vconcat"][1]["hconcat"]) == 1


def test_mixed_panel_headings_reserve_one_exact_shared_height() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    heading = model_identity_card(
        {
            "model": "yolo26m-seg-1024px-2x",
            "hash": "abcdef12",
            "source_created_at": "2026-08-13T15:42:28Z",
            "model_identity": "both",
            "model_type": "yolo26m-seg",
            "upscale_factor": 2,
            "resolution": 1024,
        },
        width=144,
    )
    item = VisualizationItem(
        image_path=Path("/tmp/aligned.png"),
        label="",
        panels=(
            VisualizationPanel(title="Original", image=image),
            VisualizationPanel(title="GT", image=image),
            VisualizationPanel(title="Prediction", image=image, heading=heading),
        ),
        foreground=np.zeros((32, 32), dtype=bool),
    )
    chart = visualize_records(
        [item],
        options=VisualizationOptions(
            samples=None, columns=1, panel_size=1.5, show=False
        ),
        prepare=lambda value: value,
    )
    panels = chart.to_dict()["vconcat"][0]["hconcat"][0]["vconcat"][0][
        "hconcat"
    ]

    def fixed_height(specification: dict) -> int:
        if isinstance(specification.get("height"), int):
            return specification["height"]
        children = specification["vconcat"]
        return sum(fixed_height(child) for child in children) + int(
            specification.get("spacing", 10)
        ) * (len(children) - 1)

    assert [fixed_height(panel["vconcat"][0]) for panel in panels] == [58, 58, 58]
    assert heading.to_dict()["spacing"] == 4


def test_static_renderer_exports_png_jpeg_pdf_and_svg(tmp_path: Path) -> None:
    image = np.zeros((45, 120, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    item = VisualizationItem(
        image_path=tmp_path / "source.png",
        label="source",
        panels=(VisualizationPanel(title="Original", image=image),),
        foreground=np.zeros((45, 120), dtype=bool),
    )
    chart = visualize_records(
        [item],
        options=VisualizationOptions(samples=None, columns=1, panel_size=1.5, show=False),
        prepare=lambda value: value,
        title="Export formats",
    )

    for suffix in ("png", "jpg", "pdf", "svg"):
        path = save_chart(chart, tmp_path / f"chart.{suffix}")
        assert path.is_file() and path.stat().st_size > 100
    with Image.open(tmp_path / "chart.png") as png, Image.open(tmp_path / "chart.jpg") as jpeg:
        assert png.width > 0 and png.height > 0
        assert jpeg.mode == "RGB"
