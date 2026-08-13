from __future__ import annotations

import ast
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "dataset_fixer"
CANONICAL_DEFINITIONS = {
    "slugify": PACKAGE_ROOT / "utils.py",
    "bounded_slug": PACKAGE_ROOT / "utils.py",
    "model_label": PACKAGE_ROOT / "comparison" / "plot_labels.py",
    "model_badges": PACKAGE_ROOT / "comparison" / "plot_labels.py",
    "model_badge_text": PACKAGE_ROOT / "comparison" / "plot_labels.py",
    "model_full_label": PACKAGE_ROOT / "comparison" / "plot_labels.py",
}
FORBIDDEN_ALIASES = re.compile(
    r"^_?(?:safe_stem|short_model_name|model_plot_label|model_plot_badges|"
    r"ranking_plot_label)$"
)
ALLOWED_SLUG_DEFINITIONS = {
    ("slugify", PACKAGE_ROOT / "utils.py"),
    ("bounded_slug", PACKAGE_ROOT / "utils.py"),
    ("slug", PACKAGE_ROOT / "model.py"),
}


def _definitions(source: str, filename: str) -> list[tuple[str, int]]:
    tree = ast.parse(source, filename=filename)
    return [
        (node.name, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def test_model_identity_and_slug_helpers_have_one_canonical_definition() -> None:
    problems: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name, line in _definitions(source, str(path)):
            canonical = CANONICAL_DEFINITIONS.get(name)
            if canonical is not None and path != canonical:
                problems.append(f"{path}:{line}: duplicate {name}()")
            if "slug" in name.lower() and (name, path) not in ALLOWED_SLUG_DEFINITIONS:
                problems.append(
                    f"{path}:{line}: {name}() bypasses utils.slugify/bounded_slug"
                )
            if FORBIDDEN_ALIASES.fullmatch(name):
                problems.append(
                    f"{path}:{line}: {name}() bypasses the public identity API"
                )
    assert not problems, "\n".join(problems)


def test_notebooks_do_not_reimplement_model_identity_or_slug_helpers() -> None:
    problems: list[str] = []
    for path in (PROJECT_ROOT / "notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            try:
                definitions = _definitions(source, f"{path}:cell-{cell_index}")
            except SyntaxError:
                # Notebook magics are not Python AST; the textual duplicate-name
                # check below still catches presentation-helper reimplementations.
                definitions = []
            for name, line in definitions:
                if (
                    name in CANONICAL_DEFINITIONS
                    or "slug" in name.lower()
                    or FORBIDDEN_ALIASES.fullmatch(name)
                ):
                    problems.append(
                        f"{path}:cell-{cell_index}:{line}: duplicate {name}()"
                    )
            for name in (*CANONICAL_DEFINITIONS, "short_model_name", "safe_stem"):
                if re.search(rf"\bdef\s+{re.escape(name)}\s*\(", source):
                    location = f"{path}:cell-{cell_index}"
                    if not any(item.startswith(location) and name in item for item in problems):
                        problems.append(f"{location}: duplicate {name}()")
    assert not problems, "\n".join(problems)
