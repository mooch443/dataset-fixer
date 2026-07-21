from __future__ import annotations

import json
from pathlib import Path

from dataset_fixer import Dataset
from examples.create_example_datasets import create_example_datasets


ROOT = Path(__file__).resolve().parents[1]


def test_mit_example_datasets_are_valid_and_deterministic(tmp_path: Path) -> None:
    first = create_example_datasets(tmp_path / "examples", seed=42)
    original = (first["detection"] / "labels" / "val" / "row-00" / "row-00__frame-000.txt").read_bytes()
    second = create_example_datasets(tmp_path / "examples", seed=42)
    assert (second["detection"] / "labels" / "val" / "row-00" / "row-00__frame-000.txt").read_bytes() == original
    assert Dataset.open(first["raw_sequences"], task="detect", progress=False).splits == ("train",)
    assert Dataset.open(first["detection"], task="detect", progress=False).splits == ("train", "val")
    assert Dataset.open(first["polo"], task="polo", progress=False).splits == ("train", "val")
    assert (tmp_path / "examples" / "DATA_LICENSE.txt").is_file()


def test_colab_notebooks_have_disclosure_license_and_clean_outputs() -> None:
    expected = {
        "01_controlled_splitting.ipynb": "Dataset.split",
        "02_task_aware_tiling.ipynb": "Dataset.tile",
        "03_fixed_cohort_model_comparison.ipynb": "compare_models",
    }
    for filename, api in expected.items():
        path = ROOT / "notebooks" / filename
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert "colab" in notebook["metadata"]
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "colab.research.google.com/github/mooch443/dataset-fixer" in text
        assert "AI-generation disclosure" in text
        assert "MIT License" in text
        assert api in text
        assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
