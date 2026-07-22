from __future__ import annotations

import json
from pathlib import Path

from dataset_fixer.public_examples import SAWIT_COMMIT, SAWIT_LICENSE, SAWIT_REPOSITORY


ROOT = Path(__file__).resolve().parents[1]


def test_public_example_source_is_pinned_and_explicitly_licensed() -> None:
    assert SAWIT_REPOSITORY == "https://github.com/dtnguyen0304/sawit"
    assert len(SAWIT_COMMIT) == 40
    assert SAWIT_LICENSE == "MIT"


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
        assert "dtnguyen0304/sawit" in text
        assert "git+https://github.com/mooch443/dataset-fixer.git" in text
        assert "git', 'clone" not in text
        assert "subprocess" not in text
        assert "examples.download_public_examples" not in text
        assert "dataset_fixer.public_examples" in text
        assert "import dataset_fixer" in text
        assert api in text
        assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
