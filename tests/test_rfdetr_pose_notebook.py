"""The notebook contains orchestration only; behavior is tested in test_training."""
import ast
import json
from pathlib import Path

from IPython.core.inputtransformer2 import TransformerManager


def test_minimal_notebook_cells_compile_without_embedded_utilities():
    path = Path(__file__).parents[1] / "notebooks/rfdetr_pose_train_dataset_fixer.ipynb"
    notebook = json.loads(path.read_text())
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    for cell in cells.values():
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(TransformerManager().transform_cell(source))
        assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in ast.walk(tree))
        assert cell["outputs"] == []
    assert '!wandb login' in ''.join(cells['wandb-login']['source'])
    assert 'drive.mount("/content/drive")' in ''.join(cells['drive']['source'])
    assert 'AUG_CONFIG' not in ''.join(cells['settings']['source'])
    workflow = ''.join(cells['train-evaluate-upload']['source'])
    assert 'disconnect="after_safe"' in workflow
    assert 'result.evaluate(samples=32, plots=6)' in workflow
    assert sum(len(c['source']) for c in cells.values() if c['cell_type'] == 'code') <= 60
