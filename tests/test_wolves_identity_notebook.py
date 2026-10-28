from pathlib import Path
import json
import zipfile

import numpy as np
from openpyxl import Workbook
from IPython.core.inputtransformer2 import TransformerManager


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "wolves_identity_colab.ipynb"


def test_wolf_notebook_cells_compile_and_have_no_outputs():
    notebook = json.loads(NOTEBOOK.read_text())
    for number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = TransformerManager().transform_cell("".join(cell["source"]))
            compile(source, f"cell-{number}", "exec")
            assert cell["execution_count"] is None and cell["outputs"] == []


def test_notebook_extracts_wrapped_npzs_and_keeps_every_identity_known(tmp_path, monkeypatch):
    import pytest
    from IPython.core.interactiveshell import InteractiveShell

    pytest.importorskip("mmnpz")
    codes = ["".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"] if cell["cell_type"] == "code"]
    scratch = tmp_path/"scratch"
    scratch.mkdir()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Global ID Summary"
    clips = [f"2509_{number}_PM3_test" for number in (202,203,204)]
    sheet.append(["global_id", *clips])
    sheet.append(["wolf-a", *[f"{number}_ID03" for number in (202,203,204)]])
    sheet.append(["wolf-b", *[f"{number}_ID04" for number in (202,203,204)]])
    book = tmp_path/"identities.xlsx"
    workbook.save(book)
    archive = tmp_path/"crops.zip"
    rng = np.random.default_rng(19)
    with zipfile.ZipFile(archive, "w") as zipped:
        for clip in clips:
            path = tmp_path/f"{clip}.npz"
            np.savez(path, images=rng.integers(0,256,(12,16,16,3),dtype=np.uint8),
                     ids=np.repeat([3,4],6), frames=np.tile(np.arange(6),2),
                     frame_segment_indexes=np.repeat([0,1],6),tracklets=np.array([[0,5],[0,5]]))
            zipped.write(path, f"enclosing-folder/{clip}/images.npz")
    context = dict(Path=Path, DATA_SOURCE=archive, SCRATCH=scratch, IDENTITIES=book,
                   LOCAL_DATA=scratch/"trex_identity_data", REMOTE_URL=tmp_path/"saved",
                   TRIM_BLACK_BORDERS=True, display=lambda value: None)
    shell = InteractiveShell(user_ns=context)
    extraction = next(code for code in codes if code.startswith("# Colab"))
    extraction = "\n".join(line for line in extraction.splitlines() if line.startswith("!"))
    shell.run_cell(extraction).raise_error()
    # Restore environment/cwd/tempfile state after executing the shared setup cell.
    import os
    import tempfile
    monkeypatch.chdir(tmp_path)
    for name in ("HF_HOME", "TORCH_HOME", "DATASET_FIXER_CACHE_DIR", "WANDB_PROJECT"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    load = next(code for code in codes if code.startswith("import os\nimport shutil"))
    exec(load, context)
    assert len(context["data"]._samples)==36
    split = next(code for code in codes if code.startswith("labeled ="))
    exec(split, context)
    protocol = context["protocol"]
    assert set(protocol.training.classes.values())=={"wolf-a","wolf-b"}
    assert all(len(protocol.queries[split]._samples)==12 for split in ("val","test"))
    # unzip -n and loading can be rerun without overwriting existing images.
    shell.run_cell(extraction).raise_error()
    exec(load, context)
    assert len(context["data"]._samples)==36
