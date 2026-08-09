from __future__ import annotations

import inspect
import json
from pathlib import Path

import dataset_fixer
import dataset_fixer.comparison as comparison_api
from dataset_fixer import (
    ComparisonResult,
    Dataset,
    DatasetValidationError,
    Model,
    ModelCollection,
    PredictionResult,
    SemanticComparisonResult,
    Task,
)
from dataset_fixer.public_examples import SAWIT_COMMIT, SAWIT_LICENSE, SAWIT_REPOSITORY


ROOT = Path(__file__).resolve().parents[1]


def test_public_api_inventory_is_typed_and_documented() -> None:
    expected = {
        "open",
        "name",
        "location",
        "data_yaml",
        "format",
        "manifest",
        "manifest_path",
        "image_dirs",
        "mask_dirs",
        "task",
        "splits",
        "classes",
        "warnings",
        "validation_audit",
        "settings",
        "history",
        "provenance",
        "training_ready",
        "split",
        "remove_classes",
        "rename_classes",
        "rebalance_empty",
        "tile",
        "augment",
        "export",
        "export_formats",
        "visualize",
        "assert_trainable",
        "trace",
        "update",
    }
    public = {name: value for name, value in Dataset.__dict__.items() if not name.startswith("_")}
    assert set(public) == expected
    for name, raw in public.items():
        if isinstance(raw, property):
            target = raw.fget
        elif isinstance(raw, classmethod):
            target = raw.__func__
        else:
            target = raw
        assert inspect.getdoc(target), f"Dataset.{name} has no public documentation"
        assert getattr(target, "__annotations__", {}).get("return") is not None
        if callable(target) and not isinstance(raw, property):
            assert "parameters:" in inspect.getdoc(target).lower()
            if name != "augment":
                assert all(
                    parameter.kind is not inspect.Parameter.VAR_KEYWORD
                    for parameter in inspect.signature(target).parameters.values()
                )

    tile_signature = str(inspect.signature(Dataset.tile))
    open_signature = str(inspect.signature(Dataset.open))
    comparison_signature = str(inspect.signature(ModelCollection.compare))
    assert "Literal['grid', 'coverage']" in tile_signature
    assert "Literal['all', 'none']" in tile_signature
    assert "Literal['raise', 'skip']" in tile_signature
    assert "background_filter" in tile_signature
    assert "**settings" not in tile_signature
    assert "Literal['raise', 'skip']" in open_signature
    assert "save_prediction_plots" in comparison_signature
    for removed in (
        "baseline",
        "paired_comparisons",
        "comparison_space",
        "inference",
        "confidence",
        "postprocess",
        "protocol",
        "resolution",
        "comparison_unit",
        "device",
        "sahi_",
    ):
        assert removed not in comparison_signature
    assert inspect.getdoc(ComparisonResult)
    assert inspect.getdoc(SemanticComparisonResult)
    assert "folds" not in comparison_signature
    assert "upscale_factor" not in comparison_signature
    assert not hasattr(Dataset, "compare_models")
    assert not hasattr(Dataset, "load_models")
    assert not hasattr(dataset_fixer, "SemanticModelCohort")
    assert not hasattr(comparison_api, "compare_models")
    assert inspect.getdoc(DatasetValidationError)
    assert inspect.getdoc(Task)
    assert inspect.getdoc(Model)
    assert inspect.getdoc(Model.predict)
    assert inspect.getdoc(Model.compare)
    assert inspect.getdoc(Model.load_many)
    assert tuple(inspect.signature(Model.load_many).parameters) == ("models",)
    assert not hasattr(Model, "model_sha256")
    assert inspect.getdoc(ModelCollection)


def test_public_model_api_documents_every_parameter() -> None:
    entries = (
        (Model, inspect.signature(Model), inspect.getdoc(Model)),
        (Model.predict, inspect.signature(Model.predict), inspect.getdoc(Model.predict)),
        (Model.compare, inspect.signature(Model.compare), inspect.getdoc(Model.compare)),
        (Model.visualize, inspect.signature(Model.visualize), inspect.getdoc(Model.visualize)),
        (Model.load_many, inspect.signature(Model.load_many), inspect.getdoc(Model.load_many)),
        (
            ModelCollection.predict,
            inspect.signature(ModelCollection.predict),
            inspect.getdoc(ModelCollection.predict),
        ),
        (
            ModelCollection.compare,
            inspect.signature(ModelCollection.compare),
            inspect.getdoc(ModelCollection.compare),
        ),
        (
            ModelCollection.configure,
            inspect.signature(ModelCollection.configure),
            inspect.getdoc(ModelCollection.configure),
        ),
        (
            ModelCollection.visualize,
            inspect.signature(ModelCollection.visualize),
            inspect.getdoc(ModelCollection.visualize),
        ),
    )
    for target, signature, doc in entries:
        assert doc is not None
        missing = [
            name
            for name in signature.parameters
            if name not in {"self", "cls"} and f"{name}:" not in doc
        ]
        assert not missing, f"{target} does not document {missing}"
    assert inspect.getdoc(ModelCollection.compare)
    assert inspect.getdoc(PredictionResult)


def test_public_example_source_is_pinned_and_explicitly_licensed() -> None:
    assert SAWIT_REPOSITORY == "https://github.com/dtnguyen0304/sawit"
    assert len(SAWIT_COMMIT) == 40
    assert SAWIT_LICENSE == "MIT"


def test_colab_notebooks_have_disclosure_license_and_clean_outputs() -> None:
    expected = {
        "01_controlled_splitting.ipynb": "Dataset.split",
        "02_task_aware_tiling.ipynb": "Dataset.tile",
        "03_fixed_cohort_model_comparison.ipynb": "models.compare",
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
