from __future__ import annotations

import inspect
from enum import Enum

import dataset_fixer
import dataset_fixer.bundle as bundle_api
import dataset_fixer.comparison as comparison_api
import dataset_fixer.convert as convert_api
import dataset_fixer.wandb as wandb_api
from dataset_fixer import (
    ComparisonResult,
    Dataset,
    DatasetComparisonResult,
    DatasetValidationError,
    Model,
    ModelCollection,
    PredictionResult,
    SemanticComparisonResult,
    Task,
)
from dataset_fixer.public_examples import SAWIT_COMMIT, SAWIT_LICENSE, SAWIT_REPOSITORY


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
        "move_images_with_classes",
        "move_n_groups",
        "remove_classes",
        "rename_classes",
        "rebalance_empty",
        "tile",
        "augment",
        "export",
        "export_formats",
        "report",
        "compare",
        "visualize",
        "sample",
        "filter",
        "add",
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
    assert inspect.getdoc(DatasetComparisonResult)
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
    assert hasattr(Model, "hash")
    assert hasattr(Model, "wandb")
    assert not hasattr(ModelCollection, "hashes")
    assert not hasattr(ModelCollection, "hash_for")
    assert inspect.getdoc(ModelCollection)


def _assert_parameters_documented(target: object, label: str) -> None:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return
    parameters = [
        name
        for name, parameter in signature.parameters.items()
        if name not in {"self", "cls"}
        and parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    ]
    if not parameters:
        return
    documentation = inspect.getdoc(target)
    assert documentation is not None, f"{label} has undocumented public parameters"
    missing = [name for name in parameters if f"{name}:" not in documentation]
    assert not missing, f"{label} does not document {missing}"


def test_every_exported_public_api_parameter_is_documented() -> None:
    modules = (dataset_fixer, bundle_api, convert_api, wandb_api)
    for module in modules:
        for name in module.__all__:
            exported = getattr(module, name)
            label = f"{module.__name__}.{name}"
            if inspect.isclass(exported):
                # Enum's generated ``*values`` constructor is an implementation
                # detail; its own public class methods are still audited below.
                if not issubclass(exported, Enum):
                    _assert_parameters_documented(exported, label)
                for member_name, raw_member in exported.__dict__.items():
                    if member_name.startswith("_"):
                        continue
                    if isinstance(raw_member, property):
                        member = raw_member.fget
                    elif isinstance(raw_member, (classmethod, staticmethod)):
                        member = raw_member.__func__
                    elif inspect.isfunction(raw_member):
                        member = raw_member
                    else:
                        continue
                    _assert_parameters_documented(member, f"{label}.{member_name}")
            elif inspect.isfunction(exported):
                _assert_parameters_documented(exported, label)


def test_public_example_source_is_pinned_and_explicitly_licensed() -> None:
    assert SAWIT_REPOSITORY == "https://github.com/dtnguyen0304/sawit"
    assert len(SAWIT_COMMIT) == 40
    assert SAWIT_LICENSE == "MIT"
