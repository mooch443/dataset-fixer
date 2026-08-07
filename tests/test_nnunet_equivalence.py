"""Numerical equivalence between batched and sequential nnU-Net SAHI inference.

These tests need a real trained nnU-Net model folder and a semantic-mask
cohort, so they are skipped unless ``DATASET_FIXER_NNUNET_MODEL`` and
``DATASET_FIXER_NNUNET_IMAGES`` point at them:

    DATASET_FIXER_NNUNET_MODEL=/path/to/nnUNetTrainer__plans__2d \\
    DATASET_FIXER_NNUNET_IMAGES=/path/to/val/images \\
    DATASET_FIXER_NNUNET_CHECKPOINT=checkpoint_best.pth \\
    pytest tests/test_nnunet_equivalence.py

The reference implementation below is the engine this replaced: one official
``nnUNetv2_predict_from_modelfolder`` call whose per-tile NPZ probabilities are
read back from disk.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset_fixer.model import Model, ModelInput
from dataset_fixer.sahi_support import (
    build_tile_manifest,
    resolve_sahi_settings,
    stitch_probability_tiles,
)
from dataset_fixer.semantic_comparison import _predict_nnunet_sahi


MODEL_FOLDER = os.environ.get("DATASET_FIXER_NNUNET_MODEL")
IMAGE_FOLDER = os.environ.get("DATASET_FIXER_NNUNET_IMAGES")
CHECKPOINT = os.environ.get("DATASET_FIXER_NNUNET_CHECKPOINT", "checkpoint_final.pth")
UPSCALE = int(os.environ.get("DATASET_FIXER_NNUNET_UPSCALE", "2"))
DEVICE = os.environ.get("DATASET_FIXER_NNUNET_DEVICE", "cpu")
IMAGES = int(os.environ.get("DATASET_FIXER_NNUNET_COHORT", "3"))

SETTINGS = {"sahi_slice_height": 128, "sahi_slice_width": 128, "sahi_overlap": 0.15}

pytestmark = [
    pytest.mark.skipif(
        not MODEL_FOLDER or not IMAGE_FOLDER,
        reason="set DATASET_FIXER_NNUNET_MODEL and DATASET_FIXER_NNUNET_IMAGES",
    ),
    pytest.mark.skipif(
        shutil.which("nnUNetv2_predict_from_modelfolder") is None,
        reason="the official nnU-Net prediction CLI is required as the reference",
    ),
]


@pytest.fixture(scope="module")
def cohort() -> tuple[ModelInput, ...]:
    paths = sorted(Path(IMAGE_FOLDER).glob("*.png"))[:IMAGES]
    assert paths, f"no images found in {IMAGE_FOLDER}"
    inputs = []
    for index, path in enumerate(paths):
        with Image.open(path) as opened:
            width, height = opened.size
        inputs.append(
            ModelInput(
                image_id=f"case_{index:04d}",
                image_path=path,
                width=width,
                height=height,
                relative_path=path.name,
            )
        )
    return tuple(inputs)


@pytest.fixture(scope="module")
def model() -> Model:
    return Model(
        MODEL_FOLDER,
        name="equivalence",
        upscale_factor=UPSCALE,
        checkpoint=CHECKPOINT,
        inference="sahi",
        device=DEVICE,
        workers=2,
        settings=dict(SETTINGS),
    )


@pytest.fixture(scope="module")
def reference(model: Model, cohort: tuple[ModelInput, ...]) -> dict[str, dict]:
    return _sequential_reference(model, cohort)


@pytest.fixture(scope="module")
def batched(model: Model, cohort: tuple[ModelInput, ...]):
    return _predict_nnunet_sahi(
        model,
        cohort,
        device=DEVICE,
        progress=False,
        keep_native=True,
        resolution=480,
        settings=dict(SETTINGS),
    )


def test_batched_masks_are_identical_to_sequential_masks(batched, reference) -> None:
    for record in batched:
        expected = reference[record.image_id]
        assert np.array_equal(record.mask, expected["mask"]), (
            f"{record.image_id}: "
            f"{int(np.count_nonzero(record.mask != expected['mask']))} differing pixels"
        )


def test_batched_native_masks_are_identical_to_sequential_native_masks(
    batched,
    reference,
) -> None:
    for record in batched:
        expected = reference[record.image_id]
        assert np.array_equal(record.native_mask, expected["native_mask"])


def test_batched_probabilities_are_numerically_equivalent(
    model: Model,
    cohort: tuple[ModelInput, ...],
    reference,
) -> None:
    for image_id, probabilities in _batched_probabilities(model, cohort).items():
        expected = reference[image_id]["probabilities"]
        assert probabilities.shape == expected.shape
        assert np.allclose(probabilities, expected, rtol=1e-4, atol=1e-5), (
            f"{image_id}: max deviation "
            f"{float(np.max(np.abs(probabilities - expected)))}"
        )


def _batched_probabilities(
    model: Model,
    inputs: tuple[ModelInput, ...],
) -> dict[str, np.ndarray]:
    """Stitch canonical probabilities through the in-process engine."""

    from dataset_fixer.nnunet_engine import load_session
    from dataset_fixer.semantic_comparison import (
        _equal_shape_batches,
        _slice_source_tiles,
    )

    resolved = resolve_sahi_settings(dict(SETTINGS), resolution=480)
    session = load_session(
        model_folder=model.model_folder,
        folds=model.folds,
        checkpoint=model.checkpoint,
        device=DEVICE,
        workers=model.workers,
    )
    output: dict[str, np.ndarray] = {}
    try:
        for value in inputs:
            manifest = build_tile_manifest(
                width=value.width, height=value.height, settings=resolved
            )
            prepared = session.preprocess_many(
                _slice_source_tiles(value, manifest, upscale_factor=model.upscale_factor)
            )
            logits: list = [None] * len(prepared)
            for indices in _equal_shape_batches(
                [array for array, _ in prepared],
                classes=session.num_classes,
                minimum=session.resolved_batch_size,
            ):
                for index, value_logits in zip(
                    indices,
                    session.predict_logits([prepared[index][0] for index in indices]),
                ):
                    logits[index] = value_logits
            probabilities = session.to_probabilities_many(
                [(logits[index], prepared[index][1]) for index in range(len(prepared))]
            )
            native = stitch_probability_tiles(
                width=value.width,
                height=value.height,
                tiles=[
                    (
                        tile,
                        probabilities[index].reshape(
                            2,
                            tile.height * model.upscale_factor,
                            tile.width * model.upscale_factor,
                        ),
                    )
                    for index, tile in enumerate(manifest)
                ],
                scale=model.upscale_factor,
            )
            if model.upscale_factor == 1:
                output[value.image_id] = native
            else:
                output[value.image_id] = native.reshape(
                    2,
                    value.height,
                    model.upscale_factor,
                    value.width,
                    model.upscale_factor,
                ).mean(axis=(2, 4))
    finally:
        session.release()
    return output


def test_the_batched_engine_loads_each_fold_once_for_the_whole_run(batched) -> None:
    metadata = batched[0].metadata
    assert metadata["nnunet_execution_engine"] == "in-process-minibatched"
    assert metadata["nnunet_weight_loads"] == len(metadata["nnunet_folds"])
    # Real minibatches mean far fewer forward passes than tiles.
    assert metadata["nnunet_forward_passes"] < metadata["nnunet_tiles"]
    assert metadata["nnunet_resolved_batch_size"] >= 1


def _sequential_reference(
    model: Model,
    inputs: tuple[ModelInput, ...],
) -> dict[str, dict]:
    resolved = resolve_sahi_settings(dict(SETTINGS), resolution=480)
    manifests = {
        value.image_id: build_tile_manifest(
            width=value.width, height=value.height, settings=resolved
        )
        for value in inputs
    }
    records: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="dataset-fixer-sahi-reference-") as temporary:
        root = Path(temporary)
        image_dir = root / "images"
        prediction_dir = root / "predictions"
        image_dir.mkdir(parents=True)
        prediction_dir.mkdir(parents=True)
        case_ids: dict[tuple[str, int], str] = {}
        for value in inputs:
            with Image.open(value.image_path) as opened:
                source_image = opened.convert("RGB")
            for tile in manifests[value.image_id]:
                case_id = f"{value.image_id}__tile_{tile.index:06d}"
                case_ids[(value.image_id, tile.index)] = case_id
                image = source_image.crop(tile.box)
                if model.upscale_factor != 1:
                    image = image.resize(
                        (
                            tile.width * model.upscale_factor,
                            tile.height * model.upscale_factor,
                        ),
                        Image.Resampling.BICUBIC,
                    )
                image.save(image_dir / f"{case_id}_0000.png", format="PNG")
        subprocess.run(
            [
                "nnUNetv2_predict_from_modelfolder",
                "-i", str(image_dir),
                "-o", str(prediction_dir),
                "-m", str(model.model_folder),
                "-f", *model.folds,
                "-chk", model.checkpoint,
                "-device", DEVICE,
                "-npp", str(model.workers),
                "-nps", str(model.workers),
                "--save_probabilities",
            ],
            check=True,
            capture_output=True,
        )
        for value in inputs:
            tiles = []
            for tile in manifests[value.image_id]:
                path = prediction_dir / f"{case_ids[(value.image_id, tile.index)]}.npz"
                with np.load(path) as archive:
                    probabilities = np.asarray(archive["probabilities"], dtype=np.float32)
                expected = (
                    tile.height * model.upscale_factor,
                    tile.width * model.upscale_factor,
                )
                tiles.append((tile, probabilities.reshape(2, *expected)))
            native = stitch_probability_tiles(
                width=value.width,
                height=value.height,
                tiles=tiles,
                scale=model.upscale_factor,
            )
            if model.upscale_factor == 1:
                canonical = native
            else:
                canonical = native.reshape(
                    2,
                    value.height,
                    model.upscale_factor,
                    value.width,
                    model.upscale_factor,
                ).mean(axis=(2, 4))
            records[value.image_id] = {
                "mask": np.argmax(canonical, axis=0).astype(np.uint8),
                "native_mask": np.argmax(native, axis=0).astype(np.uint8),
                "probabilities": canonical,
            }
    return records
