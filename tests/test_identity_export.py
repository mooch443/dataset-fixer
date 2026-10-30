import json
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from dataset_fixer import Dataset
from dataset_fixer.artifacts import dataset_info_path
from dataset_fixer.id.data import ImageRef, SampleIndex, as_dataset
from dataset_fixer.id.model import input_index


def identity_data(path, **selectors):
    return as_dataset(SampleIndex([dict(
        path=str(path), width=40, height=30, identity_id="wolf", clip_id="clip",
        tracklet_id="track", split="val", notes="export", **selectors,
    )]), location=path.parent)


@pytest.mark.parametrize("image_format", [None, "jxl", "png"])
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA", "P"])
def test_lossless_portable_export(tmp_path, image_format, mode):
    source = tmp_path / "source" / "frame.png"
    source.parent.mkdir()
    rng = np.random.default_rng(12)
    shape = (30, 40) if mode in {"L", "P"} else (30, 40, len(mode))
    pixels = rng.integers(1, 256, shape, dtype=np.uint8)
    if mode in {"RGBA", "LA"}:
        pixels[8:12, 13:17, -1] = 0  # Hidden color must also survive losslessly.
    image = Image.fromarray(pixels)
    if mode == "P":
        image = image.convert("P")
        image.info["transparency"] = int(pixels[9, 14])
    image.save(source)
    data = identity_data(source, crop=(10, 5, 30, 25), trim_black_borders=True)
    expected = data._samples[0].image_ref.read(preserve_alpha=True)
    training = data._samples[0].image_ref.read()
    exported = data.export(destination=tmp_path / "portable", format="identity_manifest",
                           copy_images=True, image_format=image_format)
    record = json.loads((exported.location / "samples.jsonl").read_text())
    extension = image_format or "jxl"
    assert record["path"] == f"images/00000000.{extension}"
    assert record["array"] is record["row"] is record["crop"] is None
    assert record["trim_black_borders"] is False
    assert (record["width"], record["height"]) == (20, 20)
    assert record["original_id"] == data._samples[0].provenance["original_id"]
    assert exported.manifest["image_format"] == extension
    shutil.rmtree(source.parent)
    shutil.move(exported.location, tmp_path / "moved")
    loaded = Dataset.open(tmp_path / "moved/samples.jsonl", format="identity_manifest")
    sample = loaded._samples[0]
    with Image.open(sample.image_path) as decoded:
        np.testing.assert_array_equal(np.array(decoded), expected)
    np.testing.assert_array_equal(sample.image_ref.read(), training)
    assert (sample.identity_id, sample.clip_id, sample.split) == ("wolf", "clip", "val")
    assert sample.provenance["tracklet_id"] == data._samples[0].provenance["tracklet_id"]
    assert sample.provenance["identity_notes"] == "export"


def test_jxl_fresh_process_and_folder_discovery(tmp_path):
    source = tmp_path / "source" / "frame.png"
    source.parent.mkdir()
    Image.new("RGB", (40, 30), (23, 67, 123)).save(source)
    exported = identity_data(source).export(destination=tmp_path / "portable",
                                           format="identity_manifest", copy_images=True)
    script = """
import sys
from dataset_fixer import Dataset
assert 'pillow_jxl' not in sys.modules
data = Dataset.open(sys.argv[1], format='identity_manifest')
pixels = data._samples[0].image_ref.read()
assert (pixels == [23, 67, 123]).all()
"""
    subprocess.run([sys.executable, "-c", script, str(exported.location / "samples.jsonl")],
                   check=True, capture_output=True, text=True)
    folder = tmp_path / "folders" / "wolf"
    folder.mkdir(parents=True)
    shutil.copy(exported._samples[0].image_path, folder / "frame.JXL")
    loaded = Dataset.open(folder.parent, format="identity_folders", layout="{identity_id}/{image}")
    discovered = input_index(folder)
    assert len(loaded._samples) == len(discovered) == 1
    np.testing.assert_array_equal(loaded._samples[0].image_ref.read(), discovered[0].image_ref.read())


def test_jxl_npz_export_and_alpha_trim_alignment(tmp_path):
    pytest.importorskip("mmnpz")
    pixels = np.random.default_rng(2).integers(1, 256, (12, 18, 3), dtype=np.uint8)
    padded = np.pad(pixels, ((2, 2), (3, 3), (0, 0)))
    source = tmp_path / "source" / "frames.npz"
    source.parent.mkdir()
    np.savez(source, images=padded[None])
    exported = identity_data(source, array="images", row=0, trim_black_borders=True).export(
        destination=tmp_path / "portable", format="identity_manifest", copy_images=True)
    source.unlink()
    np.testing.assert_array_equal(exported._samples[0].image_ref.read(), pixels)
    assert (exported._samples[0].width, exported._samples[0].height) == (18, 12)

    rgba = np.dstack([padded, np.full(padded.shape[:2], 255, dtype=np.uint8)])
    path = tmp_path / "alpha.png"
    Image.fromarray(rgba).save(path)
    ref = ImageRef(str(path), trim_black_borders=True)
    np.testing.assert_array_equal(ref.read(), pixels)
    np.testing.assert_array_equal(ref.read(preserve_alpha=True)[..., :3], pixels)


def test_missing_jxl_codec_fails_before_publishing_and_png_still_works(tmp_path, monkeypatch):
    path = tmp_path / "source" / "frame.png"
    path.parent.mkdir()
    Image.new("RGB", (40, 30), "red").save(path)
    data = identity_data(path)
    exported = data.export(destination=tmp_path / "jxl", format="identity_manifest", copy_images=True)
    monkeypatch.setitem(sys.modules, "pillow_jxl", None)
    with pytest.raises(ImportError, match="pip install pillow-jxl-plugin"):
        data.export(destination=tmp_path / "missing", format="identity_manifest", copy_images=True)
    assert not (tmp_path / "missing").exists()
    with pytest.raises(ImportError, match="pip install pillow-jxl-plugin"):
        Dataset.open(exported.location / "samples.jsonl", format="identity_manifest")
    png = data.export(destination=tmp_path / "png", format="identity_manifest",
                      copy_images=True, image_format="png")
    assert png._samples[0].image_path.suffix == ".png"
    refs = data.export(destination=tmp_path / "references", format="identity_manifest")
    assert refs._samples[0].image_path == path


def test_legacy_png_package_without_format_metadata(tmp_path, monkeypatch):
    # Reproduce the old on-disk layout, without using the new exporter.
    folder = tmp_path / "legacy"
    image = folder / "images" / "00000000.png"
    image.parent.mkdir(parents=True)
    pixels = np.random.default_rng(7).integers(0, 256, (12, 18, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(image)
    (folder / "samples.jsonl").write_text(json.dumps(dict(
        path="images/00000000.png", width=18, height=12, array=None, row=None,
        crop=None, trim_black_borders=False, identity_id="wolf", clip_id="clip",
        split="train", original_id="original-crop",
    )) + "\n")
    metadata = dataset_info_path(folder)
    metadata.parent.mkdir()
    metadata.write_text(json.dumps(dict(name="legacy", format="identity_manifest",
                                        task="id", copy_images=True)))
    shutil.move(folder, tmp_path / "moved")
    monkeypatch.setitem(sys.modules, "pillow_jxl", None)
    loaded = Dataset.open(tmp_path / "moved/samples.jsonl", format="identity_manifest")
    assert "image_format" not in loaded.manifest
    assert loaded._samples[0].identity_id == "wolf"
    assert loaded._samples[0].provenance["original_id"] == "original-crop"
    np.testing.assert_array_equal(loaded._samples[0].image_ref.read(), pixels)


@pytest.mark.parametrize("kwargs, message", [
    ({"copy_images": False, "image_format": "jxl"}, "copy_images=True"),
    ({"copy_images": True, "image_format": "jpg"}, "'png' or 'jxl'"),
    ({"copy_images": True, "image_format": ""}, "'png' or 'jxl'"),
])
def test_invalid_format_does_not_create_output(tmp_path, kwargs, message):
    path = tmp_path / "frame.png"
    Image.new("RGB", (40, 30), "red").save(path)
    for dry_run in (False, True):
        with pytest.raises(ValueError, match=message):
            identity_data(path).export(destination=tmp_path / "output", format="identity_manifest",
                                       dry_run=dry_run, **kwargs)
    assert not (tmp_path / "output").exists()
