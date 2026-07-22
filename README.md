# dataset-fixer

[![CI](https://github.com/mooch443/dataset-fixer/actions/workflows/ci.yml/badge.svg)](https://github.com/mooch443/dataset-fixer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AI-generation disclosure:** This project is largely AI-generated. Its
> architecture, implementation, tests, and documentation were produced with
> substantial assistance from OpenAI's Codex under human direction and review.
> Users should independently validate behavior for their datasets and research
> workflows.

`dataset-fixer` loads YOLO or COCO datasets, validates them immediately, and
writes safe, reproducible YOLO derivatives for detection, segmentation, pose,
and POLO point-localization tasks.

```python
from dataset_fixer import Dataset

dataset = Dataset.open("/datasets/orchard", task="polo")
split = dataset.split({"train": 0.75, "val": 0.25}, seed=42)
clean = split.remove_classes(["damaged"])
tiled = clean.tile(mode="coverage", tile_size=480)
exported = tiled.export(destination="/datasets/orchard_ready")

print(tiled)               # virtual pipeline; nothing has been written
print(exported.location)
print(exported.data_yaml)
```

`split()`, `remove_classes()`, `rebalance_empty()`, `augment()`, and `tile()` are immutable
in-memory planning operations. Only `export()` writes files. Export validates
before atomic publication, records every effective setting and tool/environment
version, and maps each output image back to its parent and ultimate original in
`provenance.jsonl`.

## Loading and validation

`Dataset.open()` accepts a dataset root, a YOLO `data.yaml`, or a COCO JSON/root.
Pass `task=` when the annotation geometry is ambiguous. POLO labels use
`class radius x y`; their YAML must also define one positive class-level radius
per class. Use `deep=True` to add SHA-256 duplicate detection across splits.

```python
dataset = Dataset.open("/datasets/orchard/data.yaml", task="pose", deep=True)
dataset.assert_trainable()  # also checks Ultralytics when it is installed
dataset.visualize(split="train", n=12, seed=42, columns=3)
```

Validation errors identify the file, label row or COCO annotation, bad value,
expected schema, and suggested correction. Loading never changes the source.

## Transformations

```python
resplit = dataset.split(
    {"train": 0.75, "val": 0.25},
    group_by=lambda path: path.parent.name,
    seed=42,
    visualize=True,
)

clean = resplit.remove_classes(["damaged"], visualize=True)
grid = clean.tile(mode="grid", tile_size=480, overlap=0.2)
balanced = grid.rebalance_empty(0.20, splits=("train",), seed=42)
canonical = balanced.export(
    destination="/datasets/orchard_ready",
    splits=("train", "val"),
)
```

Albumentations is optional. Install it with
`pip install "dataset-fixer[augment]"`, then pass either a transform sequence,
an `A.Compose`, or an `A.to_dict()` result:

```python
import albumentations as A

augmented = dataset.augment(
    [
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-10, 10), p=0.7),
        A.RandomBrightnessContrast(p=0.3),
    ],
    copies=2,
    splits=("train",),
    include_original=True,
    seed=42,
    visualize=True,
)
exported = augmented.export(destination="/datasets/orchard_augmented")
```

The package installs its own bounding-box and keypoint processors so annotation
rows cannot become detached from instances. Pose keypoints honor `flip_idx` on
horizontal flips. Segments and POLO radii are transformed through synchronized
masks. The exact serialized pipeline, per-image seed, applied transforms,
dropped annotations, and warnings are retained in the manifest, provenance, and
`reports/augmentation.json`. Validation rejects tensor/normalized image outputs
because exported datasets require RGB `uint8` images.

No output directory is created before the final call. A virtual dataset exposes
its projected classes, splits, settings, history, and pending operation names,
but `location` still identifies the immutable source, `data_yaml` is `None`, and
`training_ready` is false until export.

Grid tiling clips boxes, polygons, and pose instances and requires POLO circles
to fit fully inside a crop. `negative_tiles` accepts `"all"`, `"none"`, or a
ratio relative to positive tiles. Coverage tiling is POLO-specific:

```python
coverage = dataset.tile(
    mode="coverage",
    tile_size=480,
    # Any notebook-derived advanced default can be overridden by keyword.
    fixed_polo_radius_px=15,
    target_coverage_per_label=5,
    sparse_coverage_per_label=1,
    max_bg_ratio=0.10,
    seed=42,
)
```

`rebalance_empty(max_empty_fraction=...)` deterministically downsamples empty
images without duplicating data. The cap is applied independently to each
selected split; non-selected splits are preserved. If the requested ratio would
require more empty images, the existing images are retained rather than copied.

Visualization preferences are recorded on virtual operations and rendered
during export. Intermediate plan steps reuse their in-memory sample index and
are not reopened or revalidated. `export(progress=True)` shows progress and ETA
for copying, tiling, and the single complete staged-output validation performed
before atomic publication. The validated index is reused after publication, so
the final dataset is not scanned a second time. Its destination must not exist
and cannot contain, equal, or be contained by the source.

Canonical exports use split-first layout:

```text
data.yaml
train/images/  train/labels/
val/images/    val/labels/
test/images/   test/labels/
```

## Colab tutorials

Three end-to-end notebooks document the central workflows using official images
and YOLO labels downloaded from a pinned commit of the public MIT-licensed
[SAWIT dataset](https://github.com/dtnguyen0304/sawit):

- [Controlled, group-aware splitting](notebooks/01_controlled_splitting.ipynb)
  demonstrates `Dataset.split()` and physical-group isolation.
- [Task-aware tiling](notebooks/02_task_aware_tiling.ipynb) demonstrates regular
  detection grids and negative-tile policies.
- [Fixed-cohort model comparison](notebooks/03_fixed_cohort_model_comparison.ipynb)
  trains two small checkpoints and demonstrates cached `Dataset.compare_models()`.

Each notebook includes an “Open in Colab” badge, kernel-safe installation,
licensing notes, expected runtime, validation checks, and source provenance.
See the [notebook guide](notebooks/README.md) and the
[public-data downloader](examples/README.md).

## Cached model comparison

`compare_models()` freezes one ordered evaluation cohort and requires every
model to predict exactly those images. It never takes a validation split from a
checkpoint or training configuration.

```python
comparison = dataset.compare_models(
    {
        "baseline": {
            "path": "/models/baseline.pt",
            "training_dataset": "/datasets/baseline-training",
            "resolution": 480,
        },
        "random-crops-2": {
            "path": "/models/crops-2.pt",
            "training_dataset": "/datasets/random-crops-2",
            "resolution": 480,
        },
    },
    split="val",
    baseline="baseline",
    protocol="validation",
    inference="auto",
)
```

`protocol="validation"` labels threshold selection as validation/model
selection. `protocol="locked"` evaluates one predetermined configuration per
model. `protocol="calibrate_then_test"` selects settings only on
`calibration_split` and applies them to the distinct requested split.

Native Ultralytics inference is supported for detection, segmentation, pose,
and POLO. If SAHI is installed, `inference="auto"` uses it for detection,
segmentation, and POLO; pose always uses native inference. Explicit SAHI use
never falls back silently. Install optional integrations with:

```shell
pip install 'dataset-fixer[comparison,sahi]'
```

Predictions are cached independently of metrics and figures in an atomic,
content-addressed NumPy format. Compatible notebook `.gridcache.pkl` and
`.gridcache_v2` caches are validated and imported; new pickle caches are never
written. Single-class POLO/SAHI results can be mirrored back to the notebook
format with `write_notebook_cache=True`.

The returned `ComparisonResult` reports factual state: the cohort fingerprint,
cohort verification, training overlap, provenance completeness, cache
verification, ranking, settings, and explicit limitations. Its output contains
machine-readable metric grids, paired cluster statistics, cache/cohort/leakage
audits, aligned qualitative panels, and PDF/SVG/600-DPI PNG figures with CSV
data and JSON metadata sidecars.

## Output and reproduction

Derived datasets contain `images/{split}`, `labels/{split}`, `data.yaml`,
`dataset-fixer.json`, `provenance.jsonl`, and operation reports. Coverage mode
also writes its four CSV reports and annotated originals under
`coverage_summary/`. Publication is atomic and happens only after the private
output tree passes full validation.

The manifest records resolved defaults and callbacks, seeds, class mappings,
source fingerprints, package/setuptools-scm revision, dirty state, caller Git
revision, Python/platform/dependency versions, timings, warnings, audit paths,
and operation history. Each provenance row contains immediate-parent and
ultimate-original paths/hashes plus crop, scale, tile, class mapping, and
transformation-chain data.

The intentionally small public API is:

- `Dataset.open`
- `Dataset.split`
- `Dataset.remove_classes`
- `Dataset.tile`
- `Dataset.export`
- `Dataset.compare_models`
- `Dataset.visualize`
- `Dataset.assert_trainable`

Operation-specific previews and audits are controlled with each method's
`visualize=` parameter. Consistency checks run automatically during loading.

## Development

The GitHub Actions pipeline tests Python 3.10–3.13, exercises the optional
comparison and SAHI dependency set, builds the wheel and source distribution,
checks package metadata, and smoke-tests installation from the wheel.

## License

`dataset-fixer` is released under the [MIT License](LICENSE).
