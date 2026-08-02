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
expected schema, and suggested correction. To keep loading while omitting
recoverably bad records from the in-memory dataset, opt in explicitly:

```python
dataset = Dataset.open(
    "/datasets/orchard/data.yaml",
    task="segment",
    errors="skip",
)
print(*dataset.warnings, sep="\n")
```

The default is `errors="raise"`. With `errors="skip"`, recoverable bad images,
label rows, annotations, duplicate records, incompatible COCO split files,
missing split entries, orphan labels, and incomplete provenance are omitted
virtually and listed in `dataset.warnings`. An unusable class schema or a dataset
with no valid images still raises. Loading never changes the source.

## Transformations

```python
resplit = dataset.split(
    {"train": 0.75, "val": 0.25},
    group_by=lambda path: path.parent.name,
    seed=42,
    visualize=True,
)

clean = resplit.remove_classes(["damaged"], visualize=True)
renamed = resplit.rename_classes({"damaged": "blemished"})
grid = clean.tile(mode="grid", tile_size=480, overlap=0.2)
balanced = grid.rebalance_empty(0.20, splits=("train",), seed=42)
canonical = balanced.export(
    destination="/datasets/orchard_ready",
    splits=("train", "val"),
)
```

`rename_classes()` changes only class metadata; class IDs, annotation rows,
geometry, POLO radii, and pose metadata remain unchanged until the renamed
metadata is written by `export()`.

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

Grid tiling uses deterministic, fixed-size source windows; it does not randomly
crop, resize, or zoom. It clips boxes, polygons, and pose instances and requires
POLO circles to fit fully inside a crop. `negative_tiles` accepts `"all"`,
`"none"`, or a final output background fraction in `[0, 1)`, including
uncropped small images. Coverage tiling supports detection, segmentation, pose,
and POLO and provides notebook-derived random crop/zoom with per-object
appearance targets:

```python
coverage = dataset.tile(
    mode="coverage",
    tile_size=480,
    scale_range=(0.75, 1.25),
    target_appearances_per_object=5,
    sparse_appearances_per_object=1,
    background_ratio=0.10,
    # The predicate receives the final RGB background candidate. True keeps it.
    background_filter=lambda tile: tile.getbbox() is not None,
    seed=42,
)
```

Coverage crops can also sample a fresh Albumentations virtual camera view for
each output. The complete source image and synchronized annotations are
transformed first; crop selection happens afterward in transformed coordinates:

```python
coverage = dataset.tile(
    mode="coverage",
    tile_size=480,
    crop_transforms=A.Compose([
        A.Affine(scale=(1.0, 1.3), rotate=(-20, 20), p=1.0),
        A.RandomBrightnessContrast(p=0.3),
    ]),
    augment_val=True,   # needed when inspecting/using transformed val crops
    allow_lossy=True,
    errors="skip",      # reject rare non-exportable crop geometries and resample
    seed=42,
)
```

Resize, crop, and pad transforms are supported. Generated border pixels are
tracked separately and never accepted inside an output tile, even when a
transform requests reflection or fill. With `allow_lossy=False`, transformed
views or final crops that cut annotations are resampled. With `allow_lossy=True`,
representable geometry is clipped; disconnected segment results retain their
largest YOLO-representable polygon with an explicit warning. Small coverage
pass-through images are unchanged. Tiling defaults to `errors="raise"`, whose
diagnostic includes the source image, annotation index, crop coordinates, and
the exact Shapely result and component types. Set `errors="skip"` to reject
those whole candidates instead: coverage mode resamples replacements, grid
mode omits the affected windows, and details are written to
`reports/tiling_skips.json`.

Each produced training crop samples its own independently seeded full-source
virtual camera. Validation stays in the original orientation unless
`augment_val=True`; test is never transformed. This matters when previewing
validation with `visualize(split="val", ...)`. The exact transformed and
unchanged splits, along with per-candidate sampling statistics, are recorded in
`reports/crop_augmentation.json`; it also audits the accepted crop count,
distinct seed count, and `fresh_seed_per_accepted_crop` invariant.

Set both appearance parameters to the same value to request a uniform count for
every object. `object_appearance_overrides={source_id: count}` overrides
individual annotations. `background_ratio=0.10` targets 10% annotation-free
images across the complete coverage output of each split, including copied
small images and newly generated crops. Half of the target is sampled from
wholly empty source images and half from object-free regions of populated
images where possible. If either source cannot supply its half, the other
cross-fills the target and the exact counts and reason are written to
`coverage_summary/background_sampling.json`. IDEs can autocomplete the literal
choices for `mode`, `negative_tiles`, `errors`, tasks, splits, comparison
protocols, and inference backends; all coverage controls are explicit `tile()`
parameters. `background_filter` also applies to negative grid windows, copied
empty source images, ordinary background crops, and virtual-camera background
cropping paths. It runs after any applicable transforms, cropping, and final
resizing, never sees a positive tile, and receives an isolated RGB PIL image so
it cannot mutate output pixels. Absolute accepted/rejected counts, percentages,
per-source-path breakdowns, and the callback description are recorded in
`reports/background_filter.json`.

Coverage tiling also audits what the output actually represents. It always
writes per-source unioned spatial coverage to
`coverage_summary/source_pixel_coverage.csv` and aggregate train/validation
statistics to `source_pixel_coverage.json`. With visualization enabled (the
default), `source_pixel_coverage.jpg` compares pixel-weighted, mean, median, and
per-source coverage, while `label_coverage.jpg` compares labels reached at
least once with requested appearances produced. Virtual affine/projective crop
footprints are inverse-mapped into source coordinates before unioning; an
unsupported non-projective transform is reported explicitly and excluded from
exact aggregates. Semantic-mask exports preserve the complete
`coverage_summary/` directory.

`rebalance_empty(max_empty_fraction=...)` deterministically downsamples empty
images without duplicating data. The cap is applied independently to each
selected split; non-selected splits are preserved. If the requested ratio would
require more empty images, the existing images are retained rather than copied.

Composition parameters always refer to the complete output of their operation:
coverage `background_ratio`, numeric grid `negative_tiles`, and
`rebalance_empty(max_empty_fraction=...)` all count pass-through images as well
as generated images. Geometric fractions such as `overlap`, `min_area_ratio`,
augmentation `min_visibility`, and SAHI overlap apply to windows or individual
annotations rather than dataset composition. Split weights accept either
fractions (`0.8/0.2`) or percentages (`80/20`) and are normalized; grouping and
explicit assignments can make the achieved split counts approximate.

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

For polygon segmentation datasets, `format="semantic_masks"` publishes binary
foreground-union masks for U-Net-style training and returns a
`SemanticMaskExport`:

```python
from dataset_fixer import SemanticMaskExport

masks = dataset.export(
    destination="/datasets/orchard_masks",
    format="semantic_masks",
)
print(masks.image_dirs["train"])
print(masks.mask_dirs["train"])

# The same artifact can be reopened in a later process or notebook.
masks = SemanticMaskExport.open("/datasets/orchard_masks")
```

Each original image is copied byte-for-byte to `<split>/images/`; its
single-channel PNG mask is written to `<split>/masks/0/` with matching nested
relative path and stem. Mask values are `0` for background and `255` for the
union of all polygons, intentionally discarding class and instance identity.
Semantic-mask exports contain no YOLO labels or `data.yaml`.

The returned artifact can compare official nnU-Net v2 models directly on one
frozen exported split. Pass the trained-model directory that contains
`dataset.json`, `plans.json`, and `fold_*` subdirectories—not a standalone
checkpoint file:

```python
comparison = masks.compare_models(
    {
        "resenc-capped": {
            "model_folder": capped_fold_output.parent,
            "folds": (0,),
            "checkpoint": "checkpoint_best.pth",
            "upscale_factor": 2,
        },
        "resenc-l": {
            "model_folder": large_fold_output.parent,
            "folds": (0,),
            "checkpoint": "checkpoint_best.pth",
            "upscale_factor": 1,
        },
    },
    split="val",
    baseline="resenc-capped",
    device="cuda",
)
```

Install the optional official backend with
`pip install 'dataset-fixer[nnunet]'`. Comparison calls
`nnUNetv2_predict_from_modelfolder` and `nnUNetv2_evaluate_folder`, verifies
that every model predicted the exact same image set, and ranks the official
foreground Dice/IoU results. Each model receives inputs at its own configured
training-adapter scale; predictions are projected back to the original export
resolution before all models are evaluated against the same canonical masks.
The report includes per-case metrics, paired Dice
deltas with bootstrap intervals, model/checkpoint hashes, official summaries,
retained prediction masks, a ranking figure, and a qualitative error grid.
For the attached official-training notebook, `model_folder` is
`FOLD_OUTPUT.parent`, `folds=(FOLD,)`, `checkpoint=PREDICTION_CHECKPOINT`, and
the model configuration's `upscale_factor` is `UPSCALE_FACTOR`.

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

Derived datasets contain `{split}/images`, `{split}/labels`, `data.yaml`,
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
- `Dataset.rename_classes`
- `Dataset.rebalance_empty`
- `Dataset.tile`
- `Dataset.augment`
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
