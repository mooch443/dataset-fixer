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
the compressed `reports/lineage.json.gz` index.

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

Every skipped or ignored load-validation failure is also counted in
`dataset.validation_audit`. Loading prints the total and renders at most four
examples in a compact grid, prioritizing readable images and invalid geometry
such as self-intersecting polygons. The grid is written outside the source
dataset and is carried into later exports as
`reports/load_validation_examples.png`, alongside the aggregate audit in
`reports/load_validation_audit.json`.

## Transformations

```python
resplit = dataset.split(
    {"train": 0.75, "val": 0.25},
    group_by=lambda path: path.parent.name,
    seed=42,
    visualize=True,
)

clean = resplit.remove_classes(["damaged"], visualize=True)
merged = resplit.remove_classes(["damaged"], merge_into="fruit")
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

`remove_classes()` normally discards annotations belonging to the selected
classes. Pass `merge_into=` with a surviving class name or integer ID to retain
those annotations under that class; surviving class IDs are still compacted in
the exported dataset.

Albumentations is installed with `pip install dataset-fixer`; pass either a transform sequence,
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
mode omits the affected windows, and details are embedded in
`reports/dataset-info.json`.

Each produced training crop samples its own independently seeded full-source
virtual camera. Validation stays in the original orientation unless
`augment_val=True`; test is never transformed. This matters when previewing
validation with `visualize(split="val", ...)`. The exact transformed and
unchanged splits, along with per-candidate sampling statistics, are recorded in
the `audits` section of `reports/dataset-info.json`; it also audits the accepted crop count,
distinct seed count, and `fresh_seed_per_accepted_crop` invariant.

Set both appearance parameters to the same value to request a uniform count for
every object. `object_appearance_overrides={source_id: count}` overrides
individual annotations. `background_ratio=0.10` targets 10% annotation-free
images across the complete coverage output of each split, including copied
small images and newly generated crops. Half of the target is sampled from
wholly empty source images and half from object-free regions of populated
images where possible. If either source cannot supply its half, the other
cross-fills the target and the exact counts and reason are recorded in the
dataset information report. IDEs can autocomplete the literal
choices for `mode`, `negative_tiles`, `errors`, tasks, splits, comparison
protocols, and inference backends; all coverage controls are explicit `tile()`
parameters. `background_filter` also applies to negative grid windows, copied
empty source images, ordinary background crops, and virtual-camera background
cropping paths. It runs after any applicable transforms, cropping, and final
resizing, never sees a positive tile, and receives an isolated RGB PIL image so
it cannot mutate output pixels. Absolute accepted/rejected counts, percentages,
per-source-path breakdowns, and the callback description are recorded in
`reports/dataset-info.json`.

Coverage tiling also audits what the output actually represents. Per-source
unioned spatial coverage, aggregate train/validation statistics, and label-hit
counts are stored under `audits` in `reports/dataset-info.json`. With
visualization enabled, the visual summaries are panels in the single
`reports/plots.png`. Virtual affine/projective crop
footprints are inverse-mapped into source coordinates before unioning; an
unsupported non-projective transform is reported explicitly and excluded from
exact aggregates.

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

When the latest split used `group_by`, every later export automatically audits
that physical group across the complete current dataset, even if `export(splits=...)`
publishes only a subset. Export fails if a group crosses train/validation/test
or if any image has lost its group provenance. Successful YOLO and semantic-mask
exports record per-split distinct-group counts and aggregate group-size
distributions in `reports/split_group_audit.json`; individual image paths and
group membership lists are intentionally omitted.

Visualization preferences are recorded on virtual operations and rendered
during export. Intermediate plan steps reuse their in-memory sample index and
are not reopened or revalidated. `export(progress=True)` shows progress and ETA
for copying, tiling, and the single complete staged-output validation performed
before atomic publication. Staged validation streams images, labels, and
provenance instead of constructing a second complete dataset index. The
validated index is transferred to the published result, so the final dataset is
not scanned a second time. Its destination must not exist and cannot contain,
equal, or be contained by the source.

Canonical exports use split-first layout:

```text
data.yaml
train/images/  train/labels/
val/images/    val/labels/
test/images/   test/labels/
```

For polygon segmentation datasets, `format="semantic_masks"` publishes binary
foreground-union masks for U-Net-style training and returns the same `Dataset`
type used for YOLO and COCO:

```python
masks = dataset.export(
    destination="/datasets/orchard_masks",
    format="semantic_masks",
)
print(masks.image_dirs["train"])
print(masks.mask_dirs["train"])

# The same artifact can be reopened in a later process or notebook.
masks = Dataset.open("/datasets/orchard_masks")
```

To publish the same segmentation pipeline in both formats, use
`export_formats`. It validates and writes YOLO first, then builds semantic masks
from that materialized result:

```python
exports = dataset.export_formats(
    {
        "yolo": "/datasets/orchard_yolo",
        "semantic_masks": "/datasets/orchard_masks",
    }
)
yolo = exports["yolo"]
masks = exports["semantic_masks"]
```

Both destinations are checked before writing starts and must be distinct,
non-nested paths. The ordinary single-format `export()` API remains unchanged.

Each original image is copied byte-for-byte to `<split>/images/`; its
single-channel PNG mask is written to `<split>/masks/0/` with matching nested
relative path and stem. Mask values are `0` for background and `255` for the
union of all polygons, intentionally discarding class and instance identity.
Semantic-mask exports contain no YOLO labels or `data.yaml`.

Official nnU-Net v2 models use the same model-first API for sampled
visualization or a complete comparison. Pass the trained-model directory that
contains `dataset.json`, `plans.json`, and `fold_*` subdirectories—not a
standalone checkpoint file:

```python
from dataset_fixer import Model

models = Model.load_many(
    {
        "resenc-capped": {
            "path": capped_fold_output.parent,
            "folds": (0,),
            "checkpoint": "checkpoint_best.pth",
            "upscale_factor": 2,
        },
        "resenc-l": {
            "path": large_fold_output.parent,
            "folds": (0,),
            "checkpoint": "checkpoint_best.pth",
            "upscale_factor": 1,
        },
    }
)

# Predict only eight sampled validation images and return a Matplotlib figure.
figure = models.visualize(
    masks,
    split="val",
    samples=8,
    examples_per_row=1,
    destination="quick-comparison.png",
)

# Or predict and evaluate the complete validation split.
comparison = models.compare(
    masks,
    split="val",
)
```

Mixed YOLO segmentation and semantic-segmentation models use the same API.
For a semantic-mask dataset, YOLO polygons are automatically unioned into
binary foreground masks at the exported mask resolution, then every model
receives the same semantic Dice/IoU evaluation:

```python
models = Model.load_many({
    "yolo-seg": {
        "path": "/models/yolo-seg.pt",
        "task": "segment",
        "resolution": 640,
    },
    "nnunet": {
        "path": "/models/nnUNetTrainer__Plans__2d",
        "folds": (0,),
        "checkpoint": "checkpoint_best.pth",
    },
})

comparison = models.compare(
    masks,
    split="val",
    # None uses the held-out reference-object p10. This filters only the
    # area-filtered image-presence metrics; raw any-pixel values remain.
    min_connected_component_area=None,
    # The callback receives each evaluation image Path.
    group_by=lambda path: path.parent.name,
)
```

The report records each model's native task and applied projection. Same-type
collections continue through their native evaluator; incompatible mixtures
without an implemented common denominator fail with a validation error.
Image-level presence is reported both raw (any foreground pixel) and after
requiring an 8-connected predicted component at least as large as the resolved
minimum. The report heatmap includes presence precision, positive-image recall,
and empty-image specificity for both definitions. When
`min_connected_component_area=None`, the minimum is the p10 area of reference
objects in the active held-out cohort; a positive numeric value overrides it.
`group_by` is report-only: TP, FP, and FN are pooled within each returned group,
then group Dice is macro-averaged with equal group weight. It adds
`reports/grouped-metric-breakdown.png` and does not alter inference, the cohort,
or the primary ranking.

The sampled figure and the full comparison use the same renderer: each example
has one filename title and columns for Original, GT, and each shortened model
name. Model panels show masks only, with Dice and IoU beneath them. The
`examples_per_row` and `panel_size` arguments control the grid. Datasets do
not expose model-loading or model-comparison methods; load models with
`Model.load_many(...)` and call `models.compare(masks, ...)`.

The official backend is included by `pip install dataset-fixer`. Comparison calls
`nnUNetv2_predict_from_modelfolder` for whole-image prediction and
`nnUNetv2_evaluate_folder` for every reported metric, verifies
that every model predicted the exact same image set, and ranks the official
foreground Dice/IoU results. Sliced (`inference="sahi"`) nnU-Net prediction
runs in process instead, against the same official preprocessing and
probability-conversion APIs. A model's `workers` setting is the CPU worker
count for preprocessing and probability conversion; it is not the neural
network batch size. With `batch_size=-1`, nnU-Net derives that from its plan,
caps the initial probe at 16 on an accelerator or 4 on CPU, and halves only on
a recognized out-of-memory error; a positive value sets the initial request.
The resolved batch size, retries, tile count, per-phase timings, device, and
execution engine are recorded in prediction and comparison manifests.
When `device` is omitted, nnU-Net execution uses
CUDA when available, then Apple MPS, and otherwise CPU. An explicit per-model
`device` remains unchanged. Each model receives inputs at its own configured
training-adapter scale. nnU-Net class probabilities are area-averaged back to
the original export resolution before `argmax`, so comparison does not depend
on an arbitrary nearest-neighbor sample from each high-resolution block. All
models are ranked against the same canonical masks; the report also retains
the official native-resolution score for each model. Aggregate metrics, finite
and total cohort support, paired Dice deltas, bounded worst cases, hashes, and
resolved settings live in `reports/result.json`; the two mandatory visual files
are `reports/plots.png` and `reports/comparison.png`. Canonical masks and
official per-case results remain only in the dataset-local cache. Set
`save_prediction_plots=True` when per-image annotated comparison grids are
wanted. For the attached official-training notebook, `path` is
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
  trains two small checkpoints and demonstrates cached `ModelCollection.compare()`.

Each notebook includes an “Open in Colab” badge, kernel-safe installation,
licensing notes, expected runtime, validation checks, and source provenance.
See the [notebook guide](notebooks/README.md) and the
[public-data downloader](examples/README.md).

## Unified model API

`Model` is the single loading and prediction interface. It detects official
nnU-Net trained-model folders from `dataset.json` and `plans.json`; model files
use the Ultralytics adapter. Resolution, nnU-Net folds/checkpoint/upscale, the
default device, and other adapter choices belong to the model and are reused by
prediction, visualization, and comparison:

```python
from dataset_fixer import Model

detector = Model(
    "/models/best.pt",
    name="detector",
    resolution=640,
    batch_size=-1,
)
predictions = detector.predict(
    "/data/example.jpg",
    confidence=0.25,
)
print(predictions.summary())
predictions.save("detector-predictions")

segmenter = Model(
    "/models/nnUNetTrainer_100epochs__nnUNetResEncUNetMPlans__2d",
    name="resenc-m",
    folds=(0,),
    checkpoint="checkpoint_best_mean_fg_dice.pth",
    upscale_factor=4,
    device="mps",
)
mask_predictions = segmenter.predict(masks, split="val")

# Direct prediction caching is opt-in. Dataset inputs use the same cache base
# as Model.compare(), so either operation can reuse compatible predictions.
cached = segmenter.predict(masks, split="val", prediction_cache=True)
```

`PredictionResult` preserves input ordering and stable image IDs. Object-style
models populate each record's `objects`; semantic models populate `mask`.
`by_id`, `masks`, `summary()`, and `save()` provide common inspection/export
utilities. Direct `Model.predict` defaults to native prediction. Configure a
model with `inference="sahi"` to opt it into tiled inference; no
availability-based selection or fallback is performed.

Use `prediction_cache=True` when a direct prediction run should be retained.
For explicit placement or reuse across calls, construct
`PredictionCache("/cache/base")` and pass it as `prediction_cache=cache`.
`PredictionCache.for_dataset(dataset)` resolves the established
`<dataset>/.cache/evaluations` base, preserving the `predictions/`,
`semantic/`, and `metrics/` namespaces used by comparison. `PredictionResult`
records the verified cache status, key, namespace, and location in
`cache_info`. Direct prediction remains uncached when the argument is omitted.

`prediction_threshold` is task-aware. Semantic models apply it to genuine
foreground probabilities after full-image reconstruction; requesting that
operation from a backend that returns only hard class maps raises
`PredictionScoreUnavailableError`. The hard-mask cache is retained and remains
usable for unthresholded prediction. Instance-segmentation models retain their
scored polygons and apply the same setting as an object-score floor. Call
`ImagePrediction.foreground_score_map()` to rasterize those polygons lazily as
the maximum covering object score per pixel. This object-confidence projection
supports the common semantic evaluation space without pretending to be a
pixel-calibrated posterior or replacing the original instances.

Inference defaults to `batch_size=-1`. Ultralytics native inputs and SAHI
tiles are sent through the official prediction API in cohort-wide batches;
recognized CUDA/MPS out-of-memory failures halve only the failed batch and the
resolved size is reused. A positive batch size fixes the initial requested
size. Both automatic and explicit batches are capped at 128. Loaded weights
are retained per model and device until `model.unload()`.

Use `Model.load_many(...)` to normalize an ordered model collection once. The
collection accepts a dataset/export at operation time and exposes `predict`,
`compare`, and—for semantic-mask exports—the shared sampled `visualize` grid.

Source geometry uses an upper bound, not an exact-size requirement. Every
Ultralytics and nnU-Net prediction/evaluation path retains images whose height
and width are at most the model's `native_tile_size`. An image exceeding either
dimension raises by default. Pass `errors="skip"` to `predict()`, `compare()`,
or `visualize()` to omit only those oversized images. Prediction results record
the skipped inputs in `settings.source_size_policy`; comparisons use one common
filtered cohort for every model and record the same audit in the report.

## Cached model comparison

`ModelCollection.compare()` freezes one ordered evaluation cohort and requires every
model to predict exactly those images. It never takes a validation split from a
checkpoint or training configuration.

Comparison continues to cache automatically when `prediction_cache` is
omitted. Pass an explicit `PredictionCache` or cache-base path to redirect the
same layout, or `prediction_cache=False` to disable persistent prediction and
metric caching for that comparison run.

```python
models = Model.load_many(
    {
        "reference": {
            "path": "/models/reference.pt",
            "training_dataset": "/datasets/reference-training",
            "resolution": 480,
            "confidence": 0.50,
            "postprocess": 0.85,
        },
        "random-crops-2": {
            "path": "/models/crops-2.pt",
            "training_dataset": "/datasets/random-crops-2",
            "resolution": 480,
            "confidence": 0.50,
            "postprocess": 0.85,
        },
    }
)

comparison = models.compare(
    dataset,
    split="val",
)

# Every comparison reports all unordered model pairs automatically.

# SAHI belongs to each model specification, not the comparison call.
sliced_models = Model.load_many({
    "sahi-512": {
        "path": "/models/crops-2.pt",
        "inference": "sahi",
        "resolution": 640,
        "sahi_slice_height": 512,
        "sahi_slice_width": 512,
        "sahi_overlap": 0.2,
    },
})
sliced = sliced_models.compare(dataset, split="val")
```

Semantic prediction caches created before logical cache identities were stored
can be migrated once with
`models.compare(dataset, trust_legacy_cache=True)`. The opt-in requires an
exact model-name, case-ID, relative-path, metadata, and complete-mask match and
refuses ambiguous matches. A successful migration records the trust decision
under the current identity, so later calls should omit the option and reuse the
cache normally.

`models.compare()` has no shared inference, resolution, threshold, protocol,
comparison-space, comparison-unit, or model-reference options. Each model
carries its own configuration, the dataset/task pair determines the canonical
metric space, and every unordered model pair is evaluated. A single-model run
has no pairwise statistics.
The report labels candidates as model variants only when their inference
systems match; otherwise it labels them as distinct systems automatically.

Native and sliced SAHI inference are supported for detection, instance
segmentation, pose, POLO, Ultralytics semantic segmentation, and nnU-Net.
Pose keypoints and POLO point/radius payloads are retained while duplicates are
merged, and dense semantic tiles are feather-blended as probabilities before a
single full-image argmax. Ultralytics tiles are submitted in bounded GPU
batches. nnU-Net sliced inference runs in process against nnU-Net's own Python
preprocessing and probability-conversion APIs: each fold's weights are loaded
once per run instead of once per tile, equally shaped tiles are grouped into
real network minibatches, and probabilities stay in memory instead of round
tripping through one PNG/NPZ pair per tile. Completed source images are
stitched and released as soon as their tiles finish. The canonical tile
manifest, overlap, mirroring TTA, upscale adapter, feathered probability
stitching, and argmax-after-stitch behavior are unchanged, and the official
evaluator CLI still produces every reported metric. All SAHI-only options use
the `sahi_` prefix; the removed unprefixed names and `inference="auto"` are
rejected. Install the complete runtime with:

```shell
pip install dataset-fixer
```

Predictions and completed per-model evaluations are cached independently in
`<dataset>/.cache/evaluations/`. Keys include the frozen cohort, checkpoint
bytes, backend, resolved model inference/SAHI settings, metric protocol, and relevant
dependency versions. Repeating the same comparison performs no inference or
evaluation; changing one model recomputes only that model. Caching is always
enabled and has no public switch.

Dataset ZIPs are accepted directly by the unchanged `Dataset.open()` API.
Likewise, `Model.load_many()` accepts local `.pt` files, nnU-Net folders,
portable bundle ZIPs, `wandb:entity/project/run-id` references, full W&B run
URLs, model objects, and mappings. Drive inputs are copied to local Colab
storage, downloads and safe extraction are atomic, and unchanged inputs are
reused from the automatic cache. Standalone checkpoints with unproven training
geometry still load; configure only the missing values after resolving names:

```python
models = Model.load_many(MODEL_SOURCES)
models = models.configure({
    "standalone-yolox": {
        "task": "semantic",
        "native_tile_size": 128,
        "upscale_factor": 2,
        "inference": "sahi",
        "device": "cuda",
        "batch_size": -1,
    }
})
```

Training preparation and portable output use focused subpackages:

```python
from dataset_fixer.convert import Kind, prepare
from dataset_fixer.bundle import Config, Outcome, create
from dataset_fixer.wandb import configure, upload

prepared = prepare(
    dataset,
    Kind.YOLO_SEM,
    native_tile_size=128,
    upscale_factor=2,
    errors="skip",  # omit only images larger than 128x128
)
config = Config(
    name="islands-sem",
    framework="ultralytics",
    task="semantic",
    geometry=prepared.geometry,
    dataset=prepared,
)
configure(existing_run, config)  # never initializes or logs in
bundle = create(config, Outcome(checkpoint="runs/segment/train/weights/best.pt"))
bundle = upload(existing_run, bundle)  # local ZIP survives every outcome
```

YOLO preparations are published in a split-first layout:

```text
train/images/  train/labels/
valid/images/  valid/labels/
```

Their portable `data.yaml` keeps the Ultralytics `val` key but points it to
`valid/images`. It contains no absolute `path`, so it continues to resolve
relative to the YAML file after a cache move or atomic publication. YOLO
semantic masks are PNG files under the corresponding `labels` directory and
are selected with `masks_dir: labels`.

`prepare()` records exact interpolation and label mapping, requires an explicit
threshold for binary JPEG recovery, never derives polygons from semantic masks,
and content-addresses all generated data. Images at or below
`native_tile_size` are expected: they are retained and resized with their masks
to the configured training input size. Only an image exceeding either native
dimension is an error. The default `errors="raise"` aborts on that image;
`errors="skip"` omits it and records the omission in `preparation-skips.json`.
`bundle.create()` always creates the ZIP in local storage. Neither helper
uploads or copies a bundle to Google Drive.

The returned `ComparisonResult` reports factual state: the cohort fingerprint,
cohort verification, training overlap, provenance completeness, cache
verification, ranking, settings, and explicit limitations. Every evaluation
writes `reports/result.json`, `reports/plots.png`, and
`reports/comparison.png`. The comparison image uses a deterministic seeded
random sample of non-empty ground-truth cases. Pass
`save_prediction_plots=True` to add one annotated
`predictions/<image>.png` grid per source image, with at most two model panels
per row.

## Output and reproduction

Derived datasets contain `{split}/images`, `{split}/labels`, `data.yaml`, and a
compact `reports/` directory holding exactly four files.
`reports/dataset-info.json` holds the dataset ID,
settings, source path/ID/basic metadata, per-split total/labeled/background
image counts, validation, history, and operation audits; `reports/source.json`
is the immediate-source snapshot;
`reports/lineage.json.gz` maps every present output to its physical parent and
ultimate original; `reports/plots.png` summarizes the dataset as it is on disk.
Publication remains atomic.

`reports/plots.png` is regenerated from the physically present dataset rather
than accumulated from previous operations. It shows the dataset name, task,
format, and classes; one annotated/background composition bar per split with
image counts and percentages; and one example row per split with four
deterministically selected images, drawn with task-aware annotation overlays or
red mask overlays for semantic datasets. Operation audits stay structured in
`reports/dataset-info.json` instead of becoming report images.

Generated `data.yaml` files contain only relative split entries and metadata,
with no `path` key, so a dataset resolves correctly wherever it is moved or
mounted. Absolute physical locations remain recorded in
`reports/dataset-info.json` and `reports/source.json` for tracing.

The manifest records resolved defaults and callbacks, seeds, class mappings,
source fingerprints, package/setuptools-scm revision, dirty state, caller Git
revision, Python/platform/dependency versions, timings, warnings, audit paths,
and operation history. Each lineage record contains immediate-parent and
ultimate-original paths/hashes plus crop, scale, tile, class mapping, and
transformation-chain data.

Datasets can be traced after a move or remount using stable IDs and optional
path-prefix rewrites:

```python
trace = exported.trace(
    search_paths=["/mnt/datasets"],
    path_rewrites={"/old/datasets": "/mnt/datasets"},
)
print(trace.summary())
tile = trace.for_sample("train/images/example__tile-0001.jpg")
print(tile.original_image, tile.crop)
```

Generated datasets can be migrated to the current compact artifact schema
without changing image, label, or mask bytes. Omit `dest` for an atomic in-place
report update, or pass a string/`Path` to create an upgraded physical copy. The
upgrade also removes any legacy `path` key from `data.yaml`, making an older
dataset portable. `progress=True` (the default) reports copying, hashing and
indexing, report generation, and validation; pass `progress=False` to run
silently:

```python
updated = dataset.update()
copied = dataset.update(dest="/datasets/islands-current")
quiet = dataset.update(progress=False)
```

The intentionally small public API is:

- `Dataset.open`
- `Dataset.split`
- `Dataset.remove_classes`
- `Dataset.rename_classes`
- `Dataset.rebalance_empty`
- `Dataset.tile`
- `Dataset.augment`
- `Dataset.export`
- `Dataset.visualize`
- `Dataset.assert_trainable`
- `Dataset.trace`
- `Dataset.update`
- `Model`
- `Model.predict`
- `Model.compare`
- `Model.load_many`
- `ModelCollection.configure`
- `ModelCollection.predict`
- `ModelCollection.compare`
- `PredictionCache`

Operation-specific previews and audits are controlled with each method's
`visualize=` parameter. Consistency checks run automatically during loading.

## Development

The GitHub Actions pipeline tests supported Python versions and the complete
runtime dependency set, builds the wheel and source distribution,
checks package metadata, and smoke-tests installation from the wheel.

## License

`dataset-fixer` is released under the [MIT License](LICENSE).
