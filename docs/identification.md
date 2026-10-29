# Individual identification

Install `pip install 'dataset-fixer[id]'`. ID dependencies import lazily. Existing detection calls keep their defaults; the shared `TrainingConfig.augmentation_samples=12` also enables a startup augmentation grid for detection trainers (`0` disables it).

One encoder and one enrollment gallery handle both closed and open queries. Training does not have an `open_set` flag. Closed queries choose an enrolled individual; open queries apply a rejection threshold to the same similarity scores. Protocol roles keep unknown and novel identities out of encoder training so those capabilities can be measured honestly.

## Input and clip/tracklet policies

```python
import dataset_fixer as df
from dataset_fixer import id

# Choose one source:
data = df.Dataset.open('/data/animals', format='identity_folders',
    layout='{split}/{identity_id}/{clip_id}/{image}')
data = df.Dataset.open('/data/samples.jsonl', format='identity_manifest')
data = df.Dataset.open('/data/tracklet_npzs', format='trex_npz',
    identities='/data/global_id_summary.xlsx')

# Immutable metadata views; pixels stay at their original locations.
data = data.filter(gt_annotated=True, identity_ids=['wolf-1', 'wolf-2'])
data = data.split({'train': .7, 'val': .15, 'test': .15}, group_by='clip_id',
                  classes_must_appear_in=('train',))
data = data.holdout(clip_id='observation-17', identity_ids=['wolf-2'], to_split='test')
data = data.with_sampling(
    balance_by=('identity_id', 'clip_id', 'tracklet_id'),
    samples_per_epoch=50_000,
)
```

Omit `identity_ids` to hold out the entire clip. `group_by=('identity_id', 'clip_id')` groups each individual/clip combination; path callbacks still work. Individual holdouts warn when the clip remains in training for other animals. Group boundaries take precedence over exact split ratios.

`filter(identity_ids=[...])` selects individuals using metadata without opening
images or rewriting archives. It combines with `gt_annotated`, `split`, and
prediction filters. Unknown labels raise an error; an empty list selects no
images. Omit the option to retain all identities.

To test uneven clip/tracklet diversity, restrict selected individuals with a
metadata-only view:

```python
limited = data.limit_coverage({
    "wolf-1": {"clips": 1, "tracklets_per_clip": 1},
    "wolf-2": {"clips": 2},
}, split="train", seed=42)
display(id.stats(limited, by=("identity_id", "split")))
```

Split first to reduce training diversity while preserving validation/test data.
`split="train"` is the default; `split=None` applies one set of caps across all
splits together. Each cap is a positive integer upper bound; omitted caps are
unlimited. Unspecified individuals and other splits remain unchanged. Complete
clips are selected first, then complete tracklets within each selected clip;
all observations in retained groups keep their original order and identifiers.
There is no per-frame thinning or replacement when fewer groups are available.
The seed fixes group selection independently of row order and other identities;
smaller caps select nested subsets. Target identities must have observations in
the chosen scope and non-missing clip IDs; tracklet caps also need tracklet IDs.
The source and pixels are unchanged. The view works with training, `with_sampling`,
and normal identity exports; directory exports also retain the `coverage_limits`
history in their manifest. This works for synthetic and imported identity data.

`classes_must_appear_in=('train',)` requires every observed identity in training;
`('train', 'val', 'test')` requires every identity in all three splits. The default
`()` adds no coverage constraint. Only labeled identities in the selected
`source_splits` count. Whole groups and explicit assignments remain intact; a
class appearing in one clip cannot satisfy a requirement for two splits. Invalid
or impossible requirements raise an error, including conflicting class overlaps.
SciPy's integer solver assigns all groups together: enforce required coverage,
minimize deviation from the requested overall and per-class image fractions,
then prefer broader class coverage among equally good proportional allocations.
Each solver call is bounded to five seconds. If proving
optimal ratios takes longer, the best feasible allocation is retained and one
warning is emitted; required coverage and group boundaries are never relaxed.
The broader-coverage pass is skipped when every class is already required in
every split. Export the resulting dataset to reuse exact assignments: a timed
optimization can return different feasible assignments on different machines.
Overall fractions and the mean per-class fractions have equal weight. Extra
coverage does not override the target proportions; only explicitly required
coverage and indivisible groups can force that tradeoff. Required splits
must have positive ratios, and positive-ratio splits must be nonempty; infeasible
requests raise an error rather than silently omitting validation or test.
Unrequired splits can lack individuals: ID dataset warnings report those
gaps, and `id.stats(data, by=('identity_id', 'split'))` shows the actual counts.
The same option applies to annotation classes in ordinary detection datasets and
is retained when a planned split is exported.

Sampling first balances individuals, then clips within each individual, then tracklets if requested. Images are sampled within the final group using PyTorch's weighted sampler. `samples_per_epoch` controls workload, not a fixed subset. Draws and augmentations change reproducibly each epoch, including after resume.

The batch builder first spreads each identity's pending draws proportionally
across batches, then prefers distinct clips and tracklets within those shares.
This prevents identities with many clips from being consumed early and leaving
identities with fewer clips concentrated at epoch end. It balances repeated
groups when needed to fill the batch and reorders up to eight batches of metadata
at a time, without dropping or replacing any draw or changing the epoch's weighted
selection. This is best effort within that bounded lookahead, not a global optimum
over the entire epoch. Missing clip/tracklet metadata simply disables that preference.
No hard limits apply by default. If explicitly supplied,
`max_per_batch={'tracklet_id': 2}` remains a strict cap and can produce smaller
batches. The sampler version is stored in checkpoints: old sequential-order runs
can initialize a new run with `weights=`, but cannot silently resume with a different
batch ordering algorithm. Proportional ordering with complete-batch workloads is
recorded as `diverse_v3`; older checkpoints remain valid for prediction and
`weights=` initialization, but require a new run for the new workload policy and
cumulative source-coverage tracking.

CSV/JSONL rows require `identity_id` (null is allowed) and either `image_path`/`path`, or `archive`, `array`, `row`. Optional fields include `split`, `clip_id`, `tracklet_id`, `local_id`, `frame_index`, `notes`, and `original_id`. Paths are relative to the manifest. Unsplit rows belong to `train`. A JSONL `crop: [left, top, right, bottom]` is an optional deterministic pixel crop. Use `data.export(destination=..., format='identity_manifest')` to save a view without copying pixels.

`tracklet_id` always identifies the same source tracklet across dataset views.
Import qualifies native tracklet IDs with their clip ID, so repeated native
numbers in different clips cannot collide. Filtering, splitting, cropping,
sampling, and export/reload preserve that qualified ID exactly; it does not
depend on row order, identity labels, or destination paths. Keep it intact when
writing a derived manifest. With no clip metadata, supplied tracklet IDs must
already distinguish tracks across the input. Missing IDs remain missing.
Reopen older saved datasets to upgrade their IDs without rewriting image files.

TRex imports use uncompressed NPZ memory maps and the full observation directory name as `clip_id`. Workbook mappings come from the `Global ID Summary` sheet; confidence-qualified assignments are accepted and their original notes retained. Unmapped rows remain unlabeled, are excluded from supervised training, and are never considered unknown by inference or evaluation.

`Dataset.open(..., trim_black_borders=True)` applies to **all identity sources**: folder PNGs, manifests, and NPZs. Purely black outer rows/columns are trimmed lazily after any explicit source crop and before augmentation. Non-black background and interior black pixels remain; all-black images keep their original dimensions. New identity inputs enable trimming by default. With the default `None`, saved manifest settings are preserved; explicit `True`/`False` overrides them. The flag is part of source references/cache identities and is saved in manifests, without rewriting images. Stored width/height describe the original source crop; the actual trimmed dimensions are determined on pixel access. Disable trimming if a custom augmentation relies on fixed coordinates in the original padded image.

TRex `frame_segment_indexes` are global across archive parts, whereas `tracklets` tables can be part-local and truncated mid-tracklet. The importer namespaces native segment indexes by clip/local tracking ID, preserves the bounds tables in dataset metadata, and reports inconsistencies. It does not guess new tracklets from gaps between sampled frames. Some exporter versions can conservatively group multiple actual tracklets under one segment ID; use corrected manifest `tracklet_id` values when exact bounds are necessary.

Metadata and sampler weights scale with observation count. Pixels are decoded only for the current batch; worker prefetch and memory-map readers are bounded. There is no permanent subsampling of a large archive.

## Crops, augmentations and training

```python
import albumentations as A

# Explicitly interpret annotation classes as individual labels, or use
# identity='identity_id' / a callable (sample, annotation) -> identity label.
crops = id.from_dataset(detection_dataset, identity='class', context_fraction=.1)

# A body-part crop runs on the original, orientation-normalized pixels.
augmentations = [
    A.Crop(x_min=0, y_min=0, x_max=128, y_max=80),
    A.ColorJitter(p=.3),
]

with df.TrainingSession(wandb=df.WandbConfig(project='wolf-id')) as session:
    result = id.train(
        data, backbone='resnet18', augmentations=augmentations,
        config=df.TrainingConfig(
            resolution=128, epochs=30, batch_size=64, workers=4,
            augmentation_samples=12,
            backend_options={'embedding_dim': 256, 'loss': 'subcenter_arcface'},
        ), session=session,
    )
    report = result.evaluate(split='test')
```

`context_fraction=.1` adds 10% of box width on each horizontal side and 10% of height on each vertical side, clipped to the image boundary. Its default is zero. Already-cropped images cannot recover pixels outside their source. Context cropping happens before augmentation; it does not rotate or resize the image. Letterboxing in the display renderer is unrelated to crop context.

Augmentation lists, Albumentations `Compose`, and serialized pipelines all use the existing serializer. `None` uses `id.default_augmentations()`: border cropping up to 10% per side (p=.35), proportional affine scale 0.8–1.6, rotation ±5°, translation ±18% (p=.7), HSV changes (±18 hue, ±20 saturation, ±50 value; p=.4), explicit brightness and contrast (both ±.15; p=.5), mild 3–5 pixel blur (sigma .1–.6, p=.1), and two random-pixel noise erasers (2–12% area at p=.35; 2–6% at p=.2). Grayscale runs last (p=.25), so it also converts any noise patches. There are no flips or shear. More aggressive affine ranges can be supplied in a custom list; zoom and translation can cut off parts of a tightly cropped animal. The function returns a normal editable Albumentations list; `[]` disables random augmentation. Spatial crops and transforms are supported. They must return nonempty RGB `uint8` pixels: the model applies final resizing and normalization exactly once. Do not add `Normalize` or tensor conversion to the augmentation pipeline. Validation/prediction preprocessing is deterministic. A fixed body-part focus needed during prediction should be stored in the manifest's `crop` or an equivalent dataset crop view; stochastic training crops are not silently applied during inference.

Startup grids use the same augmentation builder and preprocessing as training, with isolated randomness. They contain original/augmented pairs, identity/clip captions, source references and the transforms actually applied to each image (brightness/contrast signs show the sampled direction). Transform probability decisions consume successive draws from shared Albumentations RNGs; they are not reset to identical child streams for every image. A small preview is a random sample, so it need not contain every transform. The existing renderer saves the local image, bundles it and logs that same file to W&B. Native detection trainers use their existing augmentation adapters for these grids too.

Predefined backbones: `resnet18`, `efficientnet_v2_s`, `convnext_tiny`. Arbitrary timm names are accepted, for example `timm:vit_tiny_patch16_224`. For frozen-feature training set `freeze_backbone=True`; for the classification baseline set `loss='cross_entropy'`. The default is normalized embeddings with PyTorch Metric Learning's sub-center ArcFace loss. Advanced Lightning options, such as `precision`, go under `backend_options={'trainer': {...}}`. One device per run is currently supported to preserve the defined sampling distribution and batch caps.

### Epoch workload and CUDA

Training rounds `samples_per_epoch` up to the nearest batch-size multiple and
prints one startup notice if it changes. For example, 5,000 becomes 5,056 at
batch size 64, or 5,120 at batch size 256. Without an explicit workload, the
eligible training row count is rounded in the same way. Extra draws use the
same weighted sampler and receive fresh augmentations. The immutable dataset
and requested configuration stay intact; run metadata records both
`requested_samples_per_epoch` and `effective_samples_per_epoch`.

Without explicit batch caps, the progress bar reports the complete batch count:
50,000 requested samples at batch size 64 become 50,048 samples in 782 full
batches. Explicit `max_per_batch` caps still take precedence, can produce smaller
batches, and keep their count unknown until sampled. Validation is never padded
or repeated to fill a batch.

ID training uses Lightning's native Rich display when available, otherwise tqdm. Native `trainer.enable_progress_bar` and `trainer.enable_model_summary` switches still apply; native callback instances can be passed in `backend_options['trainer']['callbacks']`. Flushed startup messages identify W&B initialization, the backup-directory check, augmentation rendering, sampler preparation, monitor-cohort selection and validation sanity checks. Timings separate dataset preparation from actual training. There are no additional per-batch printouts.

Lightning receives an example input at the effective resolution for its native
FLOPs summary. This measures one image's encoder/projection forward pass, not a
training batch or GPU utilization. The identity head/loss are outside that
forward pass and may show zero. Lightning's `bf16-mixed` model-summary warning
concerns its parameter-memory estimate; it does not disable mixed precision.

By default every eligible validation image is evaluated every epoch. For a large dataset, set `backend_options={"validation_samples": 8192}` to select a fixed validation subset balanced across identities, clips and tracklets. It is selected once from the seed, is unchanged across epochs/resume, and is used for validation loss and checkpoint selection. The limit must cover every validation identity. Startup output and training metadata report selected/available counts. `None` retains full validation. Final `result.evaluate(split="test")` still uses the full test query cohort.

To visit the rest of validation over successive epochs, also set
`validation_sampling="cycle"` in `backend_options`. Each identity advances through
a seeded order of its observations, interleaved across clips and tracklets;
smaller identities wrap independently so identity shares stay balanced. The
epoch number determines the selection, including after resume. This eventually
covers all eligible validation sources when complete validation passes run.
Only metadata is indexed; pixels remain lazy. The default `"fixed"` keeps the
original behavior, and `validation_samples=None` still evaluates all images.
Cycling changes the images underlying validation loss and checkpoint selection,
so adjacent losses are not measured on identical observations. The embedding
monitor remains fixed. Use fixed or full validation for controlled model comparisons.

On an NVIDIA L4, enable native Lightning mixed precision and PyTorch's faster float32 matmul policy explicitly:

```python
import torch
if DEVICE.startswith("cuda"):
    torch.set_float32_matmul_precision("high")

# Merge these with your existing TrainingConfig.backend_options:
backend_options = {
    "validation_samples": 8192,
    "trainer": {"precision": "bf16-mixed"} if DEVICE.startswith("cuda") else {},
}
```

CUDA loaders pin host memory for transfers; CPU/MPS loaders do not. Keep extracted data on local scratch. Worker count should fit the available CPUs; increasing it beyond that may slow loading. These settings reduce work and enable native acceleration, but GPU throughput must be measured on the target machine.

### Locating an input bottleneck

Use Lightning's native `PyTorchProfiler` under `backend_options['trainer']['profiler']`.
It records CPU activity and, on CUDA, GPU kernels/transfers. Named input ranges
are enabled only when a profiler is configured: `id/sampling`,
`id/source_lookup`, `id/read_image`, `id/augment`, and `id/preprocess`.
No separate profiling implementation or GPU synchronization is added to normal training.

First profile a short run with the actual worker count. Look for GPU idle gaps
aligned with Lightning's `train_dataloader_next` range; this includes sampling,
worker/IPC waiting, and collation. Sampling runs in the main process. Worker
image/augmentation ranges are not collected by the parent profiler: repeat with
`workers=0` to break down those CPU stages. That serial run diagnoses component
costs, not production throughput. `id/read_image` includes source cropping and
black-border trimming; `id/preprocess` includes final resize and normalization.
CPU and GPU work overlap, so their durations must not be added as if sequential.

```python
from lightning.pytorch.profilers import PyTorchProfiler
import torch

profiler = PyTorchProfiler(
    dirpath='/tmp/id-profile', filename='input', record_module_names=False,
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=4, repeat=1),
    record_shapes=False, profile_memory=False, with_stack=False,
    acc_events=True, row_limit=40, sort_by_key='cpu_time_total',
)
# Merge into config.backend_options['trainer'] for a disposable, one-epoch run:
trainer_options = {
    'profiler': profiler, 'limit_train_batches': 12, 'limit_val_batches': 1,
    'num_sanity_val_steps': 0, 'enable_model_summary': False,
}
# Use a fresh output directory, augmentation_samples=0, embedding_samples=0,
# and wandb=False for the profiling run. Keep its model separate from real runs.
# After fitting: print(profiler.summary()); Chrome trace JSONs are in dirpath.
```

Sampler selection uses successive minima/maxima instead of sorting the entire
lookahead for every image, preserving the exact draws, ordering and resume policy.
The process-local NPZ LRU retains at most 32 memory mappings to avoid repeatedly
reopening diverse clips. Archives remain lazy; mapping more files can increase
resident pages and file handles, but does not eagerly decode whole image arrays.
Loader workers use single-threaded OpenCV while retaining Lightning worker
seeding, preventing nested OpenCV thread pools from competing for the same CPUs.

Tune from the measured bottleneck: local storage/caching for read time, worker
count and crop-first pipelines for augmentation time, and batch size/precision
when GPU kernels dominate. The default augmentation pipeline already crops
early, combines geometry in one affine transform and retains uint8 pixels.
Custom pipelines keep their ordering and original crop coordinates; resizing
before a custom body-part crop must be an explicit pipeline choice.

### Custom models through the same API

```python
custom = MyEncoder()  # forward(images) returns an N × D feature tensor
id.register_backbone(
    'my-encoder', factory=MyEncoder, model=custom, feature_dim=512,
    preprocessing={'resolution': 128, 'mean': [.5]*3, 'std': [.5]*3},
)
result = id.train(data, backbone=custom, config=df.TrainingConfig(epochs=20))
# backbone='my-encoder' reconstructs a fresh instance using the same factory.
```

The zero-argument factory must reconstruct the exact architecture. Register it before loading in another process. Models are saved as state dictionaries, not pickled model objects. `embedding_dim=0` retains the custom model's output dimension and normalizes those features without adding a learned projection.

## Initialization, resume and publication

```python
base = df.Model('wandb:team/wolf-id/runs/run123')
result = id.train(new_data, weights=base, config=df.TrainingConfig(epochs=20))
model = df.Model(result.bundle.path)  # local checkpoint, bundle or W&B reference
```

`weights=` restores the encoder/projection and creates a new identity head and optimizer. The roster may change. Architecture and preprocessing are inferred; incompatible overrides fail. `resume=` restores full Lightning state and verifies identities, data fingerprint, sampling, batch size, seed, transforms and architecture. Changing the validation subset or augmentation RNG policy requires a new run. Checkpoints created before the independent augmentation RNG fix can still be loaded for inference or `weights=` initialization, but cannot strictly resume under the corrected policy. Preserve the previous best checkpoint alongside a standalone resume checkpoint, or use a full-state bundle. Native checkpoints remain local; `wandb_contents='full'` enables remote resume.

With W&B enabled, ID defaults to `CheckpointConfig(wandb_upload='best', keep_wandb_versions=1, wandb_contents='best')`. The first best and every validation improvement upload immediately. A replacement must be confirmed before the previous managed version is pruned; protected artifacts are retained. Session completion republishes the selected best with reports and grids before finishing an owned run. Failed publication preserves local recovery files and reports incomplete finalization. Use a `TrainingSession` block when evaluation/calibration should be included in the final publication.

W&B uses the SDK defaults for mode and console capture. Explicit
`WandbConfig(init={'settings': ...})` settings, environment settings and
caller-owned runs are preserved. Ordinary stdout/stderr can be captured by W&B;
Rich's notebook HTML display bypasses those streams, including prints redirected
through an active Rich live display. To keep native Rich bars while making their
output available to both the notebook and W&B, select Rich's terminal rendering
through the native controls before training:

```python
import rich
from lightning.pytorch.callbacks import RichProgressBar

console = {'force_jupyter': False, 'force_terminal': True}
rich.reconfigure(**console)  # also applies to the startup Rich model summary
# Include this in config.backend_options['trainer']:
trainer_options = {
    'callbacks': [RichProgressBar(console_kwargs=console, leave=True)],
    'enable_progress_bar': True,
}
# W&B needs no custom console setting:
wandb_config = df.WandbConfig(project='wolf-id')
```

These remain Lightning callbacks; its enable/disable, refresh and summary
controls still apply. Structured training metrics do not depend on console
capture. Notebook HTML widgets themselves are not copied into W&B console logs.

During ID training, a scoped compatibility filter suppresses only Lightning's
internal `LeafSpec` constructor `FutureWarning` from its pytree module. Other
warnings remain visible, and the filter is removed when fitting exits. This is
an upstream PyTorch/Lightning deprecation, independent of W&B uploads.

## Enrollment and query mode

```python
gallery = model.enroll(reference_crops, samples_per_identity=20)
gallery.save('gallery.npz')
closed = model.predict(query_crops, gallery=gallery)  # mode='closed'

rejection = model.calibrate(validation_queries, gallery=gallery,
    unknown_identities=validation_unknown_ids, target_fpir=.01)
opened = model.predict(query_crops, gallery=gallery, mode='open', rejection=rejection)
```

Enrollment selects references across clips/tracklets. New individuals can be added by enrolling a new reference dataset without retraining the encoder. Maximum cosine similarity over each identity's references determines its score. Scores are not probabilities. `model.embed(...)` returns the same `PredictionResult`, with `record.embedding`; identified records additionally expose `identity_id`, `similarity` and `rejected`. `predict(..., track_by='tracklet_id')` optionally averages features within supplied tracking groups. Ground-truth global identity labels cannot be used to form prediction groups.

`predictions.visualize(samples=12, columns=4)` shows one query image per card,
with ground truth and a prediction colored green for correct, red for incorrect
(including tied top scores), or gray when correctness is unavailable. The footer
shows the strongest competitor, the top-two similarity margin, and cosine
similarity. A rejected query is labeled `unknown (rejected)` and retains its best
gallery match in the footer; rejection is correct only when the labeled identity
is absent from the gallery. Unlabeled queries and embedding-only outputs stay
neutral. Margins require saved rankings; no inference is rerun for visualization.
Model details appear once above the grid; `show_model_slugs=False` hides them.

Gallery and calibration fingerprints bind to the exact model. A changed model or gallery requires recalibration. `Gallery.load(...)` and `RejectionCalibration.load(...)` restore saved artifacts. Embeddings use the existing verified prediction cache; gallery/rejection changes reuse the encoder output.

Reference selection defaults to the existing seeded, clip/tracklet-balanced random
selection. Opt into representative images with `kmedoids` (included in the `id`
extra); this changes enrollment, not model training:

```python
protocol = id.Protocol(
    train_data,
    gallery_samples=20,
    gallery_selection="medoids",
    gallery_candidates=256,  # Maximum decoded candidates PER identity.
)
result = id.train(protocol, config=config, session=session)
report = result.evaluate(split="val")  # Uses the protocol's enrollment policy.
gallery = result.model.enroll(protocol)

# Also works with a loaded model and an independent labeled reference dataset:
gallery = model.enroll(reference_crops, samples_per_identity=20,
                       selection="medoids", candidates_per_identity=256)
display(gallery.summary)
gallery.visualize(samples=12, columns=3)
```

For medoids, the protocol preserves eligible enrollment candidates separately in
`reference_candidates`, and excludes their source observations from queries before
model-dependent selection. `references` retains the historical balanced subset;
pass the **protocol** to `enroll()` to use the configured selector and candidate pool.
Changing a balanced protocol to medoids requires constructing a new protocol with
that policy so its query exclusions are established first.

Candidates are balanced across clips and tracklets, embedded with deterministic
preprocessing, and deduplicated by pixels before final selection. The existing
hierarchical sampler determines reference quotas across clips and tracklets;
FasterPAM chooses cosine-distance medoids **within those groups**, rather than
letting a large clip dominate a global clustering. Distance matrices are bounded
by the candidate count, and clustering uses one CPU thread. If the pool contains
too few unique images, the gallery is smaller and `gallery.summary.shortfall`
reports the deficit. Balanced selection keeps its previous selection and
post-selection deduplication behavior for comparability.

The gallery NPZ stores the seed, settings, candidate source references, selected
tracklet metadata, and per-identity counts. Training-result evaluation also puts
`identity/enrollment.csv` into the bundle. `gallery.visualize()` delegates to the
existing dataset renderer and requires the original reference images to remain
accessible. These reports add **no W&B metrics or automatic gallery plots**.

The epoch embedding monitor keeps its existing cohort, scoring, and W&B keys by
default. With medoids enabled it applies the same selector to the training half
of its fixed, bounded cohort; it does not decode an extra candidate pool each
epoch. Its rank metric remains a small-cohort diagnostic, not the full protocol
evaluation. Prediction, evaluation, monitoring, and Captum share maximum-cosine
identity scoring. No top-k voting, prototypes, or reranking are enabled.

## Evaluation roles

```python
protocol = id.Protocol(
    data,
    unknown_identities={'val': validation_unknown_ids, 'test': test_unknown_ids},
    novel_identities={'val': validation_new_ids, 'test': test_new_ids},
    gallery_samples=20,
)
# id.Protocol.open_set(...) is an equivalent convenience constructor.
with df.TrainingSession(wandb=df.WandbConfig(project='wolf-id')) as session:
    result = id.train(protocol, session=session)
    rejection = result.calibrate(target_fpir=.01)
    report = result.evaluate(split='test', rejection=rejection)
```

All novel/unknown identities are excluded from encoder training. Their validation and test roles must be disjoint. The protocol fixes reference/query membership and follows the supplied dataset split: enrollment uses training observations, including shared clips when the dataset was split by tracklet or image. No additional switch is needed. Shared clips are recorded in protocol limitations. Novel identities without explicit training-partition enrollment observations still reserve a separate gallery clip from their query partition and need at least two clips. Shared source observations/derived crops across training and evaluation are rejected. Identical reference/query pixels are excluded during evaluation.

To compare clip separation with tracklet separation, split the same selected
data by tracklet instead of clip. Complete tracklets stay together, while
different tracklets from a clip may appear in different splits:

```python
train_data = labeled.filter(identity_ids=keep_ids).split(
    {"train": .7, "val": .15, "test": .15},
    group_by="tracklet_id",
    classes_must_appear_in=("train", "val", "test"),
    seed=7,
).with_sampling(
    balance_by=("identity_id", "tracklet_id"),
    samples_per_epoch=SAMPLES_PER_EPOCH,
)
protocol = id.Protocol(train_data, gallery_samples=20)
```

All three splits remain present. This sampling balances identities, then their
tracklets; include `clip_id` between them only to additionally equalize clips.
Keep your existing sampling configuration if you want to isolate just the split
change. Use a fresh training run with the same model settings. High scores with
shared clips measure recognition in familiar clips, not performance on new clips.

Reports distinguish known individuals, newly enrolled individuals unseen during training, and explicit unknowns absent from the gallery. They include Rank 1/5, macro identity recall, image-retrieval mAP, correct identification at the rejection threshold, and achieved test FPIR, both per crop and balanced over identity/clip groups. Missing labels are excluded. Tied identity similarities count conservatively against ranking. PyTorch Metric Learning computes image-retrieval mAP.

Calibration uses scikit-learn's complete ROC thresholds over maximum gallery scores from unknown validation queries, choosing the most permissive attainable point that does not exceed the target FPIR. Ties remain together. Reports distinguish requested and attained FPIR and flag limited independent clip coverage; correlated frames are not independent calibration evidence. Bob was intentionally replaced because its NumPy 1.x requirements conflict with this framework's NumPy 2/OpenCV requirements.

`TrainingResult.evaluate()` evaluates the complete fixed ID query cohort; the detection-only `samples` limit does not subsample it. `plots` controls the number of prediction examples (`0` disables the grid). Evaluation writes the shared `ComparisonResult`, frozen cohort, query CSV, metrics and protocol JSON, and figures through the existing report renderer. Training provenance currently checks clip membership and source-reference overlap; it does not certify pixel-level duplicate exclusion against the entire training corpus under renamed/copied files.

Evaluation also works after leaving `TrainingSession`, when its W&B run has finished, or without W&B. The report and metrics remain local; a finished run is not reopened. Missing or unavailable W&B logging does not prevent training or evaluation. Calling `session.finish()` again packages new reports and refreshes a configured filesystem backup. Automatic Colab disconnect requires confirmed persistent storage of the updated results; a previous model upload alone does not cover new local reports.

Embedding extraction and retrieval scoring show completed-image counts, elapsed
time and progress bars. Retrieval metrics use bounded batches of 128 queries;
PyTorch Metric Learning still computes each image's AP before the existing crop
and identity/clip summaries. Gallery tensors are reused across batches.
Session evaluation also logs enrollment, embedding, metric and report progress
under `evaluation/<split>/`, using elapsed seconds as its own step axis. The run
summary's `evaluation/<split>/stage` becomes `complete` only after report writing.
These metrics remain available when console capture is off. Direct
`model.embed(..., progress=False)` and `model.predict(..., progress=False)` silence
their local progress output. Training's `samples_per_epoch` and
`validation_samples` do not reduce this final evaluation cohort.

## Implementation map

Only three public classes are added: `id.Protocol`, `id.Gallery`, and `id.RejectionCalibration`. Configuration, datasets, models and results remain the existing `Dataset`, `TrainingConfig`, `TrainingSession`, `TrainingResult`, `Model`, `ImagePrediction`, `PredictionResult` and `ComparisonResult` classes.

| Module | Internal implementation and reuse |
| --- | --- |
| `id.data` | `ImageRef` and `SampleIndex`; existing `Sample`, `Annotation`, `DatasetMetadata` and shared group-allocation policy |
| `id.training` | `_IdentityModule`, `_IdentityDataModule`, `_TrainingBridge`; Lightning optimization/checkpointing connected to the existing training session |
| `id.training` | `_Pixels` and `_DiverseBatches`, private PyTorch Dataset/Sampler adapters for lazy reads and optional batch caps |
| `id.model` | Backbone registration, shared augmentation serializer, embedding and query functions; existing model source resolver and `PredictionCache` |
| `id.gallery` | Reference selection and compatible gallery persistence |
| `id.evaluation` | Protocols/calibration/library metrics; shared frozen cohorts, comparison results, image grids, report tables and static rendering |

No parallel plotting engine, identity-specific model/result hierarchy, or custom optimization loop is introduced.

## Monitoring and inspection

Every epoch logs one scalar, `dataset_coverage/percent`: the percentage of
distinct eligible train-or-validation source images consumed so far. Plot it
against `epoch` in one W&B line panel. Sources shared between those splits count
once in this combined percentage. Test images are excluded. Eligibility is
determined after identity/unlabeled filtering and before epoch/validation limits;
validation includes identities supported by the training head. Raw split counts
remain in local metadata/CSV rather than additional W&B scalar streams.

A physical source is a file path, or an NPZ path/array/row. Repeated draws,
augmentations, and derived crops of that source count once; copied files at
different paths are distinct (no expensive pixel-hash deduplication). Counters
advance only after completed batches. Prefetch, sanity checks, augmentation and
embedding previews do not count. A fixed validation subset therefore produces
flat validation coverage after the first epoch; use `validation_sampling="cycle"`
to advance through validation, or `validation_samples=None` to reach 100% in one
complete pass. Training draws are weighted with replacement: coverage is
nondecreasing, but rare images can take many epochs to appear and no finite
epoch count guarantees 100%. Full resume restores compact
coverage bitmaps through Lightning's DataModule checkpoint state; `weights=`
starts coverage for the new run.

`dataset_coverage/chart` is a two-bar image per epoch, rendered with the existing
chart machinery and browsable through W&B's media history. Each bar uses its
own split's eligible source pool as the denominator. The chart and combined
percentage are the only two W&B coverage fields. The latest image and `coverage.csv`
enter the bundle. Chart failures disable only the coverage image; counts and
training continue. Coverage metadata stays on the CPU and requires no per-batch
GPU synchronization. Disabling embedding plots does not disable these counters.

Every epoch renders a scikit-learn t-SNE projection of a fixed, balanced cohort of up to 512 training/validation crops, using the **current** encoder and deterministic preprocessing. Selection and monitoring preserve training RNGs and model mode. Identity colors and cohort membership remain fixed; refitted 2D coordinates are not a longitudinal distance metric. Set `backend_options={"embedding_samples": 512, "embedding_method": "tsne"}` (`"pca"` is also supported; samples `0` disables it).

`embedding_samples` is the total plot budget, independent of training batch size
and `validation_samples`. Half is allocated to train and half to validation
(an odd extra sample goes to train), balancing identities, clips and tracklets
within each split. For example, 600 requests up to 300 train and 300 validation
images. A split with fewer available images contributes fewer; unused capacity
is not transferred to the other split. The selected images stay fixed across
epochs. Requests above 2,048 are capped with a single `⚠` notice, bounding the
projection and pairwise-score work. Monitor inference starts in chunks of `min(batch_size, 64)`;
this chunk size does not change the number of plotted points. W&B's
`embeddings/samples` records the actual cohort size.

Monitoring is optional: a recoverable memory failure skips that epoch's image
and halves the cohort and inference chunk for the next attempt. The reduced
cohort stays fixed after recovery; its metrics are not directly comparable with
the earlier cohort. Other diagnostic failures disable further monitoring while
training, checkpointing and publication continue. Each fallback reason prints
one `⚠` notice and is recorded in bundle metadata and W&B's `embeddings/notice`
summary. `embeddings/skipped=1` marks a failed/disabled diagnostic epoch. User
interrupts still propagate. Setting samples to zero avoids cohort selection.

W&B receives one image per epoch under `embeddings/tsne`, giving a single media panel with step history/slider. `epoch` is the shared zero-based axis; captions display one-based epoch numbers. Monitor `train/loss`, `val/loss`, `val/monitor_rank1`, `optim/lr`, `time/epoch_seconds`, and `time/monitor_seconds`. The diagnostic Rank-1 uses only the small fixed cohort and is separate from final evaluation. Selection remains validation loss. The cohort manifest, latest coordinates and image enter the checkpoint bundle; local epoch images and W&B history retain earlier views.

Before an epoch finishes, `progress/batch`, `progress/epoch` (zero-based),
`train/batch_loss`, `train/batch_size` and `train/batch_identities` report completed
batches against `progress/global_step` (optimizer steps). They update on the
first and last batches, and every `backend_options['trainer']['log_every_n_steps']`
batches (default 50), or after 15 seconds between updates. Zero retains only
first/last updates. Batch loss is the loss for that batch before
gradient-accumulation scaling; `train/loss` remains the sample-weighted epoch
aggregate. Size/identity counts help investigate tail-batch loss spikes.

```python
display(id.stats(data))  # samples, labeled/unlabeled, individuals, clips, tracklets
display(id.stats(data, by=("identity_id", "clip_id")))
display(id.stats(data, by="clip_id"))
data.report()  # canonical report API, metadata only

id.visualize_attention(model, queries, gallery=gallery,
                       contrast=True, samples=12, columns=4, destination="attention.png")
```

The attention visualizer calls [Captum Occlusion](https://captum.ai/api/occlusion.html) directly. It explains maximum cosine similarity to the selected identity; by default, it subtracts the maximum over all competing identities. Each occluded patch uses the preprocessing channel mean. Red supports the target score/margin and blue opposes it. Each sample has one overlay card; `columns` controls the grid width (default 2). By default, all cards share one symmetric effect scale, so equally strong colors represent equally strong effects within the grid. Pass `color_scale="per_image"` to reveal weak patterns independently. Each card shows `Max |Δ|`, the actual maximum absolute effect before display scaling. An entirely zero map remains transparent. Equal colors under per-image scaling do not imply equal effect magnitudes.

Cards show ground truth and prediction. Prediction text alone indicates correctness: green for a correct unique top prediction, red for an incorrect prediction or tied top scores, and gray without ground truth. The footer names the original input's strongest competing identity and shows the target's signed similarity margin and cosine similarity; these are not probabilities. `identity_id=` explains a specified enrolled individual and can produce a negative margin; otherwise the target is the model's own top prediction. The competitor may change during occlusion. With a single enrolled identity and `contrast=False`, only similarity is shown. Raw attribution arrays and metadata (including ground truth, prediction, correctness, competitor, margin, actual effect magnitude, and chosen color scale) are saved alongside the grid. This measures sensitivity to masking, not anatomical ground truth or internal transformer attention. The method supports custom encoders without layer-specific hooks.

To summarize the enrolled references, make two separate plots. The averaging limit only affects the visualization; the enrolled gallery and prediction scores remain unchanged:

```python
gallery.visualize_attention(
    model,
    samples_per_identity=4,
    contrast=False,
    color_scale="per_image",
    columns=3,
    destination="reference_attention.png",
)
gallery.visualize_attention(
    model,
    samples_per_identity=4,
    contrast=True,
    color_scale="per_image",
    columns=3,
    destination="reference_contrast.png",
)
```

Each call inspects up to four references per identity here, balanced across clips and tracklets, and shows one card per identity (`samples=` optionally limits displayed identities). For each reference, Captum explains similarity to its own identity. Its own source and any exact pixel duplicates are excluded from scoring; at least two distinct reference images per identity are required. Scoring still uses the complete remaining gallery, regardless of `samples_per_identity`.

With `contrast=False`, the plot shows each identity's mean raw signed attribution map. With `contrast=True` (the method's default), it shows that mean minus the equally weighted mean of all other identities' mean maps. Each other identity contributes equally, even if its reference count differs. Other identities still participate when `samples=` limits which cards are displayed. This compares evidence across images; it differs from query `id.visualize_attention(..., contrast=True)`, which explains own similarity minus strongest-rival similarity on the same query image.

In the mean plot, red supports the wolf's own similarity and blue opposes it. In the difference plot, red means greater signed supporting evidence at that location for this wolf than for the others; blue means less. Differences use raw attribution units, never differences between independently normalized heatmaps. The backdrop is a real embedding medoid chosen from the inspected images by the existing `kmedoids` implementation. Each card includes the count and actual peak effect. These spatial averages can blur across poses and can cancel positive/negative effects; they do not align body parts or prove which anatomical features are unique.

The NPZ includes the displayed `attribution_*` maps, the `mean_*` identity maps and individual `reference_*` maps. JSON identifies all contributing sources, other-identity map keys, scores, self-exclusions, the medoid source and effect magnitudes. To improve the enrolled reference set itself, the existing `model.enroll(..., selection="medoids")` option still applies.

ID training automatically generates both figures from the selected best checkpoint, displays them locally, includes their PNG/NPZ/JSON files in the final bundle and backup, and logs the two images to `inspection/reference_attention` and `inspection/reference_contrast` in the session-owned W&B run. This adds no scalar training metrics. The default averages up to four references per identity with per-image display scaling. Customize it without changing enrollment:

```python
backend_options={
    # Other training settings...
    "reference_attention": {"samples_per_identity": 2, "color_scale": "shared"},
}
```

Set `reference_attention=False` to disable automatic end-of-training plots. Diagnostic failures, including insufficient distinct references, leave training results and checkpoints intact and report a warning. Without an active W&B run, figures remain local.

The [wolf Colab notebook](../notebooks/wolves_identity_colab.ipynb) installs the optional dependencies, extracts Drive NPZs into `/mnt/local-scratch`, reports identity/clip coverage, enrolls all labeled wolves, and trains on CUDA (or local MPS). `DATASET_FIXER_CACHE_DIR` directs the shared cache into scratch. Drive backups retain resumable state and final reports. Install from a published revision containing these changes; staging alone does not update GitHub.
