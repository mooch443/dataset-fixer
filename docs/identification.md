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
data = data.split({'train': .7, 'val': .15, 'test': .15}, group_by='clip_id',
                  classes_must_appear_in=('train',))
data = data.holdout(clip_id='observation-17', identity_ids=['wolf-2'], to_split='test')
data = data.with_sampling(
    balance_by=('identity_id', 'clip_id', 'tracklet_id'),
    samples_per_epoch=50_000,
)
```

Omit `identity_ids` to hold out the entire clip. `group_by=('identity_id', 'clip_id')` groups each individual/clip combination; path callbacks still work. Individual holdouts warn when the clip remains in training for other animals. Group boundaries take precedence over exact split ratios.

`classes_must_appear_in=('train',)` requires every observed identity in training;
`('train', 'val', 'test')` requires every identity in all three splits. The default
`()` adds no coverage constraint. Only labeled identities in the selected
`source_splits` count. Whole groups and explicit assignments remain intact; a
class appearing in one clip cannot satisfy a requirement for two splits. Invalid
or impossible requirements raise an error, including conflicting class overlaps.
SciPy's integer solver assigns all groups together: enforce required coverage,
minimize deviation from the requested overall and per-class image fractions,
then prefer broader class coverage among equally good proportional allocations.
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

The batch builder prefers distinct clips and tracklets, then balances repeated
groups when needed to fill the batch. It reorders up to eight batches of metadata
at a time, without dropping or replacing any draw or changing the epoch's weighted
selection. This is best effort within that bounded lookahead, not a global optimum
over the entire epoch. Missing clip/tracklet metadata simply disables that preference.
No hard limits apply by default. If explicitly supplied,
`max_per_batch={'tracklet_id': 2}` remains a strict cap and can produce smaller
batches. The sampler version is stored in checkpoints: old sequential-order runs
can initialize a new run with `weights=`, but cannot silently resume with a different
batch ordering algorithm.

CSV/JSONL rows require `identity_id` (null is allowed) and either `image_path`/`path`, or `archive`, `array`, `row`. Optional fields include `split`, `clip_id`, `tracklet_id`, `local_id`, `frame_index`, `notes`, and `original_id`. Paths are relative to the manifest. Unsplit rows belong to `train`. A JSONL `crop: [left, top, right, bottom]` is an optional deterministic pixel crop. Use `data.export(destination=..., format='identity_manifest')` to save a view without copying pixels.

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

Without explicit batch caps, the progress bar reports the complete batch count: `samples_per_epoch=50_000` and `batch_size=64` produce 782 training batches. Explicit `max_per_batch` caps can produce smaller batches and keep the count unknown until sampled.

ID training explicitly uses Lightning's tqdm progress bar and plain model summary, avoiding an additional Rich live-rendering layer in notebooks with W&B console capture. Native `trainer.enable_progress_bar` and `trainer.enable_model_summary` switches still apply. Flushed startup messages identify W&B initialization, the backup-directory check, augmentation rendering, sampler preparation, monitor-cohort selection, validation sanity checks and the first completed training batch. Timings separate dataset preparation from actual training.

By default every eligible validation image is evaluated every epoch. For a large dataset, set `backend_options={"validation_samples": 8192}` to select a fixed validation subset balanced across identities, clips and tracklets. It is selected once from the seed, is unchanged across epochs/resume, and is used for validation loss and checkpoint selection. The limit must cover every validation identity. Startup output and training metadata report selected/available counts. `None` retains full validation. Final `result.evaluate(split="test")` still uses the full test query cohort.

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

## Enrollment and query mode

```python
gallery = model.enroll(reference_crops, samples_per_identity=20)
gallery.save('gallery.npz')
closed = model.predict(query_crops, gallery=gallery)  # mode='closed'

rejection = model.calibrate(validation_queries, gallery=gallery,
    unknown_identities=validation_unknown_ids, target_fpir=.01)
opened = model.predict(query_crops, gallery=gallery, mode='open', rejection=rejection)
```

Enrollment selects references across clips/tracklets. New individuals can be added by enrolling a new reference dataset without retraining the encoder. Maximum cosine similarity over each identity's references determines its score. Scores are not probabilities. `model.embed(...)` returns the same `PredictionResult`, with `record.embedding`; identified records additionally expose `identity_id`, `similarity` and `rejected`. `predict(..., track_by=('clip_id','tracklet_id'))` optionally averages features within supplied tracking groups. Ground-truth global identity labels cannot be used to form prediction groups.

Gallery and calibration fingerprints bind to the exact model. A changed model or gallery requires recalibration. `Gallery.load(...)` and `RejectionCalibration.load(...)` restore saved artifacts. Embeddings use the existing verified prediction cache; gallery/rejection changes reuse the encoder output.

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

All novel/unknown identities are excluded from encoder training. Their validation and test roles must be disjoint. The protocol fixes reference/query membership and separates enrollment clips from query clips where metadata exists. Novel identities need enough clips for this separation; the API reports infeasibility rather than substituting adjacent frames. Shared source observations/derived crops across training and evaluation are rejected. Identical reference/query pixels are excluded during evaluation.

Reports distinguish known individuals, newly enrolled individuals unseen during training, and explicit unknowns absent from the gallery. They include Rank 1/5, macro identity recall, image-retrieval mAP, correct identification at the rejection threshold, and achieved test FPIR, both per crop and balanced over identity/clip groups. Missing labels are excluded. Tied identity similarities count conservatively against ranking. PyTorch Metric Learning computes image-retrieval mAP.

Calibration uses scikit-learn's complete ROC thresholds over maximum gallery scores from unknown validation queries, choosing the most permissive attainable point that does not exceed the target FPIR. Ties remain together. Reports distinguish requested and attained FPIR and flag limited independent clip coverage; correlated frames are not independent calibration evidence. Bob was intentionally replaced because its NumPy 1.x requirements conflict with this framework's NumPy 2/OpenCV requirements.

`TrainingResult.evaluate()` evaluates the complete fixed ID query cohort; the detection-only `samples` limit does not subsample it. `plots` controls the number of prediction examples (`0` disables the grid). Evaluation writes the shared `ComparisonResult`, frozen cohort, query CSV, metrics and protocol JSON, and figures through the existing report renderer. Training provenance currently checks clip membership and source-reference overlap; it does not certify pixel-level duplicate exclusion against the entire training corpus under renamed/copied files.

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

Every epoch renders a scikit-learn t-SNE projection of a fixed, balanced cohort of up to 512 training/validation crops, using the **current** encoder and deterministic preprocessing. Selection and monitoring preserve training RNGs and model mode. Identity colors and cohort membership remain fixed; refitted 2D coordinates are not a longitudinal distance metric. Set `backend_options={"embedding_samples": 512, "embedding_method": "tsne"}` (`"pca"` is also supported; samples `0` disables it).

W&B receives one image per epoch under `embeddings/tsne`, giving a single media panel with step history/slider. `epoch` is the shared zero-based axis; captions display one-based epoch numbers. Monitor `train/loss`, `val/loss`, `val/monitor_rank1`, `optim/lr`, `time/epoch_seconds`, and `time/monitor_seconds`. The diagnostic Rank-1 uses only the small fixed cohort and is separate from final evaluation. Selection remains validation loss. The cohort manifest, latest coordinates and image enter the checkpoint bundle; local epoch images and W&B history retain earlier views.

Before an epoch finishes, `progress/batch`, `progress/epoch` (zero-based) and `train/batch_loss` report completed batches against `progress/global_step` (optimizer steps). They update after the first batch of each epoch and every `backend_options['trainer']['log_every_n_steps']` batches (default 50; zero retains just the first-batch update). Batch loss is the loss for that batch before gradient-accumulation scaling; `train/loss` remains the epoch aggregate. These metrics help distinguish an inactive loader/process from a long epoch without changing checkpoint selection.

```python
display(id.stats(data))  # samples, labeled/unlabeled, individuals, clips, tracklets
display(id.stats(data, by=("identity_id", "clip_id")))
display(id.stats(data, by="clip_id"))
data.report()  # canonical report API, metadata only

id.visualize_attention(model, queries, gallery=gallery,
                       contrast=True, samples=4, destination="attention.png")
```

The attention visualizer calls [Captum Occlusion](https://captum.ai/api/occlusion.html) directly. It explains maximum cosine similarity to the selected identity; by default, it subtracts the maximum over all competing identities. Each occluded patch uses the preprocessing channel mean. Red supports the target score/margin and blue opposes it; per-image scales and raw attribution arrays are saved alongside the grid. This measures sensitivity to masking, not anatomical ground truth or internal transformer attention. `identity_id=` explains a specified enrolled individual; otherwise it explains the model's own top prediction. The method supports custom encoders without layer-specific hooks.

The [wolf Colab notebook](../notebooks/wolves_identity_colab.ipynb) installs the optional dependencies, extracts Drive NPZs into `/mnt/local-scratch`, reports identity/clip coverage, enrolls all labeled wolves, and trains on CUDA (or local MPS). `DATASET_FIXER_CACHE_DIR` directs the shared cache into scratch. Drive backups retain resumable state and final reports. Install from a published revision containing these changes; staging alone does not update GitHub.
