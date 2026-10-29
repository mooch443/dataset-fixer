# Synthetic scenes and identity datasets

`dataset_fixer.synthetic` exposes two functions returning the normal `Dataset`
class. Scene generation uses NumPy and OpenCV, camera effects use the existing
Albumentations integration, and visualization/export use the shared framework.
No optional identity-training packages are needed to generate or inspect data.

The examples below can run directly in a local notebook.
Use a local Python kernel with `dataset-fixer` installed; it needs no GPU or
cloud service. Generated datasets are reused through the usual dataset-fixer
cache when settings match.

```python
from dataset_fixer import synthetic

scenes = synthetic.generate(individuals=12, clips=3, frames_per_clip=100)
scenes.visualize(samples=6, columns=3)

data = synthetic.identity(scenes, normalization="posture", bbox_jitter=0.05, workers=4)
data.visualize(samples=12, columns=4)
```

`identity(..., workers=4)` extracts crops from up to four clips concurrently.
The default is `1`. Each clip retains sequential frames and correlated detection
jitter; crop pixels, tracklet IDs, sample ordering, and cache reuse are identical
across worker counts. This uses the same standard thread pool as scene generation
and works directly in macOS notebooks. A single clip uses one worker.

`frames_per_clip` counts scene frames. With 12 individuals and 100 frames there
are at most 1,200 identity crops per clip. Overlaps, truncation, and exits lose
observations; those observations are not replaced. Every overlap ends both
individuals' tracklets, including the individual in front. Their next usable
observation starts a new tracklet. Unaffected individuals continue normally.
Scenes retain their correct visible annotations, including disconnected masks.

## Generation

```python
import albumentations as A

scenes = synthetic.generate(
    individuals=12,
    clips=6,
    frames_per_clip=80,
    difference=0.25,
    variation=0.03,
    size=(960, 640),
    individual_scale=1.0,
    patterns=("stripes", "dots", "stars"),
    neck_focus=0.0,
    texture_noise={"pattern": 0.25, "fur": 0.18},
    backgrounds=("grass", "water"),
    weather={"clouds": 0.25, "fog": 0.12, "speed": 1.0},
    augmentations=[
        A.ChromaticAberration(
            primary_distortion_limit=0.01,
            secondary_distortion_limit=0.02,
            p=0.8,
        ),
        A.OpticalDistortion(distort_limit=0.03, p=0.8),
        A.GaussianBlur(sigma_limit=(0.2, 0.6), p=0.3),
    ],
    seed=42,
    workers=4,
    destination="./synthetic-scenes",
)
```

Omit `destination` to reuse an output in the managed cache. An explicit
destination must be new; existing directories are never overwritten. Outputs
are validated privately before publication.

Set `workers=4` to render up to four clips concurrently using Python's standard
thread pool. The default is `1`; positive values are capped at the clip count.
This works in macOS notebooks without multiprocessing setup. Frames within each
clip remain sequential to preserve motion and augmentation replay. Workers write
frames directly to separate clip directories; image sequences are not buffered
in memory. Pixels, annotations, ordering, and cache identity are independent of
worker count. More workers use more memory and speed depends on available CPU
and disk throughput; a single clip cannot benefit from clip-level concurrency.

Generation shows one `tqdm.auto` progress bar in notebooks and terminals,
counting completed scene frames across all workers. Pass `progress=False` to
hide it. Reusing a cached dataset skips rendering and does not create a bar.
Like `workers`, this display option does not affect output or cache identity.

Each clip uses a shared procedural grass/water environment with smooth temporal
changes and independently initialized trajectories. The same individuals appear
across clips. Consecutive motion, orientation, limb deformation, and crop jitter
are correlated. Tracks can leave and re-enter the camera field.

Cloud shadows and local fog drift smoothly across the entire scene, including
the individuals. `weather=None` selects the mild settings above; `weather={}`
disables them. `clouds` controls maximum shadow strength and `fog` controls
maximum haze opacity, each in `[0, 1]`. Missing strengths in a supplied mapping
are zero, so `weather={"fog": 0.3}` enables fog alone. `speed` is a drift multiplier
in `[0, 10]`, default `1`; `0` freezes the weather pattern. Each clip gets its own
seeded multiscale fields, advected with OpenCV interpolation. They evolve without
per-frame resampling and preserve prefixes when requesting shorter clips.

Weather applies after compositing, followed by the camera effects, before
detection crops. Weather is photometric: masks, landmarks, and geometric
overlap/truncation decisions remain unchanged. It is separate from `augmentations`
so the atmosphere can move while camera parameters stay constant within a clip.

Camera effects apply **after compositing all individuals**, before detection
crops. Their random parameters are sampled once per clip and replayed, with
masks and head/tail landmarks transformed together. `augmentations=None` uses
the mild preset above; `[]` disables it. Lists, Compose, and serialized pipelines
use the existing serializer. Output must remain RGB uint8: do not include
Normalize or ToTensor in scene generation.

Walking uses a support/swing cycle and analytic
[two-bone inverse kinematics](https://docs.unity3d.com/Packages/com.unity.animation.rigging@1.2/manual/constraints/TwoBoneIKConstraint.html).
A supporting foot stays anchored in world coordinates as the body passes over
it. The foot then swings forward to its next placement, bending the knee while
keeping both bone lengths fixed. Short reaches or sharp turns can trigger early
foot lift. Step progress follows distance travelled; body bending, breathing,
and lateral sway follow the walking cycle with small correlated variation.
This is planar kinematic animation; it does not simulate forces or terrain height.

`variation` controls body deformation, default `0.03`; `0` gives a rigid body
with walking legs. Limb roots have fixed DNA-defined material coordinates, and
the roots, markings, eyes, and head/tail landmarks deform together. Per-frame
animation cannot change limb count. NumPy handles the small analytic rig and
OpenCV draws it; no animation engine or extra dependency is needed.

`individual_scale` multiplies body dimensions, limbs, and markings; `2.0` doubles
their size. It is positive and defaults to `1.0`. Frame dimensions, initial
positions, and travel speeds remain unchanged. Larger animals can overlap or
leave the frame more often, reducing the number of usable identity crops.

## Patterns and neck differences

```python
scenes = synthetic.generate(
    individuals=8, clips=2, frames_per_clip=100,
    patterns=("dots", "stars"),
    neck_focus=0.9,
    texture_noise={"pattern": 0.25, "fur": 0.18},
    individual_scale=1.5,
)
```

`patterns` selects allowed motifs: `stripes`, `dots`, and `stars`. The default
allows all three; use `patterns=("stars",)` for stars alone. Numeric DNA weights
blend the motifs continuously, so small gene edits do not abruptly switch a
pattern family. Size, spacing, offsets, strength, and colors also come from DNA.
OpenCV draws the pattern tiles and SciPy interpolates them in the body's material
coordinates, so they bend with the body and stay attached during walking.

`texture_noise` controls two independent genetic layers. `pattern` adds small
irregularities to motif edges/placement and pigment coverage. `fur` adds fine,
directional grain across the coat, including marked regions. Both strengths are
in `[0, 1]`; `None` uses `pattern=0.25, fur=0.18`. Use `{}` to disable both, or
`{"fur": 0.3}` to enable only fur noise. Noise changes color and pattern detail,
without changing silhouettes or labels.

The chromosomes contain separate strengths, frequencies, and continuous field
coefficients for each layer. Nearby chromosomes produce related noise fields;
increasing `difference` increases population texture diversity. `difference=0`
gives the same noise to every individual, rather than removing texture. The
noise is anchored to the body, follows its deformation, and is reused unchanged
across clips, new generation seeds, or renamed identity IDs when DNA is reused.
It is not sampled anew each frame. Fine grain is filtered for pixel size to
reduce shimmer during motion. OpenCV and SciPy handle drawing/interpolation and
Gaussian filtering; no new dependency is required.

`neck_focus` ranges from `0` (ordinary whole-body differences, the default) to
`1` (identity differences confined to a neck band immediately behind the eyes).
At `1`, generated individuals share body shape, limbs, colors, and patterns
outside that band. Intermediate values retain some variation outside the neck
while emphasizing local neck color and motif differences. `difference` still
controls the population spread; `difference=0` gives identical baseline appearances.
The band spans 15–60% of the body half-length toward the head, with soft edges;
the eyes are farther forward at 65%.

Pattern and fur noise also respect `neck_focus`: at `1`, their body fields are
shared across individuals and only their neck fields vary.

`patterns`, `neck_focus`, and `texture_noise` apply when generating a population
from a count. Reused DNA already contains these settings; omit the arguments
when passing a DNA mapping. `individual_scale` can still be set independently.

## Scene tasks

`task="segment"` is the default. `"detect"`, `"pose"`, and `"polo"` are also
supported. Selecting a task changes the annotation representation, not the
rendered scene. All scenes use one class, `individual`; identity is stored
separately. All samples initially belong to `train`.

`task="pose"` exports a bounding box and exactly two keypoints per individual:
`head` and `tail`, joined by the skeleton edge `[1, 2]`. Their coordinates follow
the body deformation and geometric camera augmentations. Visibility is `2`
when visible, `1` when occluded, and `0` outside the frame. The animated leg
joints and the neck band are not additional pose targets. Identity extraction
and posture normalization work with pose scenes just as with segment scenes.
Landmarks dropped by a geometric transform keep their original slots with
visibility `0`; they are never connected to another individual's landmarks.
An observation with an unavailable head or tail does not produce an identity
crop, and its next usable observation starts a new tracklet.

## DNA

`difference` controls population spread in `[0, 1]`; the default is `0.25`.
Zero gives identical baseline appearances. Nearby vectors give similar bodies,
colors, markings, and limbs. Larger spread can change limb presence and body
roundness substantially; it does not enforce minimum pairwise distances.

```python
dna = scenes.manifest["synthetic"]["individuals"]
gene_names = scenes.manifest["synthetic"]["gene_names"]

more = synthetic.generate(
    individuals=dna, clips=2, frames_per_clip=30, seed=123,
)
```

DNA is a mapping from identity strings to ordinary numeric lists. Every gene is
in `[0, 1]`; its name and the DNA version are persisted. Genes cover body length,
width and roundness, body/limb/mark colors, markings, and eight limb slots. Each
slot has presence, attachment, offset, length, angle and width genes. DNA version
2 appends motif and neck appearance genes to the original 62-gene vector;
version 3 appends the separate pattern/fur noise genes. Both older versions
remain accepted unchanged and retain their original, unnoised appearance;
all individuals in a mapping must use the same DNA version. Presence
is fixed for an individual (active above `0.4`); `variation` cannot add/remove
limbs. Explicit DNA is used unchanged, so omit `difference` when supplying it.
The separate `simulation_version` invalidates older cached renders when motion
or rendering changes. DNA schema versions and rendering revisions are separate.

Reusing a seed/settings reproduces output within the same dependency versions.
Fewer frames/clips preserve the corresponding prefix of a larger run. No global
NumPy or Python random state is changed.

## Identity crops and training

```python
data = synthetic.identity(
    scenes,
    normalization="posture",  # Or "none".
    bbox_jitter=0.05,
    context_fraction=0.15,
    orientation_jitter=5.0,    # Degrees, only for posture normalization.
    size=128,
    destination="./synthetic-identities",
)
```

Posture normalization places the head lower-right and the tail upper-left.
It rotates the image and surrounding context together. `none` preserves the
observed orientation. Each detected box edge can move by `bbox_jitter` times
the box's corresponding dimension (allowed range `[0, 0.45]`). The square crop
preserves aspect ratio; context outside the scene uses reflected boundary pixels.

The identity repository contains PNG crops and `samples.jsonl`, with identity,
clip, frame, local individual, tracklet, and original-observation IDs. Synthetic
metadata also records true/detected boxes, visibility and the crop transform.
The crop seed defaults to the scene seed.

Use `augmentations=` at generation time. Arbitrary later image transformations
invalidate the simulation coordinates and are rejected by `identity()`.
Geometry-preserving exports can be used directly.

```python
from dataset_fixer import Dataset, id

loaded = Dataset.open(data.location / "samples.jsonl", format="identity_manifest")
split_data = loaded.split(
    {"train": 0.5, "val": 0.25, "test": 0.25}, group_by="clip_id",
)
closed = id.Protocol(split_data)
# id.train(split_data) uses the existing training API.
```

Use `id.Protocol.open_set` with explicit unknown/novel identity IDs for open-set
experiments. Choose roles with usable observations in their required splits:
short, crowded clips can lose all observations for an individual. Generation
does not silently add clips or replacement samples to satisfy a protocol.

For development with uneven training diversity, limit coverage after splitting:

```python
limited = split_data.limit_coverage({
    "individual-0000": {"clips": 1, "tracklets_per_clip": 1},
    "individual-0001": {"clips": 2},
})
display(id.stats(limited, by=("identity_id", "split")))
# Pass limited directly to id.train(...).
```

Only training is restricted by default; other individuals and held-out splits
retain their observations. These are upper bounds on existing groups, with all
frames of each selected tracklet retained. Selection is deterministic (`seed=42`
by default). Use `split=None` to restrict coverage across the whole dataset,
including on freshly extracted crops. No scene regeneration is needed. The
ordinary Dataset view supports the same exports and training APIs; see
[coverage limits](identification.md#input-and-cliptracklet-policies).

## Exports

Play a generated scene clip directly in a local Jupyter/VS Code notebook:

```python
video_path = scenes.export_video(clip_id="clip-0000", fps=20)
```

This displays an embedded player with playback controls and looping, and returns
the MP4 path. Omit `clip_id` if the dataset has only one clip. The default output
is cached, so rerunning the cell is safe. For a regular file and optional labels:

```python
scenes.export_video(
    clip_id="clip-0000", fps=20,
    destination="./clip-0000.mp4", annotations=True, show=False, overwrite=True,
)
```

Frames are streamed in frame-index order. The MP4 contains every scene frame,
including overlaps that caused missing identity crops. At 20 fps, 100 frames
play for five seconds; lower `fps` to watch slowly. Existing output files are
preserved by default. `overwrite=True` replaces the file only after the new MP4
has encoded and decoded successfully; it also forces regeneration of a cached
video when no destination is supplied. H.264/yuv420p enables macOS/browser playback; an odd width or
height gains a single padding pixel on the right or bottom.

```python
# Portable copies, including lazy file/NPZ crops from ordinary identity datasets.
data.export(
    destination="./portable-identities",
    format="identity_manifest",
    copy_images=True,
)

# Exact foreground union, including disconnected regions and holes.
scenes.export(destination="./scene-masks", format="semantic_masks")

# YOLO polygons may need explicit permission to discard unrepresentable parts.
scenes.export(destination="./scene-yolo", allow_lossy=True)
```

Identity exports default to references (`copy_images=False`). Portable exports
require a new directory and use relative image paths; moving that directory does
not require access to the original scenes or cache. Scene DNA and metadata are
retained through the normal scene exporters and portable identity export.

YOLO cannot represent every occluded silhouette as one polygon. Its existing
loss policy remains explicit. When exact semantic masks matter, export them
directly from `scenes`; `export_formats` with a lossy YOLO output derives its
semantic output from those converted polygons.
