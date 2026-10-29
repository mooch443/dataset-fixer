"""Frozen identity roles, library metrics, and gallery-specific rejection calibration."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .data import SampleIndex, as_dataset, index_of, balanced_rows
from .model import embed, fingerprint


@dataclass(frozen=True)
class Protocol:
    """One protocol for identification; unknown identities enable rejection testing.

    The encoder/loss/gallery machinery is identical for open and closed queries.
    Roles only control which identities may train/enroll and which queries evaluate.

    Parameters:
        data: Identity Dataset containing fixed train/val/test observations.
        unknown_identities: Validation/test identities absent from the gallery.
        novel_identities: Validation/test identities enrolled after encoder training.
        gallery_samples: Maximum references per enrolled identity.
        seed: Deterministic enrollment selection seed.
        gallery_selection: "balanced" (existing random selection) or "medoids".
        gallery_candidates: Maximum candidates per identity for medoid selection.
    """
    data: object
    unknown_identities: dict = field(default_factory=dict)
    novel_identities: dict = field(default_factory=dict)
    gallery_samples: int = 20
    seed: int = 7
    gallery_selection: str = "balanced"
    gallery_candidates: int = 256
    training: object = field(init=False, repr=False)
    references: object = field(init=False, repr=False)
    reference_candidates: object = field(init=False, repr=False)
    queries: dict = field(init=False, repr=False)
    limitations: tuple = field(init=False, default=())

    @classmethod
    def open_set(cls, data, *, unknown_identities, novel_identities=None, gallery_samples=20, seed=7,
                 gallery_selection="balanced", gallery_candidates=256):
        """Construct the same protocol with explicit unknown-identity roles.

        Parameters:
            data: Identity dataset with fixed splits.
            unknown_identities: Disjoint validation/test unknown labels.
            novel_identities: Optional disjoint newly enrolled labels.
            gallery_samples: Maximum references per enrolled individual.
            seed: Enrollment selection seed.
            gallery_selection: "balanced" or "medoids" reference selection.
            gallery_candidates: Maximum medoid candidates per identity.
        """
        return cls(data, unknown_identities, novel_identities or {}, gallery_samples, seed,
                   gallery_selection, gallery_candidates)

    def __post_init__(self):
        from .gallery import _validate_selection
        _validate_selection(self.gallery_selection, self.gallery_samples, self.gallery_candidates)
        index = index_of(self.data)
        frame = index.frame
        unknown = {s: frozenset(map(str, self.unknown_identities.get(s, ()))) for s in ("val", "test")}
        novel = {s: frozenset(map(str, self.novel_identities.get(s, ()))) for s in ("val", "test")}
        if set(self.unknown_identities) - {"val", "test"} or set(self.novel_identities) - {"val", "test"}:
            raise ValueError("Identity roles must use val/test keys")
        roles = [unknown["val"], unknown["test"], novel["val"], novel["test"]]
        for i, values in enumerate(roles):
            if any(values & other for other in roles[i+1:]):
                raise ValueError("Validation/test unknown and novel identity roles must be disjoint")
        reserved = set().union(*roles)
        missing = reserved - set(frame.identity_id.dropna())
        if missing:
            raise ValueError(f"Identity roles refer to missing identities: {sorted(missing)}")
        # Detect shared source observations without decoding every training crop.
        # Different derived crops retain original_id and remain one observation.
        source_keys = pd.util.hash_pandas_object(frame[["path", "array", "row"]].astype(object), index=False)
        originals = frame.original_id.notna()
        source_keys.loc[originals] = pd.util.hash_pandas_object(frame.loc[originals, "original_id"], index=False)
        train_keys = set(source_keys[(frame.split == "train") & ~frame.identity_id.isin(reserved)])
        if source_keys[(frame.split.isin(["val", "test"])) & ~frame.identity_id.isin(reserved)].isin(train_keys).any():
            raise ValueError("Training and evaluation share source observations or derived crops; revise the split")
        supervised = ~frame.identity_id.isin(reserved)
        train_rows = np.flatnonzero(supervised & frame.split.isin(["train", "val"]))
        training_index = index.take(train_rows)
        # Training labels are contiguous, and exclude every reserved identity.
        labels = sorted(training_index.frame.loc[training_index.frame.split == "train", "identity_id"].dropna().unique())
        if len(labels) < 2:
            raise ValueError("Identification training requires at least two training individuals")
        val_labels = set(training_index.frame.loc[training_index.frame.split == "val", "identity_id"].dropna())
        if val_labels - set(labels):
            raise ValueError("Unseen validation identities must be declared novel or unknown")
        training_index.names = dict(enumerate(labels))
        training_index.labels = {v: k for k, v in training_index.names.items()}
        training = as_dataset(training_index, parent=self.data)
        training._metadata.names = training_index.names
        selected, eligible, excluded_clips, limitations = [], [], {}, []
        enrolled = set(labels) | novel["val"] | novel["test"]
        for label in sorted(enrolled):
            rows = np.flatnonzero(frame.identity_id == label)
            per_identity = frame.iloc[rows]
            if label in labels:
                candidates = rows[(per_identity.split == "train").to_numpy()]
                query_clips = set(per_identity.loc[per_identity.split.isin(["val", "test"]), "clip_id"].dropna())
                if query_clips:
                    candidates = candidates[~frame.iloc[candidates].clip_id.isin(query_clips).to_numpy()]
            else:
                role_split = "val" if label in novel["val"] else "test"
                candidates = rows[(per_identity.split == "train").to_numpy()]
                queries = per_identity[per_identity.split == role_split]
                query_clips = set(queries.clip_id.dropna())
                candidates = candidates[~frame.iloc[candidates].clip_id.isin(query_clips).to_numpy()]
                if not len(candidates):
                    clips = sorted(queries.clip_id.dropna().unique())
                    if len(clips) < 2:
                        raise ValueError(f"Novel identity {label} needs separate enrollment and query clips")
                    gallery_clip = clips[0]
                    candidates = rows[((per_identity.split == role_split) & (per_identity.clip_id == gallery_clip)).to_numpy()]
                    excluded_clips[label] = gallery_clip
            if not len(candidates):
                raise ValueError(f"Identity {label} has no enrollment clip separate from queries; revise the split")
            if frame.iloc[candidates].clip_id.isna().any():
                limitations.append(f"Identity {label}: clip coverage unavailable; only observation/pixel disjointness can be checked")
            selected.extend(candidates[balanced_rows(index.take(candidates), self.gallery_samples, self.seed)])
            if self.gallery_selection == "medoids":
                eligible.extend(candidates)
        # The default keeps its historical fixed query cohort. Model-dependent
        # selection reserves the entire eligible pool before any model is used.
        reserved_rows = selected if self.gallery_selection == "balanced" else eligible
        duplicate = np.zeros(len(frame), dtype=bool)
        duplicate[reserved_rows] = True
        original_ids = set(frame.iloc[reserved_rows].original_id.dropna())
        if original_ids:
            duplicate |= frame.original_id.isin(original_ids).to_numpy()
        fields = ["clip_id", "local_id", "frame_index"]
        complete = frame[fields].notna().all(axis=1)
        references = frame.iloc[reserved_rows].dropna(subset=fields)
        if len(references):
            keys = pd.MultiIndex.from_frame(frame.loc[complete, fields])
            duplicate[np.flatnonzero(complete)] |= keys.isin(pd.MultiIndex.from_frame(references[fields]))
        query_sets = {}
        for split in ("val", "test"):
            allowed = set(labels) | set(novel[split]) | set(unknown[split])
            candidates = np.flatnonzero((frame.split == split) & frame.identity_id.isin(allowed))
            mask = ~duplicate[candidates]
            for label, clip in excluded_clips.items():
                mask &= ~((frame.iloc[candidates].identity_id == label) & (frame.iloc[candidates].clip_id == clip)).to_numpy()
            candidates = candidates[mask]
            query_sets[split] = as_dataset(index.take(candidates), parent=self.data)
            for label in unknown[split] | novel[split]:
                if label not in set(query_sets[split]._samples.frame.identity_id):
                    raise ValueError(f"No {split} query observations for reserved identity {label}")
        object.__setattr__(self, "unknown_identities", unknown)
        object.__setattr__(self, "novel_identities", novel)
        object.__setattr__(self, "training", training)
        references = as_dataset(index.take(selected), parent=self.data)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "reference_candidates", as_dataset(index.take(eligible), parent=self.data)
                           if self.gallery_selection == "medoids" else references)
        object.__setattr__(self, "queries", query_sets)
        object.__setattr__(self, "limitations", tuple(limitations))



@dataclass(frozen=True)
class RejectionCalibration:
    """Validation-selected operating point bound to an encoder and gallery.

    Parameters:
        threshold: Minimum accepted maximum identity similarity.
        target_fpir: Requested maximum false-positive identification rate.
        achieved_fpir: Attained validation false-positive identification rate.
        model_fingerprint: Inference tensor/preprocessing identity.
        gallery_fingerprint: Enrollment contents used during calibration.
        validation_fingerprint: Validation cohort metadata fingerprint.
        unknown_queries: Number of labeled unknown validation queries.
        independent_groups: Number of validation identity/clip groups.
        limitations: Coverage, discreteness and independence qualifications.
    """
    threshold: float
    target_fpir: float
    achieved_fpir: float
    model_fingerprint: str
    gallery_fingerprint: str
    validation_fingerprint: str
    unknown_queries: int
    independent_groups: int
    limitations: tuple[str, ...] = ()

    def validate(self, model, gallery):
        """Reject incompatible inference sources.

        Parameters:
            model: Loaded identity encoder.
            gallery: Enrollment gallery to query.
        """
        gallery.validate(model)
        if self.model_fingerprint != fingerprint(model) or self.gallery_fingerprint != gallery.fingerprint:
            raise ValueError("Model or gallery changed; recalibrate rejection on validation queries")

    def save(self, path):
        """Save calibration metadata.

        Parameters:
            path: Destination JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path):
        """Restore calibration metadata.

        Parameters:
            path: Saved calibration JSON file.
        """
        return cls(**json.loads(Path(path).read_text()))


def threshold_at_fpir(scores, target):
    """Select the most permissive attainable library ROC point not exceeding FPIR.

    A sentinel positive lets sklearn enumerate every negative-score operating
    point. It does not enter the unknown-query denominator. Ties stay together.
    """
    from sklearn.metrics import roc_curve
    scores = np.asarray(scores, dtype=np.float64)
    if not 0 <= target <= 1 or scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration needs finite unknown scores and target_fpir in [0, 1]")
    sentinel = np.nextafter(scores.max(), np.inf)
    fpr, _, thresholds = roc_curve(np.r_[np.zeros(len(scores)), 1], np.r_[scores, sentinel], drop_intermediate=False)
    selected = np.flatnonzero((fpr <= target) & np.isfinite(thresholds))[-1]
    return float(thresholds[selected]), float(fpr[selected])


def _query_embeddings(model, data, gallery, **kwargs):
    prediction = embed(model, data, **kwargs)
    original = {r.get("original_id") for r in gallery.references if r.get("original_id")}
    hashes = {r.get("pixel_sha256") for r in gallery.references}
    records = tuple(r for r in prediction if r.metadata.get("original_id") not in original
                    and r.metadata["pixel_sha256"] not in hashes)
    if not records:
        raise ValueError("No independent queries remain after excluding enrollment observations and identical pixels")
    vectors = np.asarray([r.embedding for r in records])
    return records, vectors, len(prediction)-len(records)


def calibrate(model, data, *, gallery, unknown_identities, target_fpir=0.01):
    gallery.validate(model)
    unknown_ids = set(map(str, unknown_identities))
    if not unknown_ids or unknown_ids & set(gallery.identities):
        raise ValueError("Explicit validation unknown identities must be absent from the gallery")
    index = index_of(data)
    unknown = index.where(identity_id=unknown_ids)
    records, vectors, duplicates = _query_embeddings(model, unknown, gallery)
    scores = gallery.scores(vectors).max(axis=1)
    threshold, achieved = threshold_at_fpir(scores, target_fpir)
    groups = {(r.metadata["reference_identity"], r.metadata.get("clip_id")) for r in records}
    limitations = []
    if len(groups) < (1 / target_fpir if target_fpir > 0 else 100):
        limitations.append(f"Only {len(groups)} independent identity/clip groups; correlated crops do not establish a precise low-FPIR operating point")
    if not np.isclose(achieved, target_fpir):
        limitations.append(f"Discrete/tied validation scores achieve FPIR={achieved:g}, below requested {target_fpir:g}")
    if duplicates:
        limitations.append(f"Excluded {duplicates} duplicate enrollment observations")
    return RejectionCalibration(threshold, target_fpir, achieved, fingerprint(model), gallery.fingerprint,
                                unknown.fingerprint, len(records), len(groups), tuple(limitations))


def evaluate(model, data, *, gallery, rejection=None, novel_identities=(), unknown_identities=(),
             destination=None, split="test", limitations=(), plots=0, progress=True, _on_progress=None):
    import torch
    from ..comparison.types import ComparisonResult
    from ..sources import cache_root
    from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
    from pytorch_metric_learning.utils.inference import CustomKNN
    from pytorch_metric_learning.distances import CosineSimilarity
    from ._progress import batches
    gallery.validate(model)
    if rejection is not None:
        rejection.validate(model, gallery)
    records, vectors, duplicates = _query_embeddings(model, data, gallery, progress=progress, _on_progress=_on_progress)
    identities = np.asarray(gallery.identities)
    known = set(identities)
    explicit_unknown = set(map(str, unknown_identities))
    novel = set(map(str, novel_identities))
    label_codes = {v: i for i, v in enumerate(gallery.identities)}
    references = torch.tensor(gallery.embeddings.copy())
    # PML returns per-class means. Tag each query with its batch-local row ID
    # so each result is one query's AP, while relevance still compares only
    # identity (the second column). This uses PML's documented 2D-label API.
    reference_labels = np.column_stack((np.full(len(gallery.labels), -1),
                                       [label_codes[v] for v in gallery.labels]))
    calculator = AccuracyCalculator(include=("mean_average_precision",), k=len(gallery.labels),
        return_per_class=True, label_comparison_fn=lambda a, b: a[..., 1] == b[..., 1],
        device=torch.device("cpu"), knn_func=CustomKNN(CosineSimilarity(), batch_size=128))
    rows = []
    for start, end in batches(len(records), 128, stage="metrics", description="ID retrieval metrics",
                              progress=progress, on_progress=_on_progress):
        labels = [r.metadata.get("reference_identity") for r in records[start:end]]
        invalid = set(labels) - known - explicit_unknown - {None}
        if invalid:
            raise ValueError(f"Query identities {sorted(invalid)} are absent from the gallery but not explicitly declared unknown")
        enrolled_rows = np.array([i for i, label in enumerate(labels) if label in known], dtype=int)
        average_precision = dict()
        if len(enrolled_rows):
            query_labels = np.column_stack((enrolled_rows, [label_codes[labels[i]] for i in enrolled_rows]))
            values = calculator.get_accuracy(vectors[start:end][enrolled_rows], query_labels, references,
                reference_labels, ref_includes_query=False)["mean_average_precision"]
            average_precision = dict(zip(enrolled_rows, values))
        scores = gallery.scores(vectors[start:end])
        for i, (record, label, score) in enumerate(zip(records[start:end], labels, scores)):
            if label is None:
                continue  # Unlabeled never means unknown.
            enrolled = label in known
            target = score[label_codes[label]] if enrolled else None
            # Pessimistic tie handling: indistinguishable impostors count ahead.
            rank = 1 + int(np.sum(np.delete(score, label_codes[label]) >= target)) if enrolled else None
            accepted = rejection is None or float(score.max()) >= rejection.threshold
            rows.append(dict(identity_id=label, clip_id=record.metadata.get("clip_id"),
                role="novel" if label in novel else "known" if enrolled else "unknown",
                rank1=float(rank == 1) if enrolled else None, rank5=float(rank <= 5) if enrolled else None,
                average_precision=average_precision.get(i), correct_identification=float(rank == 1 and accepted) if enrolled else None,
                false_positive=float(accepted) if not enrolled and rejection is not None else None,
                similarity=float(score.max()), accepted=bool(accepted)))
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("Identity evaluation requires labeled independent queries; unlabeled images are not unknown identities")
    summaries = []
    metric_columns = ["rank1", "rank5", "average_precision", "correct_identification", "false_positive"]
    for role, group in table.groupby("role", sort=True):
        for balance in ("crop", "identity_clip"):
            metric = group[metric_columns] if balance == "crop" else group.groupby(["identity_id", "clip_id"], dropna=False)[metric_columns].mean().groupby(level="identity_id").mean()
            summary = {key: float(value) if pd.notna(value) else None for key, value in metric.mean().items()}
            summary.update(model=model.name, role=role, weighting=balance, queries=len(group),
                macro_identity_recall=float(metric.rank1.mean() if balance == "identity_clip" else group.groupby("identity_id").rank1.mean().mean()) if role != "unknown" else None)
            summaries.append(summary)
    location = Path(destination) if destination else cache_root() / "identity-evaluation" / hashlib.sha256((fingerprint(model)+index_of(data).fingerprint+gallery.fingerprint).encode()).hexdigest()[:16]
    if progress:
        print(f"Saving ID {split} evaluation report: {location}", flush=True)
    if _on_progress is not None:
        _on_progress("report", 0, 1)
    location.mkdir(parents=True, exist_ok=True)
    table.to_csv(location / "queries.csv", index=False)
    ranking = pd.DataFrame(summaries)
    ranking.to_json(location / "metrics.json", orient="records", indent=2)
    from ..comparison.cohort import freeze_cohort, write_cohort
    cohort = freeze_cohort(data, split, progress=progress)
    write_cohort(cohort, location / "cohort.jsonl")
    from .model import runtime
    training_clips = set(runtime(model).identity.get("training_clips", ()))
    shared_clips = training_clips & set(table.clip_id.dropna())
    notes = [*limitations, *(rejection.limitations if rejection else ()),
             "Similarity is cosine similarity, not a probability. Ties count against identification rank."]
    if explicit_unknown and rejection is None:
        notes.append("Unknown FPIR is unavailable without validation-calibrated rejection")
    if shared_clips:
        notes.append(f"Training and evaluation share clips/backgrounds: {sorted(shared_clips)}")
    notes.append("Training provenance records clip membership; pixel-level training-duplicate exclusion is not certified")
    if duplicates:
        notes.append(f"Excluded {duplicates} enrollment/duplicate queries")
    if table.clip_id.isna().any():
        notes.append("Some clip metadata is missing; clip independence cannot be verified")
    settings = dict(split=split, gallery=gallery.fingerprint, rejection=asdict(rejection) if rejection else None,
                    metrics_library="pytorch-metric-learning; sklearn ROC", excluded_duplicates=duplicates)
    (location / "protocol.json").write_text(json.dumps(settings, indent=2))
    _metric_chart(location, ranking, settings)
    if plots:
        from .model import predict
        from .data import _file_record
        from ..visualization import select_visualization_records, VisualizationOptions
        # Select only observations that actually entered the labeled report.
        chosen = select_visualization_records([record for record in records
            if record.metadata.get("reference_identity") is not None], VisualizationOptions(samples=plots, seed=7))
        selected = SampleIndex([_file_record(dict(zip(("path", "array", "row", "crop", "trim_black_borders"),
            json.loads(record.metadata["source_ref"])+[False]), identity_id=record.metadata["reference_identity"],
            clip_id=record.metadata.get("clip_id")), Path.cwd()) for record in chosen])
        prediction = predict(model, selected, gallery=gallery, rejection=rejection,
                             mode="open" if rejection is not None else "closed", progress=progress)
        prediction.visualize(samples=plots, columns=min(3, plots),
                             destination=location / "predictions.png", show=False)
    if progress:
        print(f"ID {split} evaluation complete: {len(table):,} labeled queries", flush=True)
    if _on_progress is not None:
        _on_progress("report", 1, 1)
    return ComparisonResult(location, ranking, cohort.fingerprint, True, bool(shared_clips), False, True,
                            {}, "identity", settings, tuple(notes))


def visualize_identity(source, *, options, split=None):
    """Prepare identity captions/pixels for the existing generic grid renderer."""
    from ..visualization import VisualizationItem, VisualizationPanel, visualize_records, finish_visualization
    from ..model import PredictionResult
    from .data import ImageRef
    if isinstance(source, PredictionResult):
        from ..comparison.plot_labels import model_identity_card
        heading = model_identity_card(source.model_metadata or {"name": source.model_name}, width=max(144, round(options.panel_size*96)))
        records = source.records
        def prepare(record):
            ref = ImageRef(*json.loads(record.metadata["source_ref"]))
            pixels = ref.read()
            label = f"{record.identity_id or 'unknown'} · cosine {record.similarity:.3f}" if record.similarity is not None else "Embedding"
            truth = record.metadata.get("reference_identity") or "unlabeled"
            caption = f"Reference: {truth} · {record.metadata.get('clip_id') or 'no clip'}"
            return VisualizationItem(record.image_path, caption,
                (VisualizationPanel(source.model_name, pixels, footer=label, heading=heading),), np.ones(pixels.shape[:2], bool))
    else:
        index = index_of(source)
        if split is not None:
            index = index.where(split=split)
        records = balanced_rows(index, options.samples or len(index), options.seed)
        def prepare(row):
            sample = index[row]
            pixels = sample.image_ref.read()
            return VisualizationItem(sample.image_path, sample.identity_id or "unlabeled",
                (VisualizationPanel(sample.clip_id or "no clip", pixels, footer=sample.image_path.name),), np.ones(pixels.shape[:2], bool))
    chart = visualize_records(records, options=options, prepare=prepare, title="Individual identification")
    finish_visualization(chart, options)
    return chart


def _metric_chart(location, ranking, settings):
    import altair as alt
    from ..tabular import chart_data
    from ..comparison.reporting import _save_figure
    columns = ["rank1", "rank5", "average_precision", "correct_identification", "false_positive"]
    values = ranking.melt(id_vars=["model", "role", "weighting"], value_vars=columns,
                          var_name="metric", value_name="value").dropna(subset=["value"]).reset_index(drop=True)
    if values.empty:
        return
    values["metric"] = values.metric.map({"rank1": "Rank 1", "rank5": "Rank 5", "average_precision": "Image retrieval mAP",
        "correct_identification": "Correct identification", "false_positive": "Unknown FPIR"})
    values["weighting"] = values.weighting.map({"crop": "Crop", "identity_clip": "Identity / clip"})
    base = alt.Chart(chart_data(values)).encode(
        x=alt.X("metric:N", title=None, scale=alt.Scale(type="band"), axis=alt.Axis(labelAngle=-25)),
        y=alt.Y("value:Q", title="Rate", scale=alt.Scale(domain=[0, 1.05]), axis=alt.Axis(values=[0,.25,.5,.75,1], format=".2f")),
        xOffset=alt.XOffset("weighting:N", scale=alt.Scale(type="band")), color=alt.Color("weighting:N", title="Weighting"),
        tooltip=["role:N", "metric:N", "weighting:N", alt.Tooltip("value:Q", format=".3f")],
    )
    bars = base.mark_bar() + base.mark_text(dy=-6).encode(color=alt.value("black"), text=alt.Text("value:Q", format=".2f"))
    chart = alt.hconcat(*(bars.transform_filter(alt.datum.role == role).properties(
        width=350, height=220, title=role.title()) for role in sorted(values.role.unique())), spacing=25)
    _save_figure(location, "identity_metrics", chart, values, settings)

def augmentation_preview(dataset, augmentations, config, split, options, *, weights=None):
    from dataclasses import replace
    from ..training.config import TrainingConfig
    from ..training.preview import _preview_seed
    from .data import index_of, balanced_rows
    from .model import augmentation_pipeline, preprocess, seed_augmentations
    from ..static_rendering import letterbox_image, format_label
    from ..visualization import VisualizationItem, VisualizationPanel, visualize_records, finish_visualization
    index = index_of(dataset)
    if split is not None:
        index = index.where(split=split)
    index = index.take(np.flatnonzero(index.frame.identity_id.notna()))
    chosen = balanced_rows(index, options.samples or len(index), options.seed)
    config = config or TrainingConfig()
    prep = config.backend_options.get("preprocessing")
    if weights is not None:
        from ..model_sources import resolve_model_source
        from .model import load_checkpoint
        prep = load_checkpoint(resolve_model_source(weights, progress=False).path)["identity"]["preprocessing"]
        if config.resolution is not None and config.resolution != prep["resolution"]:
            raise ValueError("Preview resolution conflicts with checkpoint preprocessing")
    if prep is None:
        from torchvision.models import ResNet18_Weights
        native = ResNet18_Weights.DEFAULT.transforms()
        prep = dict(resolution=native.crop_size[0], mean=native.mean, std=native.std, interpolation=native.interpolation.value)
    prep = {**prep, "resolution": config.resolution or prep["resolution"]}
    pipeline, _ = augmentation_pipeline(augmentations, seed=options.seed)
    pipeline.save_applied_params = True

    def prepare(position):
        sample = index[position]
        original = sample.image_ref.read()
        seed_augmentations(pipeline, options.seed+position)
        applied = pipeline(image=original.copy())
        transformed = applied["image"]
        tensor = preprocess(transformed, prep)
        # Undo normalization only for display; show exactly the training crop/resize.
        pixels = tensor.numpy().transpose(1, 2, 0) * np.asarray(prep["std"]) + np.asarray(prep["mean"])
        augmented = np.clip(pixels*255, 0, 255).astype(np.uint8)
        label = f"{sample.identity_id or 'unlabeled'} · {sample.clip_id or 'no clip'}"
        source = sample.image_path.name + (f" · row {sample.image_ref.row}" if sample.image_ref.row is not None else "")
        names = {"RandomCropFromBorders": "crop", "Affine": "affine", "HueSaturationValue": "HSV",
                 "GaussianBlur": "blur", "Erasing": "erase", "ToGray": "gray"}
        effects = []
        for name, params in applied["applied_transforms"]:
            if name == "RandomBrightnessContrast":
                effects.extend((f"bright{'+' if params['beta'] >= 0 else '−'}",
                                f"contrast{'+' if params['alpha'] >= 1 else '−'}"))
            else:
                effects.append(names.get(name, name))
        width = max(18, max(144, round(options.panel_size * 96)) // 7)
        caption = "\n".join(format_label(", ".join(effects) or "No random transform",
                                         mode="wrap", maximum=width, wrap_width=width))
        return VisualizationItem(sample.image_path, label,
            (VisualizationPanel("Original", np.asarray(letterbox_image(original, width=augmented.shape[1], height=augmented.shape[0])), footer=source),
             VisualizationPanel("Augmented", augmented, footer=caption)),
            np.ones(augmented.shape[:2], dtype=bool))

    with _preview_seed(options.seed):
        chart = visualize_records(chosen, options=replace(options, samples=len(chosen)), prepare=prepare, title="Training augmentations")
    return finish_visualization(chart, options)
