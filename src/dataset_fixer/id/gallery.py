"""Enrollment references independent of the encoder's training roster."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import balanced_rows, SampleIndex, ImageRef, as_dataset, index_of, open_dataset
from .model import input_index, embed, fingerprint
from ..utils import qualified_tracklet_id


def _reference_hashes(gallery):
    """Pixel identities for leave-one-observation-out inspection of a gallery."""
    hashes = []
    index = None
    for row, reference in enumerate(gallery.references):
        digest = reference.get("pixel_sha256")
        if digest is None:
            if index is None:
                index = gallery._index()
            pixels = index[row].image_ref.read()
            digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        hashes.append(digest)
    return hashes


def _reference_mask(gallery, hashes, source_ref, pixel_sha256):
    return np.array([digest != pixel_sha256 and ref["source_ref"] != source_ref
                     for digest, ref in zip(hashes, gallery.references)])


@dataclass(frozen=True)
class Gallery:
    """Reference embeddings independent of the encoder's training roster.

    Parameters:
        embeddings: Finite N×D reference features, normalized on construction.
        labels: Individual label for each reference.
        references: Source, clip and observation metadata for each reference.
        model_fingerprint: Encoder/projection/preprocessing identity.
    """
    embeddings: np.ndarray
    labels: tuple[str, ...]
    references: tuple[dict, ...]
    model_fingerprint: str
    metadata: dict = field(default_factory=dict)
    _reference_dataset: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        vectors = np.array(self.embeddings, dtype=np.float32, copy=True)
        if vectors.ndim != 2 or not len(vectors) or len(vectors) != len(self.labels) or len(vectors) != len(self.references):
            raise ValueError("Gallery needs one embedding, identity and reference per observation")
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) < 1e-8):
            raise ValueError("Gallery embeddings must be finite and nonzero")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.allclose(norms, 1, atol=1e-6):
            vectors /= norms
        vectors.flags.writeable = False
        object.__setattr__(self, "embeddings", vectors)
        object.__setattr__(self, "labels", tuple(str(label) for label in self.labels))
        object.__setattr__(self, "references", tuple(
            {**r, "tracklet_id": qualified_tracklet_id(r.get("clip_id"), r["tracklet_id"])}
            if "tracklet_id" in r else dict(r) for r in self.references))

    @property
    def identities(self):
        return tuple(sorted(set(self.labels)))

    @property
    def fingerprint(self):
        digest = hashlib.sha256(self.embeddings.tobytes())
        digest.update(json.dumps([self.model_fingerprint, self.labels, self.references], sort_keys=True).encode())
        return digest.hexdigest()

    def validate(self, model):
        """Check inference compatibility.

        Parameters:
            model: Loaded identity model to compare with this gallery.
        """
        if self.model_fingerprint != fingerprint(model):
            raise ValueError("Gallery belongs to a different model; re-enroll references")

    def scores(self, embeddings):
        """Return maximum cosine similarity for each enrolled identity.

        Parameters:
            embeddings: Normalized query features in N×D form.
        """
        return identity_scores(np.asarray(embeddings), self.embeddings, self.labels)

    @property
    def summary(self):
        """Local enrollment counts per identity; no additional W&B metrics."""
        import pandas as pd
        return pd.DataFrame(self.metadata.get("identities", []))

    def visualize(self, **kwargs):
        """Show the selected source images using Dataset.visualize options."""
        return as_dataset(self._index(), location=Path.cwd()).visualize(**kwargs)

    def _index(self):
        index = (index_of(open_dataset(self._reference_dataset, format="identity_manifest"))
                 if self._reference_dataset is not None else
                 input_index([ImageRef(*json.loads(r["source_ref"])) for r in self.references]))
        if len(index) != len(self.labels):
            raise ValueError("Gallery reference images do not match its embeddings")
        frame = index.frame.assign(identity_id=list(self.labels))
        for key in ("clip_id", "tracklet_id", "original_id", "frame_index", "local_id"):
            frame[key] = [r.get(key) for r in self.references]
        return SampleIndex(frame)

    def visualize_attention(self, model, *, samples_per_identity=20, **kwargs):
        """Mean own-identity evidence, or its difference from other identity means.

        Uses at most samples_per_identity clip-balanced references per identity.
        This only limits maps being averaged; it does not change the gallery,
        reference selection for prediction, or the references used for scoring.
        Each reference is scored without itself or identical gallery pixels.
        contrast=False shows each identity's mean own-similarity attribution.
        contrast=True (default) subtracts the equally weighted mean of the other
        identities' own-similarity attribution maps, computed on their images.
        These are separate calls/plots. Query id.visualize_attention instead
        contrasts scores on the same input image.
        Each reference card pairs the unoverlaid representative with its overlay;
        columns counts identity pairs per row (default 2).
        Other options match id.visualize_attention (except identity_id); samples
        limits identities here, defaulting to all enrolled identities. Orientation
        and crop alignment must be comparable for spatial averaging to be useful.
        """
        from .inspection import _visualize_reference_attention
        return _visualize_reference_attention(model, self, samples_per_identity=samples_per_identity, **kwargs)

    def save(self, path, *, copy_images=False):
        """Save an enrollment gallery without Python pickle objects.

        Parameters:
            path: Output NPZ filename.
            copy_images: Materialize the selected crops in a new sibling
                ``<stem>.references`` directory. Move it together with the NPZ.
                Original provenance and gallery/calibration fingerprints stay intact.
        """
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(labels=self.labels, references=self.references, model=self.model_fingerprint,
                        enrollment=self.metadata)
        if copy_images:
            from .data import export_manifest
            destination = path.with_name(f"{path.stem}.references")
            export_manifest(as_dataset(self._index(), location=path), destination, copy_images=True)
            metadata["reference_dataset"] = f"{destination.name}/samples.jsonl"
        elif self._reference_dataset is not None:
            metadata["reference_dataset"] = str(self._reference_dataset)
        with path.open("wb") as handle:
            np.savez(handle, embeddings=self.embeddings, metadata=json.dumps(metadata))
        return path

    @classmethod
    def load(cls, path):
        """Restore a gallery.

        Parameters:
            path: Saved gallery NPZ filename.
        """
        path = Path(path).expanduser().resolve()
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            dataset = (path.parent / meta["reference_dataset"]).resolve() if meta.get("reference_dataset") else None
            return cls(data["embeddings"], tuple(meta["labels"]), tuple(meta["references"]), meta["model"],
                       meta.get("enrollment", {}), dataset)


def identity_scores(embeddings, references, labels):
    """Canonical maximum-cosine identity scores, including differentiable tensors."""
    labels = np.asarray(labels)
    identities = sorted(set(labels))
    numpy = isinstance(embeddings, np.ndarray)
    if not numpy:
        import torch
    masks = [labels == label for label in identities]
    if not numpy:
        masks = [torch.as_tensor(mask, device=embeddings.device) for mask in masks]
    rows = []
    for start in range(0, len(embeddings), 256):
        similarities = embeddings[start:start+256] @ references.T
        values = [similarities[:, mask].max(axis=1) if numpy else similarities[:, mask].max(dim=1).values
                  for mask in masks]
        rows.append(np.stack(values, axis=1) if numpy else torch.stack(values, dim=1))
    if rows:
        return np.concatenate(rows) if numpy else torch.cat(rows)
    return np.empty((0, len(identities))) if numpy else embeddings.new_empty((0, len(identities)))


def _validate_selection(selection, count, candidates):
    if selection not in {"balanced", "medoids"}:
        raise ValueError("gallery selection must be 'balanced' or 'medoids'")
    for name, value in (("samples_per_identity", count), ("candidates_per_identity", candidates)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if selection == "medoids" and candidates < count:
        raise ValueError("candidates_per_identity must be at least samples_per_identity")


def _candidate_rows(index, count, seed):
    selected = []
    for label in sorted(index.frame.identity_id.unique()):
        rows = np.flatnonzero(index.frame.identity_id == label)
        selected.extend(rows[balanced_rows(index.take(rows), count, seed)])
    return selected


def _medoid_rows(index, embeddings, count, seed):
    """Keep hierarchical clip/tracklet quotas; choose representatives within them."""
    chosen = _candidate_rows(index, count, seed)
    fields = [key for key in ("identity_id", "clip_id", "tracklet_id") if index.frame[key].notna().all()]
    groups = index.frame.groupby(fields, observed=True, sort=True).ngroup().to_numpy()
    selected = []
    for group, quota in zip(*np.unique(groups[chosen], return_counts=True)):
        rows = np.flatnonzero(groups == group)
        if len(rows) > quota:
            rows = rows[_embedding_medoids(embeddings[rows], int(quota), seed)]
        selected.extend(rows)
    return selected


def _embedding_medoids(embeddings, count, seed):
    from kmedoids import fasterpam
    from sklearn.metrics.pairwise import cosine_distances
    distances = cosine_distances(embeddings).astype(np.float32)
    return fasterpam(distances, count, random_state=seed, n_cpu=1).medoids


def enroll(model, source, *, samples_per_identity=None, selection=None, candidates_per_identity=None,
           split=None, seed=None, **kwargs):
    from .evaluation import Protocol
    protocol = source if isinstance(source, Protocol) else None
    if protocol is not None and split is not None:
        raise ValueError("A Protocol already defines enrollment roles; omit split")
    selection = selection if selection is not None else (protocol.gallery_selection if protocol else "balanced")
    count = samples_per_identity if samples_per_identity is not None else (protocol.gallery_samples if protocol else 20)
    candidates = candidates_per_identity if candidates_per_identity is not None else (protocol.gallery_candidates if protocol else 256)
    # Balanced Protocol enrollment historically reselected its frozen references
    # with seed 7, independently of the seed that created those references.
    seed = seed if seed is not None else (protocol.seed if protocol and selection == "medoids" else 7)
    _validate_selection(selection, count, candidates)
    if protocol is not None:
        if selection == "medoids" and protocol.gallery_selection != "medoids":
            raise ValueError("Set gallery_selection='medoids' on the Protocol to reserve candidate observations")
        source = protocol.reference_candidates if selection == "medoids" else protocol.references
    index = input_index(source, split)
    if not len(index) or index.frame.identity_id.isna().any():
        raise ValueError("Every enrollment reference requires an explicit identity_id")
    selected = _candidate_rows(index, count if selection == "balanced" else candidates, seed)
    result = embed(model, index.take(selected), **kwargs)
    # Identical references do not count as independent gallery observations.
    unique = {}
    unique_rows = []
    for row, record in zip(selected, result):
        digest = record.metadata["pixel_sha256"]
        label = record.metadata["reference_identity"]
        if digest in unique and unique[digest].metadata["reference_identity"] != label:
            raise ValueError("Identical reference pixels carry conflicting identity labels")
        if digest not in unique:
            unique[digest] = record
            unique_rows.append(row)
    records = list(unique.values())
    candidate_refs = [r.metadata["source_ref"] for r in result]
    candidate_index = index.take(unique_rows)
    if selection == "medoids":
        keep = _medoid_rows(candidate_index, np.asarray([r.embedding for r in records]), count, seed)
        records = [records[i] for i in keep]
    summaries = []
    for label in sorted(index.frame.identity_id.unique()):
        refs = [r.metadata for r in records if r.metadata["reference_identity"] == label]
        summaries.append(dict(identity_id=label, eligible=int((index.frame.identity_id == label).sum()),
            candidates=sum(r.metadata["reference_identity"] == label for r in result),
            unique_candidates=int((candidate_index.frame.identity_id == label).sum()), references=len(refs),
            clips=len({r["clip_id"] for r in refs if r.get("clip_id") is not None}),
            tracklets=len({r["tracklet_id"] for r in refs if r.get("tracklet_id") is not None}),
            shortfall=max(0, count-len(refs))))
    metadata = dict(selection=selection, seed=seed, samples_per_identity=count,
                    candidates_per_identity=candidates if selection == "medoids" else count,
                    candidates=candidate_refs, identities=summaries)
    return Gallery(np.asarray([r.embedding for r in records]), tuple(r.metadata["reference_identity"] for r in records),
                   tuple({key: r.metadata.get(key) for key in ("source_ref", "original_id", "pixel_sha256", "clip_id", "frame_index", "local_id", "tracklet_id")} for r in records),
                   fingerprint(model), metadata)
