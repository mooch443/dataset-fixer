"""Enrollment references independent of the encoder's training roster."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import balanced_rows
from .model import input_index, embed, fingerprint


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
        object.__setattr__(self, "references", tuple(dict(r) for r in self.references))

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
        rows = []
        labels = np.asarray(self.labels)
        for start in range(0, len(embeddings), 256):
            similarities = np.asarray(embeddings[start:start+256]) @ self.embeddings.T
            rows.append(np.stack([similarities[:, labels == label].max(axis=1) for label in self.identities], axis=1))
        return np.concatenate(rows) if rows else np.empty((0, len(self.identities)))

    def save(self, path):
        """Save an enrollment gallery without Python pickle objects.

        Parameters:
            path: Output NPZ filename.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(labels=self.labels, references=self.references, model=self.model_fingerprint)
        with path.open("wb") as handle:
            np.savez(handle, embeddings=self.embeddings, metadata=json.dumps(metadata))
        return path

    @classmethod
    def load(cls, path):
        """Restore a gallery.

        Parameters:
            path: Saved gallery NPZ filename.
        """
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            return cls(data["embeddings"], tuple(meta["labels"]), tuple(meta["references"]), meta["model"])


def enroll(model, source, *, samples_per_identity=20, split=None, seed=7, **kwargs):
    if not isinstance(samples_per_identity, int) or samples_per_identity < 1:
        raise ValueError("samples_per_identity must be positive")
    index = input_index(source, split)
    if index.frame.identity_id.isna().any():
        raise ValueError("Every enrollment reference requires an explicit identity_id")
    selected = []
    for label in sorted(index.frame.identity_id.unique()):
        candidates = np.flatnonzero(index.frame.identity_id == label)
        selected.extend(candidates[balanced_rows(index.take(candidates), samples_per_identity, seed)])
    result = embed(model, index.take(selected), **kwargs)
    # Identical references do not count as independent gallery observations.
    unique = {}
    for record in result:
        digest = record.metadata["pixel_sha256"]
        label = record.metadata["reference_identity"]
        if digest in unique and unique[digest].metadata["reference_identity"] != label:
            raise ValueError("Identical reference pixels carry conflicting identity labels")
        unique.setdefault(digest, record)
    records = list(unique.values())
    return Gallery(np.asarray([r.embedding for r in records]), tuple(r.metadata["reference_identity"] for r in records),
                   tuple({key: r.metadata.get(key) for key in ("source_ref", "original_id", "pixel_sha256", "clip_id", "frame_index", "local_id")} for r in records), fingerprint(model))
