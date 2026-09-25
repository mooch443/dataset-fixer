"""Artifact cache migration must preserve the SDK's real integrity checks."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from dataset_fixer.model_sources import _download_wandb, resolve_model_source


@pytest.fixture(params=["checkpoint_best.pth", "weights/checkpoint_best.pth"])
def cached_artifact(tmp_path, monkeypatch, request):
    import wandb
    from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
    from wandb.sdk.lib.hashutil import md5_file_b64

    source = tmp_path / "source.pth"
    torch.save({"model_name": "RFDETRKeypointPreview", "model": {"weight": torch.ones(1)},
                "model_config": {"use_grouppose_keypoints": True, "resolution": 1296}}, source)
    artifact = SimpleNamespace(
        name="model-old-run:v51", qualified_name="team/project/model-old-run:v51",
        digest="immutable-digest", metadata={}, manifest=SimpleNamespace(entries={}),
        is_draft=lambda: False,
    )
    identity = {"name": artifact.qualified_name, "digest": artifact.digest}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    root = tmp_path / "package-cache" / "models" / "wandb-artifacts" / key
    path = root / request.param
    path.parent.mkdir(parents=True)
    path.write_bytes(source.read_bytes())

    def add_member(member):
        name = member.relative_to(root).as_posix()
        artifact.manifest.entries[name] = ArtifactManifestEntry(path=name, digest=md5_file_b64(member))

    add_member(path)
    artifact.get_entry = lambda name: artifact.manifest.entries[name]
    # Only the remote transport is stubbed; verification is the installed SDK.
    artifact.download = lambda root: root
    artifact.verify = lambda root: wandb.Artifact.verify(artifact, root=root)
    run = SimpleNamespace(summary={"best_model_artifact": artifact.qualified_name})
    monkeypatch.setattr(wandb, "Api", lambda: SimpleNamespace(run=lambda _: run, artifact=lambda _: artifact))
    return SimpleNamespace(artifact=artifact, root=root, path=path, identity=identity,
                           add_member=add_member, reference="wandb:team/project/old-run")


@pytest.mark.parametrize("legacy_sidecar", [False, True])
def test_repeated_artifact_resolution_preserves_provenance(cached_artifact, legacy_sidecar):
    cached = cached_artifact
    original = cached.path.read_bytes()
    sidecar = cached.path.with_suffix(".pth.artifact.json")
    if legacy_sidecar:
        sidecar.write_text(json.dumps(cached.identity))
        with pytest.raises(ValueError, match="not a member"):
            cached.artifact.verify(str(cached.root))

    # Preview, training and subsequently opening the cached file all work.
    for source in (cached.reference, cached.reference, cached.path):
        resolved = resolve_model_source(source, progress=False)
        assert resolved.path == cached.path
        assert resolved.options["kind"] == "rfdetr" and resolved.options["resolution"] == 1296
        assert resolved.manifest["source_artifact"] == cached.identity
        cached.artifact.verify(str(cached.root))
        assert cached.path.read_bytes() == original
        assert not sidecar.exists()


@pytest.mark.parametrize("extra", ["unrelated.txt", "unknown.pth.artifact.json",
                                    "wrong-identity", "invalid-json"])
def test_cache_repair_does_not_remove_unknown_files(cached_artifact, extra):
    cached = cached_artifact
    path = cached.path.with_suffix(".pth.artifact.json") if extra in {"wrong-identity", "invalid-json"} else cached.root / extra
    content = "not json" if extra == "invalid-json" else json.dumps(
        {**cached.identity, "digest": "different"} if extra == "wrong-identity" else cached.identity)
    path.write_text(content)
    with pytest.raises(ValueError, match="not a member"):
        _download_wandb(cached.reference, requested=None, progress=False)
    assert path.read_text() == content


def test_cache_repair_preserves_sidecars_in_artifact_manifest(cached_artifact):
    cached = cached_artifact
    sidecar = cached.path.with_suffix(".pth.artifact.json")
    sidecar.write_text(json.dumps(cached.identity))
    cached.add_member(sidecar)
    _download_wandb(cached.reference, requested=None, progress=False)
    assert sidecar.is_file()
    cached.artifact.verify(str(cached.root))


def test_cache_repair_still_rejects_corrupt_weights(cached_artifact):
    cached = cached_artifact
    cached.path.with_suffix(".pth.artifact.json").write_text(json.dumps(cached.identity))
    cached.path.write_bytes(b"corrupt checkpoint")
    with pytest.raises(ValueError, match="Digest mismatch"):
        _download_wandb(cached.reference, requested=None, progress=False)


def test_local_legacy_sidecar_is_read_without_modifying_it(cached_artifact):
    cached = cached_artifact
    sidecar = cached.path.with_suffix(".pth.artifact.json")
    content = json.dumps(cached.identity)
    sidecar.write_text(content)
    resolved = resolve_model_source(cached.path, progress=False)
    assert resolved.manifest["source_artifact"] == cached.identity
    assert sidecar.read_text() == content
