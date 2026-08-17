from __future__ import annotations

from pathlib import Path

from dataset_fixer import Model, model_badges, model_full_label, model_label
from dataset_fixer.comparison.plot_labels import model_identity_chart


def test_plot_name_uses_provenance_instead_of_dataset_version_date() -> None:
    row = {
        "model": (
            "islands-128-08.08.2026-merged-1class_masks__gnsuhtfc__"
            "yolo26x-sem__512px"
        ),
        "source_key": (
            "wandb:max-planck-institute-for-animal-behavior/"
            "schools-segmentation/gnsuhtfc"
        ),
        "source_created_at": "2026-08-10T14:07:09+00:00",
        "digest": "abcdef1234567890",
        "model_type": "yolo26x-sem",
        "upscale_factor": 4,
        "input_size": (512, 512),
    }

    assert model_label(row) == "2026-08-10 14:07:09 · abcdef12"


def test_plot_name_reads_only_model_suffix_timestamp_after_dataset_prefix() -> None:
    row = {
        "model": (
            "08.08.2026-merged-1class_masks-yolo26x-sem-512px-"
            "2026-08-11_00-21_20260811_002334"
        ),
        "source_key": (
            "/missing/08.08.2026-merged-1class_masks-yolo26x-sem-512px-"
            "2026-08-11_00-21_20260811_002334.pt"
        ),
        "digest": "0123456789abcdef",
        "model_type": "yolo26x-sem",
    }

    assert model_label(row) == "2026-08-11 00:23:34 · 01234567"


def test_plot_name_allows_timestamp_without_checkpoint_digest() -> None:
    row = {
        "model": "external-model",
        "source_created_at": "2026-08-09T08:07:06Z",
        "model_type": "external-sem",
    }

    assert model_label(row) == "2026-08-09 08:07:06"


def test_plot_name_reads_unix_training_timestamp_in_utc() -> None:
    assert model_label({
        "model": "model", "source_created_at": "1786380390", "hash": "i9xve33c"
    }) == "2026-08-10 16:46:30 · i9xve33c"


def test_public_model_identity_helpers_accept_models(tmp_path: Path) -> None:
    checkpoint = tmp_path / "external.pt"
    checkpoint.write_bytes(b"external-checkpoint")
    model = Model(
        checkpoint,
        model_type="yolo26x-sem",
        source_created_at="2026-08-11T00:23:34+00:00",
        native_tile_size=128,
        upscale_factor=4,
    )

    assert model_label(model) == f"2026-08-11 00:23:34 · {model.digest[:8]}"
    assert [badge.text for badge in model_badges(model)] == [
        "yolo26x-sem",
        "4×",
        "512px",
    ]


def test_comparison_label_puts_creation_date_before_hash_when_both_are_requested() -> None:
    label = model_label(
        {
            "model": "manually named model",
            "hash": "i9xve33c",
            "source_created_at": "2026-08-10T16:46:30+00:00",
            "canonical_name": (
                "yolo26m-sem-with-long-configuration-1024px-2x-"
                "2026-08-10_16-46-30"
            ),
            "model_identity": "both",
        }
    )

    assert label.splitlines() == ["2026-08-10 16:46:30", "i9xve33c"]
    assert "yolo26m" not in label
    assert "1024px" not in label


def test_comparison_full_label_preserves_model_slugs() -> None:
    row = {
        "model": "model",
        "hash": "44c92357",
        "source_created_at": "2026-08-10T14:54:34+00:00",
        "model_type": "yolo26m-seg",
        "upscale_factor": 2,
        "resolution": 256,
        "model_identity": "hash",
    }
    label = model_full_label(row)

    assert label == "2026-08-10 14:54:34 · 44c92357 · yolo26m-seg · 2× · 256px"
    specification = str(model_identity_chart([row]).to_dict())
    assert all(value in specification for value in ("yolo26m-seg", "2×", "256px"))
    assert all(value in specification for value in ("#2563EB", "#B45309", "#475569"))
