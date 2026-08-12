from __future__ import annotations

from pathlib import Path

from dataset_fixer import Model, model_badges, model_label


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
