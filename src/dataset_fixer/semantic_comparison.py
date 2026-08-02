from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .comparison.reporting import write_csv, write_json
from .errors import DatasetValidationError, ValidationIssue
from .models import SemanticComparisonResult, SemanticMaskExport
from .utils import (
    IMAGE_SUFFIXES,
    ensure_safe_destination,
    environment_snapshot,
    normalize_split,
    settings_fingerprint,
    sha256_file,
    slugify,
    to_jsonable,
)


@dataclass(frozen=True)
class _NNUNetModelSpec:
    name: str
    slug: str
    model_folder: Path
    folds: tuple[str, ...]
    checkpoint: str
    checkpoint_files: tuple[Path, ...]
    checkpoint_sha256: str
    model_sha256: str
    upscale_factor: int


@dataclass(frozen=True)
class _SemanticCase:
    case_id: str
    relative_path: Path
    image_path: Path
    mask_path: Path
    width: int
    height: int
    image_sha256: str
    mask_sha256: str


def compare_nnunet_models(
    export: SemanticMaskExport,
    models: Any,
    *,
    split: str,
    baseline: str | None,
    folds: tuple[int | str, ...],
    checkpoint: str,
    device: str,
    workers: int,
    bootstrap_resamples: int,
    seed: int,
    keep_predictions: bool,
    visualize: bool,
    progress: bool,
    destination: str | Path | None,
) -> SemanticComparisonResult:
    """Run official nnU-Net v2 prediction and evaluation for an export."""

    split = normalize_split(split)
    if split not in export.splits:
        raise ValueError(f"Unknown semantic-mask split {split!r}; available splits are {export.splits}")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be 'cpu', 'cuda', or 'mps'")

    specs = _parse_models(models, default_folds=folds, default_checkpoint=checkpoint)
    if baseline is None:
        baseline = specs[0].name
    if baseline not in {spec.name for spec in specs}:
        raise ValueError(f"Unknown baseline {baseline!r}")
    _require_official_commands()
    cases, cohort_fingerprint = _freeze_cohort(export, split)

    resolved_settings = {
        "backend": "nnunetv2-official",
        "split": split,
        "baseline": baseline,
        "device": device,
        "workers": workers,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "keep_predictions": keep_predictions,
        "visualize": visualize,
        "models": [
            {
                "name": spec.name,
                "model_folder": spec.model_folder,
                "folds": spec.folds,
                "checkpoint": spec.checkpoint,
                "checkpoint_sha256": spec.checkpoint_sha256,
                "model_sha256": spec.model_sha256,
                "upscale_factor": spec.upscale_factor,
            }
            for spec in specs
        ],
    }
    fingerprint = settings_fingerprint(
        {**to_jsonable(resolved_settings), "cohort_fingerprint": cohort_fingerprint}
    )
    target = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else export.location.parent / f"{export.name}__compare-nnunet__{fingerprint}"
    )
    ensure_safe_destination(export.location, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent))
    started = time.time()
    limitations = [
        "Metrics are produced by the official nnU-Net v2 folder evaluator on binary foreground masks.",
        "Training/evaluation overlap cannot be independently verified from an nnU-Net model folder alone.",
        "Paired uncertainty treats exported cases as independent; tiled cases from one source image may be correlated.",
    ]
    if any(spec.upscale_factor != 1 for spec in specs):
        limitations.append(
            "Each model received inputs at its configured training-adapter scale; predicted masks were "
            "nearest-neighbor resized back to the canonical exported resolution before evaluation."
        )

    try:
        canonical_labels = temporary / "cohort" / "canonical" / "labels"
        _prepare_labels(cases, canonical_labels)
        _write_cohort(temporary / "evaluation-cohort.jsonl", cases, split, specs)

        model_rows: dict[str, list[dict[str, Any]]] = {}
        ranking: list[dict[str, Any]] = []
        official_summaries: dict[str, str] = {}
        prediction_dirs: dict[str, Path] = {}
        for model_index, spec in enumerate(specs):
            prepared_images = temporary / "cohort" / "models" / spec.slug / "images"
            _prepare_images(
                cases,
                prepared_images,
                upscale_factor=spec.upscale_factor,
                progress=progress,
                model_name=spec.name,
            )
            native_prediction_dir = temporary / "native-predictions" / spec.slug
            native_prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction_dir = temporary / "predictions" / spec.slug
            prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction_dirs[spec.name] = prediction_dir
            summary_path = temporary / "reports" / "official" / f"{spec.slug}.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"Evaluating {spec.name!r} with official nnU-Net v2 "
                f"(folds={spec.folds}, checkpoint={spec.checkpoint}, "
                f"input_scale={spec.upscale_factor}x)"
            )
            inference_started = time.perf_counter()
            _run_command(
                [
                    "nnUNetv2_predict_from_modelfolder",
                    "-i",
                    str(prepared_images),
                    "-o",
                    str(native_prediction_dir),
                    "-m",
                    str(spec.model_folder),
                    "-f",
                    *spec.folds,
                    "-chk",
                    spec.checkpoint,
                    "-device",
                    device,
                    "-npp",
                    str(workers),
                    "-nps",
                    str(workers),
                ]
            )
            inference_seconds = time.perf_counter() - inference_started
            _assert_exact_predictions(native_prediction_dir, cases, spec.name)
            _canonicalize_predictions(
                native_prediction_dir,
                prediction_dir,
                cases,
                model_name=spec.name,
                upscale_factor=spec.upscale_factor,
            )
            _assert_exact_predictions(prediction_dir, cases, spec.name)
            shutil.rmtree(native_prediction_dir)
            _run_command(
                [
                    "nnUNetv2_evaluate_folder",
                    str(canonical_labels),
                    str(prediction_dir),
                    "-djfile",
                    str(spec.model_folder / "dataset.json"),
                    "-pfile",
                    str(spec.model_folder / "plans.json"),
                    "-o",
                    str(summary_path),
                    "-np",
                    str(workers),
                ]
            )
            summary = _load_official_summary(summary_path, spec.name)
            rows = _per_case_rows(summary, cases, spec.name)
            model_rows[spec.name] = rows
            aggregate = summary["foreground_mean"]
            dice = _metric(aggregate, "Dice")
            iou = _metric(aggregate, "IoU")
            ci_low, ci_high = _bootstrap_interval(
                [row["dice"] for row in rows],
                resamples=bootstrap_resamples,
                seed=seed + model_index,
            )
            ranking.append(
                {
                    "model": spec.name,
                    "backend": "nnunetv2-official",
                    "metric": "foreground_mean.Dice",
                    "score": dice,
                    "dice": dice,
                    "iou": iou,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "support_cases": len(cases),
                    "folds": ",".join(spec.folds),
                    "checkpoint": spec.checkpoint,
                    "checkpoint_sha256": spec.checkpoint_sha256,
                    "model_sha256": spec.model_sha256,
                    "model_folder": str(spec.model_folder),
                    "upscale_factor": spec.upscale_factor,
                    "evaluation_resolution": "canonical-export",
                    "cohort_fingerprint": cohort_fingerprint,
                    "inference_seconds": inference_seconds,
                    "throughput_cases_per_second": (
                        len(cases) / inference_seconds if inference_seconds > 0 else None
                    ),
                }
            )
            sanitized = _sanitize_official_summary(
                summary,
                cases,
                split=split,
                model_slug=spec.slug,
                keep_predictions=keep_predictions,
            )
            write_json(summary_path, sanitized)
            official_summaries[spec.name] = str(summary_path.relative_to(temporary))

        ranking.sort(key=lambda row: (-_sortable_score(row["score"]), row["model"]))
        for rank, row in enumerate(ranking, start=1):
            row["rank"] = rank
        paired = _paired_statistics(
            model_rows,
            baseline,
            resamples=bootstrap_resamples,
            seed=seed,
        )
        per_case = [row for name in [spec.name for spec in specs] for row in model_rows[name]]
        write_csv(temporary / "metrics" / "ranking.csv", ranking)
        write_csv(temporary / "metrics" / "per_case.csv", per_case)
        write_csv(temporary / "metrics" / "paired_statistics.csv", paired)
        write_json(temporary / "reports" / "limitations.json", {"limitations": limitations})

        figure_paths: list[str] = []
        qualitative_paths: list[str] = []
        if visualize:
            figure_paths = _render_ranking(temporary, ranking, cohort_fingerprint)
            qualitative_paths = _render_qualitative(
                temporary,
                cases,
                prediction_dirs,
                model_rows,
                seed=seed,
            )

        if not keep_predictions:
            shutil.rmtree(temporary / "predictions", ignore_errors=True)
        shutil.rmtree(temporary / "native-predictions", ignore_errors=True)
        shutil.rmtree(temporary / "cohort", ignore_errors=True)

        manifest = {
            "schema": 1,
            "kind": "semantic-mask-model-comparison",
            "backend": "nnunetv2-official",
            "dataset": {
                "name": export.name,
                "location": str(export.location),
                "format": export.manifest.get("format"),
                "class_handling": export.manifest.get("class_handling"),
            },
            "cohort_fingerprint": cohort_fingerprint,
            "cohort_verified": True,
            "split": split,
            "cases": len(cases),
            "baseline": baseline,
            "settings": resolved_settings,
            "settings_fingerprint": fingerprint,
            "ranking": ranking,
            "paired_statistics": paired,
            "official_summaries": official_summaries,
            "limitations": limitations,
            "figures": figure_paths,
            "qualitative": qualitative_paths,
            "environment": environment_snapshot(),
            "started_at_unix": started,
            "completed_at_unix": time.time(),
        }
        write_json(temporary / "semantic-model-comparison.json", manifest)
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"Semantic model comparison complete: {target}\n"
        f"Cohort verified: yes; cases: {len(cases)}; baseline: {baseline}"
    )
    return SemanticComparisonResult(
        location=target,
        ranking=tuple(ranking),
        cohort_fingerprint=cohort_fingerprint,
        cohort_verified=True,
        split=split,
        baseline=baseline,
        settings=resolved_settings,
        limitations=tuple(limitations),
    )


def _parse_models(
    models: Any,
    *,
    default_folds: tuple[int | str, ...],
    default_checkpoint: str,
) -> list[_NNUNetModelSpec]:
    if isinstance(models, (str, Path)):
        items = [(Path(models).name, models)]
    elif isinstance(models, Mapping):
        items = list(models.items())
    elif isinstance(models, Sequence):
        items = [(Path(value).name, value) for value in models]
    else:
        raise TypeError(
            "models must be an nnU-Net model folder, a sequence of folders, or a name-to-model mapping"
        )
    if not items:
        raise ValueError("At least one nnU-Net model is required")

    issues: list[ValidationIssue] = []
    specs: list[_NNUNetModelSpec] = []
    seen_names: set[str] = set()
    seen_slugs: set[str] = set()
    for raw_name, value in items:
        name = str(raw_name).strip()
        if not name or name in seen_names:
            issues.append(ValidationIssue("Model names must be non-empty and unique", value=name))
            continue
        seen_names.add(name)
        slug = slugify(name)
        if slug in seen_slugs:
            issues.append(
                ValidationIssue(
                    "Model names must remain unique after filename normalization",
                    value=name,
                    expected="distinct filesystem-safe names",
                )
            )
            continue
        seen_slugs.add(slug)
        settings = dict(value) if isinstance(value, Mapping) else {"model_folder": value}
        selected_upscale_factor = settings.get("upscale_factor", 1)
        if (
            not isinstance(selected_upscale_factor, int)
            or isinstance(selected_upscale_factor, bool)
            or selected_upscale_factor <= 0
        ):
            issues.append(
                ValidationIssue(
                    "Model upscale_factor must be a positive integer",
                    source=name,
                    value=selected_upscale_factor,
                )
            )
            continue
        raw_folder = settings.get("model_folder", settings.get("path"))
        if raw_folder is None:
            issues.append(ValidationIssue("nnU-Net model specification is missing model_folder", source=name))
            continue
        model_folder = Path(raw_folder).expanduser().resolve()
        explicit_folds = "folds" in settings
        selected_folds = _normalize_folds(settings.get("folds", default_folds), name, issues)
        if model_folder.name.startswith("fold_") and not (model_folder / "plans.json").is_file():
            inferred_fold = model_folder.name.removeprefix("fold_")
            model_folder = model_folder.parent
            if not explicit_folds:
                selected_folds = (inferred_fold,)
        selected_checkpoint = str(settings.get("checkpoint", default_checkpoint)).strip()
        if not selected_checkpoint or Path(selected_checkpoint).name != selected_checkpoint:
            issues.append(
                ValidationIssue(
                    "checkpoint must be a filename within each selected fold",
                    source=name,
                    value=selected_checkpoint,
                )
            )
            continue
        if not model_folder.is_dir():
            issues.append(
                ValidationIssue("nnU-Net model folder does not exist", source=name, value=str(model_folder))
            )
            continue
        if not selected_folds:
            continue
        dataset_json = model_folder / "dataset.json"
        plans_json = model_folder / "plans.json"
        for required in (dataset_json, plans_json):
            if not required.is_file():
                issues.append(
                    ValidationIssue(
                        "Incomplete nnU-Net model folder",
                        source=name,
                        value=str(required),
                        expected="dataset.json, plans.json, and selected fold checkpoints",
                    )
                )
        if not dataset_json.is_file() or not plans_json.is_file():
            continue
        if not _validate_model_dataset_json(dataset_json, name, issues):
            continue
        checkpoint_files = tuple(
            model_folder / f"fold_{fold}" / selected_checkpoint for fold in selected_folds
        )
        missing = [str(path) for path in checkpoint_files if not path.is_file()]
        if missing:
            issues.append(
                ValidationIssue(
                    "nnU-Net checkpoint is missing",
                    source=name,
                    value=missing,
                    expected=f"{selected_checkpoint} in every selected fold",
                )
            )
            continue
        checkpoint_sha256 = _combined_sha256(checkpoint_files, relative_to=model_folder)
        model_sha256 = _combined_sha256(
            (dataset_json, plans_json, *checkpoint_files),
            relative_to=model_folder,
        )
        specs.append(
            _NNUNetModelSpec(
                name=name,
                slug=slug,
                model_folder=model_folder,
                folds=selected_folds,
                checkpoint=selected_checkpoint,
                checkpoint_files=checkpoint_files,
                checkpoint_sha256=checkpoint_sha256,
                model_sha256=model_sha256,
                upscale_factor=selected_upscale_factor,
            )
        )
    if issues:
        raise DatasetValidationError(issues)
    return specs


def _normalize_folds(
    value: Any,
    name: str,
    issues: list[ValidationIssue],
) -> tuple[str, ...]:
    if isinstance(value, (str, int)):
        raw_folds = (value,)
    else:
        try:
            raw_folds = tuple(value)
        except TypeError:
            issues.append(ValidationIssue("Invalid nnU-Net folds", source=name, value=value))
            return ()
    folds: list[str] = []
    for raw in raw_folds:
        fold = str(raw).strip()
        if fold != "all":
            try:
                if int(fold) < 0 or str(int(fold)) != fold:
                    raise ValueError
            except ValueError:
                issues.append(
                    ValidationIssue(
                        "Invalid nnU-Net fold",
                        source=name,
                        value=raw,
                        expected="a non-negative integer or 'all'",
                    )
                )
                continue
        if fold not in folds:
            folds.append(fold)
    if not folds:
        issues.append(ValidationIssue("At least one nnU-Net fold is required", source=name))
    return tuple(folds)


def _validate_model_dataset_json(
    path: Path,
    name: str,
    issues: list[ValidationIssue],
) -> bool:
    try:
        dataset = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(ValidationIssue(f"Unreadable nnU-Net dataset.json: {exc}", source=name))
        return False
    if dataset.get("file_ending") != ".png":
        issues.append(
            ValidationIssue(
                "SemanticMaskExport comparison requires an nnU-Net PNG model",
                source=name,
                value=dataset.get("file_ending"),
                expected="file_ending='.png'",
            )
        )
        return False
    labels = dataset.get("labels") or {}
    values: set[int] = set()
    try:
        for value in labels.values():
            if isinstance(value, list):
                values.update(int(item) for item in value)
            else:
                values.add(int(value))
    except (TypeError, ValueError):
        values = set()
    if values != {0, 1}:
        issues.append(
            ValidationIssue(
                "SemanticMaskExport comparison requires binary nnU-Net labels",
                source=name,
                value=labels,
                expected="background=0 and one foreground label=1",
            )
        )
        return False
    return True


def _combined_sha256(paths: tuple[Path, ...], *, relative_to: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(relative_to).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _require_official_commands() -> None:
    missing = [
        command
        for command in ("nnUNetv2_predict_from_modelfolder", "nnUNetv2_evaluate_folder")
        if shutil.which(command) is None
    ]
    if missing:
        raise ImportError(
            "Official nnU-Net v2 commands are unavailable: "
            f"{', '.join(missing)}. Install the pinned notebook dependency with "
            "`pip install nnunetv2==2.8.1`."
        )


def _freeze_cohort(
    export: SemanticMaskExport,
    split: str,
) -> tuple[list[_SemanticCase], str]:
    image_root = export.image_dirs[split]
    mask_root = export.mask_dirs[split]
    if not image_root.is_dir() or not mask_root.is_dir():
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic-mask split directories are missing",
                source=split,
                value={"images": str(image_root), "masks": str(mask_root)},
            )
        )
    images = sorted(
        path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise DatasetValidationError(f"No images found in semantic-mask split {split!r}")

    cases: list[_SemanticCase] = []
    expected_masks: set[Path] = set()
    digest = hashlib.sha256()
    digest.update(f"semantic-mask-cohort-v2:{split}:canonical-export".encode("utf-8"))
    for index, image_path in enumerate(images):
        relative = image_path.relative_to(image_root)
        mask_path = mask_root / relative.with_suffix(".png")
        expected_masks.add(mask_path.resolve())
        if not mask_path.is_file():
            raise DatasetValidationError(
                ValidationIssue(
                    "Semantic mask is missing for evaluation image",
                    source=str(image_path),
                    expected=str(mask_path),
                )
            )
        with Image.open(image_path) as opened_image, Image.open(mask_path) as opened_mask:
            image = opened_image.convert("RGB")
            mask = opened_mask.convert("L")
            if image.size != mask.size:
                raise DatasetValidationError(
                    f"Semantic mask dimensions {mask.size} do not match image dimensions {image.size}: {mask_path}"
                )
            values = set(mask.getdata())
            if not values <= {0, 1, 255}:
                raise DatasetValidationError(
                    f"Semantic mask contains values outside 0/1/255: {mask_path}: {sorted(values)[:10]}"
                )
            width, height = image.size
        image_sha = sha256_file(image_path)
        mask_sha = sha256_file(mask_path)
        case = _SemanticCase(
            case_id=f"{split}_{index:06d}",
            relative_path=relative,
            image_path=image_path,
            mask_path=mask_path,
            width=width,
            height=height,
            image_sha256=image_sha,
            mask_sha256=mask_sha,
        )
        cases.append(case)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(image_sha.encode("ascii"))
        digest.update(mask_sha.encode("ascii"))
    actual_masks = {
        path.resolve()
        for path in mask_root.rglob("*.png")
        if path.is_file()
    }
    if actual_masks != expected_masks:
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic-mask split contains orphan or missing masks",
                source=split,
                value={
                    "unexpected": [str(path) for path in sorted(actual_masks - expected_masks)],
                    "missing": [str(path) for path in sorted(expected_masks - actual_masks)],
                },
            )
        )
    return cases, digest.hexdigest()


def _prepare_labels(
    cases: list[_SemanticCase],
    label_dir: Path,
) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        with Image.open(case.mask_path) as opened_mask:
            mask = opened_mask.convert("L").point(lambda value: 1 if value else 0)
        mask.save(label_dir / f"{case.case_id}.png", format="PNG", optimize=False)


def _prepare_images(
    cases: list[_SemanticCase],
    image_dir: Path,
    *,
    upscale_factor: int,
    progress: bool,
    model_name: str,
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    iterator = tqdm(
        cases,
        desc=f"Preparing {model_name} inputs ({upscale_factor}x)",
        unit="image",
        disable=not progress,
    )
    for case in iterator:
        with Image.open(case.image_path) as opened_image:
            image = opened_image.convert("RGB")
        if upscale_factor != 1:
            size = (image.width * upscale_factor, image.height * upscale_factor)
            image = image.resize(size, Image.Resampling.BICUBIC)
        image.save(image_dir / f"{case.case_id}_0000.png", format="PNG")


def _write_cohort(
    path: Path,
    cases: list[_SemanticCase],
    split: str,
    specs: list[_NNUNetModelSpec],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "split": split,
                        "relative_path": case.relative_path.as_posix(),
                        "source_image": str(case.image_path),
                        "source_mask": str(case.mask_path),
                        "width": case.width,
                        "height": case.height,
                        "evaluation_resolution": "canonical-export",
                        "model_inputs": {
                            spec.name: {
                                "upscale_factor": spec.upscale_factor,
                                "prepared_width": case.width * spec.upscale_factor,
                                "prepared_height": case.height * spec.upscale_factor,
                            }
                            for spec in specs
                        },
                        "image_sha256": case.image_sha256,
                        "mask_sha256": case.mask_sha256,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _run_command(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            text=True,
            # nnU-Net prints several lines and nested progress bars for every
            # case. Capturing keeps large evaluations readable while retaining
            # stdout/stderr for the failure diagnostic below.
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise ImportError(
            f"Official nnU-Net command is unavailable: {command[0]}. "
            "Install `nnunetv2==2.8.1`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = f"Official nnU-Net command failed ({exc.returncode}): {' '.join(command)}"
        if detail:
            message += f"\n{detail[-4000:]}"
        raise RuntimeError(message) from exc


def _assert_exact_predictions(
    prediction_dir: Path,
    cases: list[_SemanticCase],
    model_name: str,
) -> None:
    expected = {f"{case.case_id}.png" for case in cases}
    actual = {path.name for path in prediction_dir.glob("*.png") if path.is_file()}
    if actual != expected:
        raise DatasetValidationError(
            ValidationIssue(
                "nnU-Net predictions do not match the frozen evaluation cohort",
                source=model_name,
                value={
                    "unexpected": sorted(actual - expected)[:20],
                    "missing": sorted(expected - actual)[:20],
                },
                expected=f"exactly {len(expected)} prediction masks",
            )
        )


def _canonicalize_predictions(
    native_prediction_dir: Path,
    canonical_prediction_dir: Path,
    cases: list[_SemanticCase],
    *,
    model_name: str,
    upscale_factor: int,
) -> None:
    """Project a model's native-scale masks onto the frozen source raster."""

    canonical_prediction_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        source = native_prediction_dir / f"{case.case_id}.png"
        with Image.open(source) as opened:
            prediction = opened.convert("L")
        expected_native_size = (
            case.width * upscale_factor,
            case.height * upscale_factor,
        )
        if prediction.size != expected_native_size:
            raise DatasetValidationError(
                ValidationIssue(
                    "nnU-Net prediction dimensions do not match the model input adapter",
                    source=f"{model_name}/{case.case_id}",
                    value=prediction.size,
                    expected=str(expected_native_size),
                )
            )
        values = set(prediction.getdata())
        if not values <= {0, 1}:
            raise DatasetValidationError(
                ValidationIssue(
                    "nnU-Net prediction contains labels outside the binary model schema",
                    source=f"{model_name}/{case.case_id}",
                    value=sorted(values)[:20],
                    expected="0 or 1",
                )
            )
        if prediction.size != (case.width, case.height):
            prediction = prediction.resize(
                (case.width, case.height),
                Image.Resampling.NEAREST,
            )
        prediction.save(
            canonical_prediction_dir / f"{case.case_id}.png",
            format="PNG",
            optimize=False,
        )


def _load_official_summary(path: Path, model_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetValidationError(
            f"Official nnU-Net evaluator did not write its summary for {model_name}: {path}"
        )
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            f"Unreadable official nnU-Net evaluation summary for {model_name}: {exc}"
        ) from exc
    if not isinstance(summary.get("foreground_mean"), dict) or not isinstance(
        summary.get("metric_per_case"), list
    ):
        raise DatasetValidationError(
            f"Official nnU-Net evaluation summary for {model_name} lacks foreground_mean or metric_per_case"
        )
    return summary


def _per_case_rows(
    summary: dict[str, Any],
    cases: list[_SemanticCase],
    model_name: str,
) -> list[dict[str, Any]]:
    by_case = {case.case_id: case for case in cases}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in summary["metric_per_case"]:
        case_id = Path(str(result.get("prediction_file", ""))).stem
        case = by_case.get(case_id)
        if case is None:
            raise DatasetValidationError(
                f"Official nnU-Net summary for {model_name} contains unknown case {case_id!r}"
            )
        metrics = result.get("metrics") or {}
        foreground = metrics.get("1") or metrics.get(1)
        if not isinstance(foreground, dict):
            foreground_values = [
                value
                for key, value in metrics.items()
                if str(key) not in {"0", "background"} and isinstance(value, dict)
            ]
            if len(foreground_values) != 1:
                raise DatasetValidationError(
                    f"Official nnU-Net summary for {model_name}/{case_id} is not binary foreground data"
                )
            foreground = foreground_values[0]
        rows.append(
            {
                "model": model_name,
                "case_id": case_id,
                "relative_path": case.relative_path.as_posix(),
                "source_image": str(case.image_path),
                "source_mask": str(case.mask_path),
                "dice": _metric(foreground, "Dice"),
                "iou": _metric(foreground, "IoU"),
                "tp": _metric(foreground, "TP"),
                "fp": _metric(foreground, "FP"),
                "fn": _metric(foreground, "FN"),
                "tn": _metric(foreground, "TN"),
                "n_pred": _metric(foreground, "n_pred"),
                "n_ref": _metric(foreground, "n_ref"),
            }
        )
        seen.add(case_id)
    missing = set(by_case) - seen
    if missing:
        raise DatasetValidationError(
            f"Official nnU-Net summary for {model_name} is missing {len(missing)} frozen cases"
        )
    return rows


def _metric(values: Mapping[str, Any], key: str) -> float:
    value = values.get(key, math.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _bootstrap_interval(
    values: list[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(array) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def _paired_statistics(
    rows_by_model: dict[str, list[dict[str, Any]]],
    baseline: str,
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    baseline_scores = {
        row["case_id"]: row["dice"]
        for row in rows_by_model[baseline]
        if math.isfinite(row["dice"])
    }
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for name, rows in rows_by_model.items():
        if name == baseline:
            continue
        scores = {
            row["case_id"]: row["dice"]
            for row in rows
            if math.isfinite(row["dice"])
        }
        keys = sorted(set(baseline_scores) & set(scores))
        differences = np.asarray(
            [scores[key] - baseline_scores[key] for key in keys],
            dtype=float,
        )
        if not len(differences):
            continue
        samples = rng.choice(
            differences,
            size=(resamples, len(differences)),
            replace=True,
        ).mean(axis=1)
        signs = rng.choice((-1.0, 1.0), size=(resamples, len(differences)))
        randomized = (differences * signs).mean(axis=1)
        p_value = float(
            (np.sum(np.abs(randomized) >= abs(differences.mean())) + 1)
            / (resamples + 1)
        )
        raw_p.append(p_value)
        output.append(
            {
                "model": name,
                "baseline": baseline,
                "metric": "per_case.Dice",
                "difference": float(differences.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "p_value": p_value,
                "paired_cases": len(differences),
                "wins": int(np.sum(differences > 0)),
                "ties": int(np.sum(differences == 0)),
                "losses": int(np.sum(differences < 0)),
            }
        )
    order = sorted(range(len(raw_p)), key=lambda index: raw_p[index])
    adjusted = [0.0] * len(raw_p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, raw_p[index] * (len(raw_p) - rank)))
        adjusted[index] = running
    for row, value in zip(output, adjusted):
        row["p_value_holm"] = value
    return output


def _sanitize_official_summary(
    summary: dict[str, Any],
    cases: list[_SemanticCase],
    *,
    split: str,
    model_slug: str,
    keep_predictions: bool,
) -> dict[str, Any]:
    case_map = {case.case_id: case for case in cases}
    sanitized = json.loads(json.dumps(summary))
    for result in sanitized.get("metric_per_case", []):
        case_id = Path(str(result.get("prediction_file", ""))).stem
        case = case_map.get(case_id)
        if case is None:
            continue
        result["reference_file"] = (
            f"dataset://{split}/masks/0/{case.relative_path.with_suffix('.png').as_posix()}"
        )
        result["prediction_file"] = (
            f"predictions/{model_slug}/{case_id}.png" if keep_predictions else None
        )
        result["source_image"] = f"dataset://{split}/images/{case.relative_path.as_posix()}"
    sanitized["predictions_retained"] = keep_predictions
    return sanitized


def _sortable_score(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return parsed if math.isfinite(parsed) else -math.inf


def _render_ranking(
    root: Path,
    ranking: list[dict[str, Any]],
    cohort_fingerprint: str,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = list(reversed(ranking))
    figure, axis = plt.subplots(figsize=(8, max(3.5, 0.7 * len(ordered) + 1.5)))
    names = [row["model"] for row in ordered]
    scores = [
        float(row["dice"]) if math.isfinite(float(row["dice"])) else 0.0
        for row in ordered
    ]
    axis.barh(names, scores, color="#0072B2")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Official foreground mean Dice")
    axis.set_title("nnU-Net semantic-mask model comparison")
    for index, (score, row) in enumerate(zip(scores, ordered)):
        label = f"{float(row['dice']):.3f}" if math.isfinite(float(row["dice"])) else "n/a"
        axis.text(min(score + 0.01, 0.98), index, label, va="center")
    figure.tight_layout()
    paths: list[str] = []
    for suffix, dpi in (("png", 180), ("pdf", None), ("svg", None)):
        path = root / "figures" / f"ranking.{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(str(path.relative_to(root)))
    plt.close(figure)
    write_csv(root / "figures" / "data" / "ranking.csv", ranking)
    write_json(
        root / "figures" / "metadata" / "ranking.json",
        {
            "metric": "foreground_mean.Dice",
            "cohort_fingerprint": cohort_fingerprint,
            "models": names,
        },
    )
    return paths


def _render_qualitative(
    root: Path,
    cases: list[_SemanticCase],
    prediction_dirs: dict[str, Path],
    rows_by_model: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    indices = sorted(
        rng.choice(len(cases), size=min(8, len(cases)), replace=False).tolist()
    )
    selected = [cases[index] for index in indices]
    row_lookup = {
        name: {row["case_id"]: row for row in rows}
        for name, rows in rows_by_model.items()
    }
    model_names = list(prediction_dirs)
    figure, axes = plt.subplots(
        len(selected),
        2 + len(model_names),
        figsize=(4 * (2 + len(model_names)), 3.5 * len(selected)),
        squeeze=False,
    )
    for row_index, case in enumerate(selected):
        with Image.open(case.image_path) as opened_image:
            image = np.asarray(opened_image.convert("RGB"))
        with Image.open(case.mask_path) as opened_mask:
            truth = np.asarray(opened_mask.convert("L")) > 0
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(case.relative_path.name)
        axes[row_index, 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title("Ground truth")
        for model_index, name in enumerate(model_names, start=2):
            prediction_path = prediction_dirs[name] / f"{case.case_id}.png"
            with Image.open(prediction_path) as opened_prediction:
                prediction = np.asarray(opened_prediction.convert("L")) > 0
            if prediction.shape != truth.shape:
                prediction = np.asarray(
                    Image.fromarray(prediction.astype(np.uint8)).resize(
                        (truth.shape[1], truth.shape[0]),
                        Image.Resampling.NEAREST,
                    )
                ) > 0
            overlay = image.astype(np.float32) / 255.0
            true_positive = truth & prediction
            false_negative = truth & ~prediction
            false_positive = ~truth & prediction
            _paint(overlay, true_positive, (1.0, 1.0, 0.0))
            _paint(overlay, false_negative, (0.0, 1.0, 0.0))
            _paint(overlay, false_positive, (1.0, 0.0, 1.0))
            metric = row_lookup[name][case.case_id]
            axes[row_index, model_index].imshow(np.clip(overlay, 0, 1))
            axes[row_index, model_index].set_title(
                f"{name}\nDice={metric['dice']:.3f} · IoU={metric['iou']:.3f}"
            )
        for column in range(2 + len(model_names)):
            axes[row_index, column].axis("off")
    figure.tight_layout()
    output = root / "qualitative" / "comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [str(output.relative_to(root))]


def _paint(
    overlay: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float],
) -> None:
    color_array = np.asarray(color, dtype=np.float32)
    overlay[mask] = 0.35 * overlay[mask] + 0.65 * color_array
