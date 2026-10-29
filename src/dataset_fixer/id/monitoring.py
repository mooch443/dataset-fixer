"""Bounded epoch monitoring without a second logger or training loop."""
import time
import warnings
from contextlib import contextmanager, nullcontext

import numpy as np
import pandas as pd

from .data import balanced_rows, index_of

_MAX_EMBEDDING_SAMPLES = 2048


def _monitor_notice(result, key, message):
    """Report a diagnostic fallback once per reason, including in run metadata."""
    if not hasattr(result, 'metadata'):
        result.metadata = {}
    result.metadata.setdefault('monitoring', {})['notice'] = message
    notices = getattr(result, '_embedding_notices', set())
    if key not in notices:
        notices.add(key)
        result._embedding_notices = notices
        print(f"⚠ Embedding monitor: {message}", flush=True)
    if result.wandb_run is not None:
        try:
            result.wandb_run.summary['embeddings/notice'] = message
        except Exception:
            pass  # Diagnostics must also survive a broken media/logging sink.


def _monitor_cohort(index, count, seed):
    chosen = []
    for split in ('train', 'val'):
        rows = np.flatnonzero((index.frame.split == split) & index.frame.identity_id.notna())
        budget = count//2 + (count % 2 if split == 'train' else 0)
        chosen.extend(rows[balanced_rows(index.take(rows), budget, seed)])
    return index.take(chosen)


def _save_monitor_cohort(result):
    from ..sources import _atomic_json
    path = result.output_dir / 'monitoring' / 'cohort.json'
    _atomic_json(path, [dict(source_ref=s.image_ref.key, identity_id=s.identity_id,
        clip_id=s.clip_id, split=s.split) for s in result._embedding_cohort])
    result.auxiliary_files['monitoring/cohort.json'] = path


@contextmanager
def evaluation_progress(result, split):
    """Mirror post-training stages to the session-owned run, independently of stdout."""
    run = result.wandb_run
    if run is None:
        yield None
        return
    prefix = f"evaluation/{split}"
    run.define_metric(f"{prefix}/elapsed_seconds")
    run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/elapsed_seconds")
    started = time.monotonic()

    def report(stage, completed, total):
        status = "complete" if stage == "report" and completed == total else stage
        try:
            run.summary[f"{prefix}/stage"] = status
            run.log({f"{prefix}/elapsed_seconds": time.monotonic() - started,
                     f"{prefix}/{stage}_completed": completed, f"{prefix}/{stage}_total": total})
        except Exception as exc:
            warnings.warn(f"W&B evaluation progress logging failed: {exc}")

    report("starting", 0, 1)
    try:
        yield report
    except BaseException:
        report("failed", 0, 1)
        raise


def prepare_monitor(result):
    count = result.config.backend_options.get("embedding_samples", 512)
    method = result.config.backend_options.get("embedding_method", "tsne")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("embedding_samples must be a non-negative integer (0 disables monitoring)")
    if method not in {"tsne", "pca"}:
        raise ValueError("embedding_method must be tsne or pca")
    result._embedding_cohort = None
    result._embedding_disabled = False
    result._embedding_batch_size = min(result.config.batch_size or 64, 64)
    if count > _MAX_EMBEDDING_SAMPLES:
        _monitor_notice(result, 'limit', f"requested {count:,} samples; using at most "
                        f"{_MAX_EMBEDDING_SAMPLES:,} to bound diagnostic memory and plotting work.")
    try:
        if count:
            result._embedding_cohort = _monitor_cohort(index_of(result.dataset),
                min(count, _MAX_EMBEDDING_SAMPLES), result.config.seed)
            _save_monitor_cohort(result)
    except Exception as error:
        result._embedding_disabled = True
        _monitor_notice(result, 'disabled', f"disabled after cohort preparation failed "
                        f"({type(error).__name__}: {error}). Training will continue.")
    run = result.wandb_run
    if run is not None:
        run.define_metric("epoch")
        for namespace in ("train/*", "val/*", "optim/*", "time/*", "embeddings/*"):
            run.define_metric(namespace, step_metric="epoch")
        run.define_metric("val/loss", step_metric="epoch", summary="min")
        run.define_metric("val/monitor_rank1", step_metric="epoch", summary="max")
        run.define_metric("progress/global_step")
        run.define_metric("progress/*", step_metric="progress/global_step")
        for name in ('loss', 'size', 'identities'):
            run.define_metric(f"train/batch_{name}", step_metric="progress/global_step")


def monitor_epoch(result, module, epoch):
    """Best-effort diagnostics; resource failures shrink the next epoch's cohort."""
    index = result._embedding_cohort
    if getattr(result, '_embedding_disabled', False):
        return {'embeddings/samples': 0, 'embeddings/skipped': 1}
    if index is None or len(index) < 3:
        return {'embeddings/samples': 0 if index is None else len(index)}
    try:
        return _monitor_epoch(result, module, epoch)
    except Exception as error:
        problem = f"{type(error).__name__}: {error}"
        resource_failure = isinstance(error, MemoryError) or 'out of memory' in str(error).lower()
    # Outside the except block so the traceback no longer retains failed tensors.
    try:
        if resource_failure:
            size = result._embedding_batch_size
            count = max(3, len(index)//2)
            smaller_batch = max(1, min(count, size//2))
            if count < len(index) or smaller_batch < size:
                result._embedding_cohort = _monitor_cohort(index, count, result.config.seed)
                result._embedding_batch_size = smaller_batch
                import torch
                device = next(module.parameters()).device.type
                if device == 'cuda':
                    torch.cuda.empty_cache()
                elif device == 'mps':
                    torch.mps.empty_cache()
                _monitor_notice(result, 'resource', f"skipped epoch {epoch+1} ({problem}); "
                    f"next attempt uses {len(result._embedding_cohort)} samples and batches of {smaller_batch}. "
                    "The reduced cohort stays fixed after recovery; its metrics use a different cohort.")
                return {'embeddings/samples': 0, 'embeddings/skipped': 1}
    except Exception as error:
        problem = f"{problem}; recovery failed: {type(error).__name__}: {error}"
    result._embedding_disabled = True
    _monitor_notice(result, 'disabled', f"disabled after epoch {epoch+1} ({problem}). Training will continue.")
    return {'embeddings/samples': 0, 'embeddings/skipped': 1}


def _monitor_epoch(result, module, epoch):
    import torch
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from .model import preprocess
    from .inspection import embedding_chart
    from ..static_rendering import save_chart
    from ..training.preview import _preview_seed
    index = result._embedding_cohort
    if len(index) < 3:
        return {"embeddings/samples": len(index)}
    method = result.config.backend_options.get("embedding_method", "tsne")
    size = result._embedding_batch_size
    was_training, vectors = module.training, []
    device = next(module.parameters()).device
    mps_rng = torch.mps.get_rng_state() if device.type == "mps" else None
    cuda_rng = torch.random.fork_rng(devices=[device.index or 0]) if device.type == "cuda" else nullcontext()
    try:
        with _preview_seed(result.config.seed), cuda_rng, torch.inference_mode():
            module.eval()
            for start in range(0, len(index), size):
                images = torch.stack([preprocess(index[i].image_ref.read(), module.identity["preprocessing"])
                    for i in range(start, min(start+size, len(index)))])
                vectors.append(module(images.to(device)).float().cpu().numpy())
    finally:
        module.train(was_training)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)
    features = np.concatenate(vectors)
    # PCA also gives a well-defined display for constant/single-dimension encoders.
    if not np.isfinite(features).all():
        raise ValueError("Non-finite embeddings detected during epoch monitoring")
    if np.allclose(features, features[0]):
        coordinates = np.zeros((len(features), 2))
    elif method == "pca" or features.shape[1] < 2:
        coordinates = PCA(n_components=min(2, features.shape[1]), random_state=result.config.seed).fit_transform(features)
        coordinates = np.pad(coordinates, ((0,0),(0,2-coordinates.shape[1])))
    else:
        coordinates = TSNE(n_components=2, perplexity=min(30., (len(index)-1)/3),
            init="pca", learning_rate="auto", random_state=result.config.seed).fit_transform(features)
    table = index.frame[["identity_id", "clip_id", "tracklet_id", "split"]].astype(object).copy()
    table["source"] = [index[i].image_ref.key for i in range(len(index))]
    table["x"], table["y"] = coordinates.T
    folder = result.output_dir / "monitoring"
    path = folder / f"embeddings-{epoch:04d}.png"
    save_chart(embedding_chart(table, identities=sorted(index.names.values()), method=method.upper(), epoch=epoch), path)
    table.to_csv(folder / "embeddings.csv", index=False)
    result.auxiliary_files.update({"monitoring/embeddings.png": path, "monitoring/embeddings.csv": folder / "embeddings.csv"})
    _save_monitor_cohort(result)
    metrics = {"embeddings/samples": len(index), "embeddings/skipped": 0}
    train = (table.split == "train").to_numpy()
    val = (table.split == "val").to_numpy()
    labels = table.identity_id.to_numpy()
    if train.any() and val.any():
        enrolled = sorted(set(labels[train]))
        similarities = features[val] @ features[train].T
        scores = np.stack([similarities[:, labels[train] == label].max(axis=1) for label in enrolled], axis=1)
        truth = labels[val]
        correct = [label in enrolled and np.sum(row >= row[enrolled.index(label)]) == 1 for row, label in zip(scores, truth)]
        metrics["val/monitor_rank1"] = float(np.mean(correct))
    if result.wandb_run is not None:
        import wandb
        result.wandb_run.log({"epoch": epoch, f"embeddings/{method}": wandb.Image(str(path), caption=f"Epoch {epoch+1}: fixed cohort")})
    return metrics
