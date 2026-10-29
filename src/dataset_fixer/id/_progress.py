"""Bounded ID work with the same text tqdm renderer as shared inference."""
import time
import sys

from tqdm import tqdm


def batches(total, size, *, stage, description, progress=True, on_progress=None):
    """Yield bounded ranges; report completed work, including verified cache hits."""
    started = last_report = time.monotonic()

    def report(completed):
        if progress:
            # Persistent lines reach W&B console capture as well as notebooks;
            # carriage-return progress bars alone need not survive capture.
            tqdm.write(f"{description}: {completed:,}/{total:,} images "
                       f"({time.monotonic() - started:.1f}s)")
            sys.stdout.flush()
        if on_progress is not None:
            on_progress(stage, completed, total)

    with tqdm(total=total, desc=description, unit="image", disable=not progress, mininterval=1) as bar:
        report(0)
        for start in range(0, total, size):
            end = min(start + size, total)
            yield start, end
            bar.update(end - start)
            now = time.monotonic()
            if end == total or now - last_report >= 10:
                report(end)
                last_report = now
