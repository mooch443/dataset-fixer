from __future__ import annotations

from tqdm import tqdm as text_tqdm

from dataset_fixer.comparison import inference


def test_inference_progress_uses_text_renderer() -> None:
    assert inference.tqdm is text_tqdm
