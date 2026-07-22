# Public MIT-licensed example dataset

The tutorials use a small, unmodified subset of
[SAWIT](https://github.com/dtnguyen0304/sawit), a public real-world camera-trap
object-detection dataset whose repository declares the MIT License. Images and
official YOLO annotations are downloaded directly from a pinned upstream commit.

```shell
python examples/download_public_examples.py /tmp/dataset-fixer-examples
```

The downloader creates:

- `sawit_unsplit`: official images and labels in one source split for controlled
  re-splitting;
- `sawit_fixed`: the same official files in a deterministic train/validation
  layout for tiling and model comparison.

Both use the canonical split-first structure (`train/images`, `train/labels`,
`val/images`, and `val/labels`).

`SOURCE.json` records the upstream URL and commit, selection rule, license, and
SHA-256 of every downloaded image and label. `UPSTREAM_LICENSE` is the license
file fetched from that same pinned commit. The tutorials do not generate or
alter pixels or annotations.
