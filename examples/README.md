# MIT-licensed example datasets

`create_example_datasets.py` deterministically generates the small datasets
used by the Colab tutorials. The generated images and annotations are original
synthetic assets and are released under the repository's MIT License.

```shell
python examples/create_example_datasets.py /tmp/dataset-fixer-examples
```

The command creates:

- `raw_sequences`: grouped YOLO detection images for demonstrating physical
  group-aware splitting;
- `detection`: fixed train/validation detection data for training and model
  comparison;
- `polo`: point-and-radius annotations for randomized coverage tiling.

The generator uses a fixed seed by default, has no network dependency, and may
be rerun to reproduce the exact same annotations and image contents.
