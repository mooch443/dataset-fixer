# Colab tutorials

These notebooks are designed to run from top to bottom in Google Colab. Each
one installs the package from this public repository and generates deterministic
MIT-licensed example data locally in the Colab session.

| Notebook | Main API | Expected CPU runtime |
| --- | --- | --- |
| [01 — Controlled splitting](01_controlled_splitting.ipynb) | `Dataset.split()` | 2–4 minutes |
| [02 — Task-aware tiling](02_task_aware_tiling.ipynb) | `Dataset.tile()` | 3–6 minutes |
| [03 — Fixed-cohort model comparison](03_fixed_cohort_model_comparison.ipynb) | `Dataset.compare_models()` | 10–25 minutes |

The comparison tutorial trains two tiny detection runs. A GPU runtime is
recommended for that notebook, although it also runs on CPU with longer
training time.

All generated example images and annotations are covered by the repository's
[MIT License](../LICENSE). The notebooks contain the same AI-generation
disclosure as the main README.
