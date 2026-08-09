# Repository agent notes

- Work directly on `main` and push framework commits to `origin/main` when the user asks. Do not stage notebook files unless the user explicitly includes them.
- Reuse the retained Python 3.12 test environment at `/tmp/dataset-fixer-implementation-tests`. Refresh it with `/tmp/dataset-fixer-implementation-tests/bin/python -m pip install -e '.[test]'`; recreate it with `/Users/tristan/miniforge3/bin/python -m venv /tmp/dataset-fixer-implementation-tests` if it is missing.
- Never install test packages into the user's Miniforge environment or an active notebook kernel. Install only into the retained `/tmp` environment.
