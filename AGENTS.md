# Repository agent notes

- Work directly on `main` and push framework commits to `origin/main` when the user asks. Do not stage notebook files unless the user explicitly includes them.
- GitHub credentials may be unavailable only because a command is sandboxed. If `gh auth status`, `git push`, or another GitHub operation reports missing or invalid credentials in the sandbox, retry that operation with privilege escalation before asking the user to reauthenticate.
- Never edit a notebook (`.ipynb`) the user is actively working on, including to revert an earlier change. Instead, describe the exact cell and replacement content for the user to apply. Modify such a notebook only when the user explicitly authorizes the exact edit.
- Reuse the retained Python 3.12 test environment at `/tmp/dataset-fixer-implementation-tests`. Refresh it with `/tmp/dataset-fixer-implementation-tests/bin/python -m pip install -e '.[test]'`; recreate it with `/Users/tristan/miniforge3/bin/python -m venv /tmp/dataset-fixer-implementation-tests` if it is missing.
- Never install test packages into the user's Miniforge environment or an active notebook kernel. Install only into the retained `/tmp` environment.
