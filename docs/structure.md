# Project Structure

This repository keeps the final solution compact and reproducible.

```text
*.py
```

The core model and ensembling code lives in top-level Python scripts. The project was developed iteratively, so the final repository keeps only the scripts needed to explain and reproduce the selected submission.

```text
scripts/
```

Shell entrypoints for checking inputs and documenting the final run order.

```text
artifacts/
neural_artifacts/
```

Generated caches, validation predictions, checkpoints, and intermediate submissions. These directories are not tracked by git.

```text
SOLUTION.md
HISTORY.md
```

Human-readable description of the final method and experiment evolution.
