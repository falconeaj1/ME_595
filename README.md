# ME_595

Control RL Project

## Python Environment

This project uses `uv` to manage Python, packages, and the local virtual environment.

Create the environment and install the project dependencies:

```bash
uv sync
```

Run Python or tools inside the environment:

```bash
uv run python
uv run jupyter lab
```

The base environment includes:

- `numpy`
- `scipy`
- `pysindy`
- `gymnasium[mujoco]` for environments such as `InvertedDoublePendulum-v5`
- `jupyterlab` and `ipykernel`
- common notebook/plotting helpers: `matplotlib` and `pandas`

## Managing Packages With uv

Add a package:

```bash
uv add package-name
```

Add a development-only package:

```bash
uv add --dev package-name
```

Remove a package:

```bash
uv remove package-name
```

Update the installed environment after editing `pyproject.toml`:

```bash
uv sync
```

Update the lockfile without installing packages:

```bash
uv lock
```

`uv` creates the local environment in `.venv/`, which is ignored by git. Commit both
`pyproject.toml` and `uv.lock` so collaborators get the same dependency versions.
