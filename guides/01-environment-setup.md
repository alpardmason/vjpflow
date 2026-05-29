# 01 - Environment Setup

We use [`uv`](https://docs.astral.sh/uv/) for everything: Python version,
virtual environment, and dependencies. Python is pinned to **3.12**.

## Why uv

`uv` is a single fast tool that replaces `pyenv` + `venv` + `pip` + `pip-tools`.
It reads `pyproject.toml`, resolves a lockfile, and creates a reproducible
environment in seconds. No `requirements.txt`, no `setup.py`.

## Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv
```

## Create the environment

```bash
cd autograds
uv venv --python 3.12      # creates .venv with CPython 3.12
uv sync                    # installs the core engine (numpy only)
```

`uv sync` installs the project in editable mode plus the `dev` dependency group
(pytest, ruff, pyright). Run anything inside the environment with `uv run`:

```bash
uv run python -c "import autograd; print('ok')"
uv run pytest
uv run ruff check .
uv run pyright
```

## Optional: the Metal (Apple GPU) backend

The GPU backend needs Apple's Metal framework via PyObjC. It is an *optional
extra* so the core engine installs on any OS:

```bash
uv sync --extra metal      # macOS / Apple Silicon only
```

Check it actually found a GPU:

```bash
uv run python -c "from autograd.backends.metal_backend import MetalBackend; print(MetalBackend.is_available())"
```

- `True`  -> the Metal parity tests in `tests/test_backends.py` will run.
- `False` -> those tests are skipped automatically; everything else still works
  on numpy. This is expected on Linux/Windows or if PyObjC is not installed.

You do **not** need Xcode to *run* Metal from Python (PyObjC ships the runtime
bindings). You only need the full Xcode toolchain if you want to compile
`.metal` files ahead of time; we compile shaders at runtime from source, so the
Command Line Tools (`xcode-select --install`) are sufficient. See
[06](06-intro-to-metal.md).

## The three CI gates

Per house style, treat these as three separate, must-pass steps:

```bash
uv run ruff check .     # lint + import sorting
uv run pyright          # static types
uv run pytest           # gradient checks, parity, GPT-2 smoke
```

A green run looks like:

```text
All checks passed!                 # ruff
0 errors, 0 warnings               # pyright
33 passed, 5 skipped               # pytest (5 = Metal tests, if no GPU)
```

> [!note]
> If `from tests.util import ...` fails, it is because the project root must be
> on `sys.path`. That is configured in `pyproject.toml` under
> `[tool.pytest.ini_options] pythonpath = ["."]` -- run tests via `uv run
> pytest` from the repo root.
