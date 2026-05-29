# AGENTS.md - VJPFlow (educational lazy autograd engine)

Operational knowledge base for this repo. Keep it current after significant
changes (see "Maintenance" at the end).

## Project tech stack and environment

- **Language:** Python 3.12 (pinned). Modern syntax throughout (`X | Y`,
  `list[X]`, `match` where useful, `pathlib`).
- **Toolchain:** `uv` for venv + deps. No `requirements.txt` / `setup.py`.
  - Create: `uv venv --python 3.12`
  - Sync core (numpy only): `uv sync`
  - Sync with GPU backend (macOS): `uv sync --extra metal`
  - Run: `uv run python ...`, `uv run pytest`, `uv run ruff check .`, `uv run pyright`
- **Runtime deps:** `numpy>=2.0`. Optional extra `metal`:
  `pyobjc-framework-Metal` (Apple Silicon only).
- **Dev tooling (uv `dev` group):** `pytest`, `ruff`, `pyright`.
- **Build backend:** hatchling, `src/` layout, package = `src/vjpflow`.

### The three CI gates (run all three)

```bash
uv run ruff check .   # lint + import order
uv run pyright        # static types (basic mode)
uv run pytest         # gradient checks, backend parity, GPT-2 smoke
```

Green baseline: ruff "All checks passed", pyright "0 errors", pytest
"33 passed, N skipped" (skips = Metal tests when no Apple GPU).

## Architecture in one paragraph

A **lazy** computation graph (`tensor.py` nodes, `graph.py` interpreter) plus
**functional** reverse-mode autodiff (`autodiff.py` `grad`/`value_and_grad`).
Primitives (`primitives.py`) each carry a `forward` (delegated to a **backend**)
and a `vjp` (local gradient rule that builds more graph). Backends
(`backends/`) are numpy (reference, full coverage) and raw Metal via PyObjC
(core kernels + numpy fallback). Layers (`nn/functional.py`) and the GPT-2 model
(`models/gpt2.py`) are pure compositions of primitives -- no hand-written
backward. `jit.py` rewrites the graph (elementwise fusion). Guides in `guides/`
are the reading companion; math derivations live in the `notes/` Obsidian vault.

## Key technical decision records

### ADR-1: Lazy execution (MLX/JAX style), not eager (PyTorch)
- **Context:** Audience wants to see inside a framework, aimed at AI-infra/optimization.
- **Decision:** Ops build graph nodes; computation deferred until `eval()`.
- **Alternatives:** Eager (PyTorch) - simplest to debug; eager core + lazy guide.
- **Rationale:** Makes differentiation and JIT literal graph transforms; matches
  MLX. Cost: must force `eval()` to see values (mitigated by `numpy()`/`item()`).

### ADR-2: Functional autodiff (`grad(f)`), not OO `.backward()`
- **Context:** Same audience; wanted functional + JIT exposure.
- **Decision:** `grad`/`value_and_grad` transforms; no `.grad` on tensors.
- **Alternatives:** PyTorch-style `.backward()` mutation.
- **Rationale:** Pure, composable; higher-order grad falls out because the
  backward pass is itself a graph built from the same vjps. One AD mechanism to learn.

### ADR-3: VJP-only primitives (never materialize Jacobians)
- **Decision:** Each primitive defines `J^T g`, not `J`.
- **Rationale:** Softmax/CE Jacobians are `O(d^2)` (vocab-sized); the VJP is
  `O(d)` and closed-form. This is the core efficiency idea of reverse-mode AD.

### ADR-4: Raw Metal via PyObjC, core-subset coverage
- **Context:** User explicitly chose hand-written real Metal shaders, core subset.
- **Decision:** GPU kernels for matmul + elementwise + last-axis reductions
  (enough for a small MLP+softmax end-to-end); numpy fallback for index/shape
  ops; GPT-2 runs on numpy.
- **Alternatives:** Use `mlx.core` arrays as backend (rejected: less "from
  scratch"); full Metal coverage (rejected: large effort, hurts clarity).
- **Rationale:** Teaches the real GPU pipeline (device/library/pipeline/buffers/
  threadgroups) without drowning the autograd lesson. Kernels are naive on purpose.

### ADR-5: Single-output `slice`, not multi-output `split`
- **Decision:** Backend exposes `slice`; QKV split = three `slice`s; `concat`
  backward = slices.
- **Rationale:** Keeps every graph node single-output, simplifying the graph
  model and the backward sweep.

### ADR-6: float32 floats, preserved-dtype integer indices
- **Decision:** `from_numpy` normalizes floats to float32; leaves int arrays
  alone.
- **Rationale:** Matches PyTorch/MLX defaults; integer token ids/indices must
  stay integer for `gather`.

## Common errors and pitfalls

Each entry: symptom -> root cause -> fix -> prevention.

### P1: Gradient is wrong only when shapes broadcast
- **Symptom:** `check_grad` fails for a bias/scalar argument but passes for the
  matrix argument.
- **Root cause:** Forgot to reduce the gradient back over broadcasted axes.
- **Fix:** Route every binary op's vjp through `sum_to_shape(g, input.shape)`.
- **Prevention:** Always test the *broadcasting* argument
  (`test_grad_broadcasting`), not just the same-shape one.

### P2: Gradient halved/missing for a reused tensor
- **Symptom:** `df/dx` for `f = x*x + x` comes out as `2x` instead of `2x+1`.
- **Root cause:** Overwriting a cotangent instead of accumulating when a tensor
  has multiple consumers.
- **Fix:** In the backward sweep, `cotangents[id(inp)] = ig if new else add(old, ig)`.
- **Prevention:** `test_gradient_accumulation_on_shared_node`.

### P3: Embedding gradient wrong for repeated tokens
- **Symptom:** A token appearing twice gets the gradient of appearing once.
- **Root cause:** Used `out[idx] += updates` (buffered) instead of an
  accumulating scatter.
- **Fix:** `np.add.at(out, indices, updates)` (unbuffered) in `scatter_add`.
- **Prevention:** `test_gather_scatter_roundtrip` checks a repeated index sums.

### P4: `exp`/`log` overflow or NaN in softmax / cross-entropy
- **Symptom:** `inf`/`nan` for large logits.
- **Root cause:** Naive `exp(x)` overflows.
- **Fix:** Subtract `max` before `exp` (softmax); use log-sum-exp (`log_softmax`).
- **Prevention:** It is built into `nn/functional.softmax`/`log_softmax`; reuse them.

### P5: `RecursionError` on deep models
- **Symptom:** Crash building/evaluating a many-layer graph.
- **Root cause:** Recursive graph traversal hitting Python's recursion limit.
- **Fix:** Iterative (stack-based) DFS in `graph.topological_sort` and
  `autodiff._reverse_topo`.
- **Prevention:** Never reintroduce recursion in graph walks.

### P6: Memory grows every training step
- **Symptom:** RSS climbs across SGD iterations.
- **Root cause:** Updated params keep a graph reaching back through all prior
  steps.
- **Fix:** `detach()` params after each update (`gpt2.sgd_step`).
- **Prevention:** Treat optimizer outputs as fresh leaves.

### P7: Metal tests fail / Metal import errors on non-Mac
- **Symptom:** Import or device errors when PyObjC/GPU absent.
- **Root cause:** Metal is optional and platform-specific.
- **Fix:** `MetalBackend.is_available()` gates construction; parity tests
  `skipif`; `Metal` typed `Any` and imported lazily.
- **Prevention:** Never import `Metal` at package top level; keep the numpy path
  the default.

### P8: `from tests.util import ...` fails
- **Symptom:** `ModuleNotFoundError: tests`.
- **Root cause:** Project root not on `sys.path`.
- **Fix:** `pyproject.toml` -> `[tool.pytest.ini_options] pythonpath = ["."]`;
  `tests/__init__.py` present. Run from repo root via `uv run pytest`.

### P9: Metal `uint3` garbage dims in matmul
- **Symptom:** Wrong matmul results on GPU.
- **Root cause:** Metal `uint3` is 16-byte aligned; packing 3 uints under-fills.
- **Fix:** `struct.pack("IIII", M, K, N, 0)` (pad to four).
- **Prevention:** Match host struct layout to MSL alignment exactly.

## Conventions

- **Math/shape notation** follows `notes/`: row-vector convention `Y = X W`,
  `G_X = dL/dX` (denominator layout), `S`=seq, `d`=model dim, `H`=heads.
- Single-letter math names (`W`, `Q`, `K`, `V`, `S`) are allowed (ruff `E741`,
  `N812` ignored).
- Comments explain *why*, not *what*. Public functions are type-annotated.
- Every new layer needs a `check_grad` finite-difference test.

## Maintenance

After each milestone, update: this file (new ADRs / pitfalls), `README.md` if
the public API changes, and add a guide or exercise in `guides/` for any new
concept. Keep `notes/` (Obsidian, separate vault) as the math source of truth;
do not duplicate derivations into code comments -- link to the note instead.
