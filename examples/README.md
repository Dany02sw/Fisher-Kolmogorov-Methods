# Examples

This folder contains two types of entry points for running the Fisher-Kolmogorov solvers.

| Folder | Purpose |
|--------|---------|
| [`main/`](main/) | One script per solver — quick programmatic launcher, easy to customise |
| [`reproduce/`](reproduce/) | Fixed configurations used to produce the published paper results |

Both folders rely on the shared [`_run.py`](_run.py) launcher.
The package must be installed first (`pip install -e .` from the repo root).

---

## `main/` — interactive use

Each script in `main/` targets a single solver. Open the file, set the three
configuration variables at the top, and run it:

```python
_CONV_TYPE     = ConvType.SPATIAL   # SPATIAL | POLYNOMIAL | TEMPORAL
_TEST_TYPE     = TestType.COSINE    # COSINE  | WAVE
_SOLVER_KWARGS = {}
```

```bash
python3 examples/main/ldg_bdf.py
```

Model parameters (penalty coefficients, stabilisation constants, …) are set via
the `_PARAMS_CONFIG` dict — one entry per `TestType` so you can keep separate
tunings for each test case without editing the launch call.

---

## `reproduce/` — paper results

See [`reproduce/README.md`](reproduce/README.md) for full instructions.
To run everything in one shot:

```bash
python3 examples/reproduce/run_all_reproduce.py
```

Results are written to `reproduce/runs/<study>/<study>.txt`.
