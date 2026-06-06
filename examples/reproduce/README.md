# Reproducing the paper results

This folder contains entry points to reproduce the numerical results presented in:

> **[Author(s)], "[Title]", [Journal], [Year]. DOI: [doi]**

Two equivalent interfaces are provided — use whichever fits your workflow:

| Interface | Location | How to run |
|-----------|----------|------------|
| Python scripts | `py/` | `python examples/reproduce/py/<script>.py` |
| Bash scripts | `sh/` | `bash examples/reproduce/sh/<script>.sh` |

The Python scripts configure the run programmatically and call the same
runners used internally by the package. The bash scripts call the CLI
with the exact arguments used for the published results.
Both produce identical output.

---

## Hardware and expected runtimes

Results were produced on: **[machine description, e.g. Intel Core i7-12700, 32 GB RAM]**

| Study | Estimated runtime |
|-------|-------------------|
| `spatial_all` | ~ X h |
| `temporal_all` | ~ X h |
| `polynomial` | ~ X h |
| `spatial_saturation` | ~ X h |
| `polynomial_saturation` | ~ X h |

> These are wall-clock estimates for the reference hardware above.
> Runtime scales roughly as O(N³) in the mesh size and linearly in the
> number of time steps, so finer studies will take proportionally longer.

---

## Running a single study

```bash
# Python interface
python examples/reproduce/py/spatial_all.py

# Bash interface
bash examples/reproduce/sh/spatial_all.sh
```

## Running all studies sequentially

```bash
for f in examples/reproduce/sh/*.sh; do bash "$f"; done
```

> **Warning:** running all studies sequentially may take several days
> on the reference hardware. Consider running individual studies in
> separate sessions.
