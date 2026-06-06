# Reproducing the paper results

This folder contains entry points to reproduce the numerical results presented in:

> **[Author(s)], "[Title]", [Journal], [Year]. DOI: [doi]** *TODO*

Two equivalent interfaces are provided — use whichever fits your workflow:

| Interface | Location | How to run |
|-----------|----------|------------|
| Python scripts | `py/` | `python3 examples/reproduce/py/<script>.py` |
| Bash scripts (HPC) | `sh/` | `qsub examples/reproduce/sh/<script>.sh` |

The Python scripts configure the run programmatically and call the same
runners used internally by the package. The bash scripts wrap them for
PBS job submission. Both produce identical output.

---

## Hardware and expected runtimes (*TODO*)

Results were produced on: **[machine description, e.g. Intel Core i7-12700, 16 GB RAM]**

| Study | Estimated runtime |
|-------|-------------------|
| Cosine · spatial | ~ X h |
| Cosine · polynomial | ~ X h |
| Cosine · temporal | ~ X h |
| Wave · spatial saturation | ~ X h |
| Wave · polynomial saturation | ~ X h |

---

## Running a single study

```bash
python3 examples/reproduce/py/cosine/cosine_spatial_all.py
python3 examples/reproduce/py/wave/wave_spatial_saturation_all.py
```

## Running all studies sequentially

```bash
python3 examples/reproduce/run_all_reproduce.py
```

> **Warning:** running all studies sequentially may take several days if the full model spdg_bdf is used
> on the reference hardware. Consider submitting individual jobs via the
> `sh/` scripts.

## Output layout

```
reproduce/runs/
  cosine_spatial/
    .done/          ← checkpoint files (deleted on clean completion)
    cosine_spatial.txt
  cosine_temporal/
    ...
  wave_spatial/
    ...
```

Each `.txt` log grows monotonically: completed steps are appended with a
timestamp header so partial results are never lost.

## Resetting a study

```bash
# Reset everything (checkpoints + log)
rm -rf examples/reproduce/runs/cosine_spatial/

# Reset only checkpoints, keep the log
rm -rf examples/reproduce/runs/cosine_spatial/.done/
```
