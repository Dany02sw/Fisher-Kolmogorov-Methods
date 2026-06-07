# Reproducing the paper results

This folder contains entry points to reproduce the numerical results presented in:

> **[Author(s)], "[Title]", [Journal], [Year]. DOI: [doi]**

Two equivalent interfaces are provided — use whichever fits your workflow:

| Interface | Location | How to run |
|-----------|----------|------------|
| Python scripts | `py/` | `python examples/reproduce/py/<script>.py` |
| Bash scripts (HPC) | `sh/` | `qsub examples/reproduce/sh/<script>.sh` |

---

## Hardware and expected runtimes

Results were produced on: **[machine description, e.g. Intel Core i7-12700, 32 GB RAM]**

| Study | Estimated runtime |
|-------|-------------------|
| Cosine · spatial | ~ X h |
| Cosine · polynomial | ~ X h |
| Cosine · temporal | ~ X h |
| Wave · spatial saturation | ~ X h |
| Wave · polynomial saturation | ~ X h |
| Brain simulation (3 sections) | ~ X h |

---

## Running a single study

```bash
python examples/reproduce/py/cosine/cosine_spatial_all.py
python examples/reproduce/py/brain/brain_all.py
```

## Running all studies sequentially

```bash
python examples/reproduce/run_all_reproduce.py
```

> **Warning:** running all studies sequentially may take several days
> on the reference hardware. Consider submitting individual jobs via the
> `sh/` scripts.

## Output layout

```
Results/
  Reproduce/
    cosine_spatial/
      .done/          ← checkpoint files (deleted on clean completion)
      cosine_spatial.txt
    cosine_temporal/
    ...
  Simulations/
    Waves/            ← XDMF from model comparison
    Brain/            ← XDMF from brain simulations
```

## Resetting a convergence study

```bash
# Reset everything (checkpoints + log)
rm -rf Results/Reproduce/cosine_spatial/

# Reset only checkpoints, keep the log
rm -rf Results/Reproduce/cosine_spatial/.done/
```
