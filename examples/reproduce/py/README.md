# reproduce/py/

Python entry points to reproduce the numerical results of the paper.

## Running all studies

```bash
python examples/reproduce/run_all_reproduce.py
```

## Running individual studies

| Script | Study | Loop |
|--------|-------|------|
| `cosine/cosine_spatial_all.py` | Cosine · spatial | l = 1 … 8 |
| `cosine/cosine_polynomial.py` | Cosine · polynomial | — |
| `cosine/cosine_temporal_all.py` | Cosine · temporal | BDF1 … BDF6 |
| `wave/wave_spatial_saturation_all.py` | Wave · spatial | l = 2, 3 × BDF1 … BDF6 |
| `wave/wave_polynomial_saturation_all.py` | Wave · polynomial | BDF1 … BDF6 |
| `brain/brain_all.py` | Brain simulation | sagittal, coronal, horizontal |

```bash
python examples/reproduce/py/cosine/cosine_spatial_all.py
python examples/reproduce/py/brain/brain_all.py
```

## Checkpoints and resuming

Each convergence `*_all.py` script tracks completed steps via checkpoint
files under `reproduce/runs/<study>/.done/`. If a run is interrupted,
relaunching the same script skips already-completed steps automatically.
On successful completion, checkpoint files are deleted and only the result
log is kept.

The brain simulation has no checkpoint — it always runs all three sections
from scratch.

## Output

Convergence results are written to `reproduce/runs/<study>/<study>.txt`.
Brain simulation output (XDMF) is saved directly by the solver to the
paths configured in the mesh/output utilities.

## Resetting a convergence study

```bash
# Reset everything (checkpoints + log)
rm -rf Results/Reproduce/cosine_spatial/

# Reset only checkpoints, keep the log
rm -rf Results/Reproduce/cosine_spatial/.done/
```