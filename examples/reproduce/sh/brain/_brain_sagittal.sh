#!/bin/bash
#PBS -N brain_sagittal
#PBS -l walltime=03:00:00
#PBS -l select=1:ncpus=1:mem=8gb

cd "$PBS_O_WORKDIR" || exit 1

fk-brain \
    --solver  spldg_bdf_red2 \
    --section sagittal        \
    --scheme  bdf2           \
    --l       2              \
    --T       50.0           \
    --dt      0.25           \
    --tol     1e-6           \
    --max-it  500            \
    --eta0    1.0            \
    --theta   -1.0
