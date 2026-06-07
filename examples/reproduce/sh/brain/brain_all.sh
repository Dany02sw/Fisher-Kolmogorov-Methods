#!/bin/bash
#PBS -N brain_all
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=1:mem=8gb

cd "$PBS_O_WORKDIR" || exit 1

SOLVER="spldg_bdf_red2"
SCHEME="bdf2"
L=2
T=50.0
DT=0.25
TOL=1e-6
MAX_IT=500
ETA0=1.0
THETA=-1.0

for SECTION in sagittal coronal horizontal; do
    fk-brain \
        --solver  "$SOLVER"  \
        --section "$SECTION" \
        --scheme  "$SCHEME"  \
        --l       "$L"       \
        --T       "$T"       \
        --dt      "$DT"      \
        --tol     "$TOL"     \
        --max-it  "$MAX_IT"  \
        --eta0    "$ETA0"    \
        --theta   "$THETA"
done
