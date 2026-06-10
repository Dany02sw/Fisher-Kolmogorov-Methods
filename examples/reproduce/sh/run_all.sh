#!/bin/bash
#PBS -N FK_run_all
#PBS -q cpu
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe

source examples/reproduce/sh/_env.sh
source examples/reproduce/sh/_runner.sh

STUDIES=(
    "examples/reproduce/py/cosine/cosine_spatial_all.py"
    "examples/reproduce/py/cosine/cosine_polynomial.py"
    "examples/reproduce/py/cosine/cosine_temporal_all.py"
    "examples/reproduce/py/wave/wave_spatial_saturation_all.py"
    "examples/reproduce/py/wave/wave_polynomial_saturation_all.py"
)

for study in "${STUDIES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  $study"
    echo "============================================================"
    $RUNNER "$study"
done