#!/bin/bash
#PBS -N FK_run_all
#PBS -q cpu
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe

source /work/u10795357/miniconda3/etc/profile.d/conda.sh
conda activate fenics2019
cd "$PBS_O_WORKDIR"

STUDIES=(
    "reproduce/py/cosine/cosine_spatial_all.py"
    "reproduce/py/cosine/cosine_polynomial.py"
    "reproduce/py/cosine/cosine_temporal_all.py"
    "reproduce/py/wave/wave_spatial_saturation_all.py"
    "reproduce/py/wave/wave_polynomial_saturation_all.py"
)

if [ "${args:-}" = "--mpi" ]; then
    RUNNER="mpirun python3"
else
    RUNNER="python3"
fi

for study in "${STUDIES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  $study"
    echo "============================================================"
    $RUNNER "$study"
done
