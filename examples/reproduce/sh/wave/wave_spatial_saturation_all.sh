#!/bin/bash
#PBS -N FK_wave_spatial_sat
#PBS -q cpu
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe

source /work/u10795357/miniconda3/etc/profile.d/conda.sh
conda activate fenics2019
cd "$PBS_O_WORKDIR"

# Pass --mpi as first argument to launch with mpirun instead of plain python3.
#   qsub script.sh              -> python3 (serial, default)
#   qsub -v args="--mpi" script.sh  -> mpirun python3
if [ "${args:-}" = "--mpi" ]; then
    mpirun python3 reproduce/py/wave/wave_spatial_saturation_all.py
else
    python3 reproduce/py/wave/wave_spatial_saturation_all.py
fi
