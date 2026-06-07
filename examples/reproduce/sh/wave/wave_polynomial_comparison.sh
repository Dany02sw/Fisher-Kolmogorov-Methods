#!/bin/bash
#PBS -N FK_wave_poly_comparison
#PBS -q cpu
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe

source /work/u10795357/miniconda3/etc/profile.d/conda.sh
conda activate fenics2019
cd "$PBS_O_WORKDIR" || exit 1

WAVE_DIR="results/wave"

# Pass --mpi as first argument to launch with mpirun instead of plain python3.
#   qsub script.sh                  -> fk-convergence (serial, default)
#   qsub -v args="--mpi" script.sh  -> mpirun fk-convergence
if [ "${args:-}" = "--mpi" ]; then
    RUNNER="mpirun fk-convergence"
else
    RUNNER="fk-convergence"
fi

for SOLVER in dg_bdf ldg_bdf ppdg_bdf spldg_bdf_red2; do
    $RUNNER \
        --solver     "$SOLVER"          \
        --test       wave               \
        --conv       polynomial         \
        --l-list     1 2                \
        --scheme     bdf6               \
        --output-dir "$WAVE_DIR/$SOLVER"
done
