# Resolve the Python runner based on the PBS variable 'args'.
# Source this file after _env.sh. Do not execute directly.
#
# Usage:
#   source reproduce/sh/_runner.sh        -> sets RUNNER="python3"
#   qsub -v args="--mpi" script.sh        -> sets RUNNER="mpirun python3"
if [ "${args:-}" = "--mpi" ]; then
    RUNNER="mpirun python3"
else
    RUNNER="python3"
fi