# Source this file to activate the fenics2019 conda environment and move to
# the submission directory. Do not execute directly.
source /work/u10795357/miniconda3/etc/profile.d/conda.sh
conda activate fenics2019
cd "$PBS_O_WORKDIR" || exit 1