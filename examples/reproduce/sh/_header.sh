# Common PBS header — sourced by all reproduce job scripts.
# Do not submit this file directly.
#
# Usage in each script:
#   #PBS -N <job-name>
#   #PBS -q cpu
#   #PBS -l select=1:ncpus=4:mpiprocs=4:mem=32gb
#   #PBS -l walltime=24:00:00
#   #PBS -j oe
