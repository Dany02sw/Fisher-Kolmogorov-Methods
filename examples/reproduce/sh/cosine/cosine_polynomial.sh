#!/bin/bash
#PBS -N FK_cosine_polynomial
#PBS -q cpu
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe

source examples/reproduce/sh/_env.sh
source examples/reproduce/sh/_runner.sh

$RUNNER examples/reproduce/py/cosine/cosine_polynomial.py
