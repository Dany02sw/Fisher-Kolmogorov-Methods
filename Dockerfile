# Use Ubuntu 18.04
FROM ubuntu:18.04

# Avoid interactions during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install FEniCS and base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:fenics-packages/fenics \
    && apt-get update && apt-get install -y \
    fenics \
    gmsh \
    python3-pip \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Update pip for Python 3.6 (default for Ubuntu 18.04)
RUN python3 -m pip install --upgrade pip setuptools

# Create FEniCS user
RUN useradd -m -s /bin/bash fenics
USER fenics
WORKDIR /home/fenics/shared
ENV PYTHONPATH=/home/fenics/shared

# Disable MPI checks
ENV OMPI_MCA_plm=isolated
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

# Install pip dependencies compatible with Python 3.6
RUN pip3 install --no-cache-dir --user \
    fast-simplification==0.1.7 \
    meshio==4.4.6 \
    trimesh==3.9.35