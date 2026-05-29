# Official stable FEniCS (legacy) image
FROM quay.io/fenics/stable:current

# Install dependencies using ROOT user
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gmsh \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Back to default FEniCS user
USER fenics
WORKDIR /home/fenics/shared

# Install python dependencies
RUN pip install --no-cache-dir \
    fast-simplification==0.1.7 \
    h5py==3.11.0 \
    meshio==5.3.5 \
    trimesh==4.5.3