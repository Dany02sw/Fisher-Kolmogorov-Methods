# Installation & Setup

First, clone the repository and navigate into the project directory:

```bash
git clone https://github.com/Dany02sw/Fisher-Kolmogorov-Methods fisher-kolmogorov
cd fisher-kolmogorov
```


## Option 1: Native Linux / WSL2 (Via Conda)

If you are working on Linux or WSL2, you can recreate the exact development environment (with all mesh and I/O dependencies locked to their tested versions) using the provided `environment_linux.yml` file.

### 1 - Create and activate the environment:
```bash
conda env create -f environment_linux.yml
conda activate fenics-fk-env
```

### 2 - Install the project in editable mode:
```bash
pip install -e .
```
Note: This links the source code to your Conda environment, allowing you to modify the Python scripts and see the changes instantly.

## Option 2: Universal Compatibility (Via Docker)
If you are on macOS (including Apple Silicon M1/M2/M3) or Windows native, using Docker avoids any complex C++ compilation issues with FEniCS and GMSH by running everything inside an isolated Linux container.

### 1 - Build the Docker image (only needed the first time):
```bash
sudo docker build -t fisher_kolmogorov .
```

### 2 - Run the container interactively with a shared volume:
```bash
sudo docker run -ti --rm -v "$(pwd):/home/fenics/shared" fisher_kolmogorov /bin/bash
```
**Note on Visualization**: The Docker container is currently configured as a "headless" environment. It is designed specifically for reproducing numerical calculations and simulations. It is not configured to render or display plots directly (plotting routines will crash if they attempt to open an interactive window). You can use this environment to run your scripts, but for visualization, we recommend saving the output as files (.png) within the shared volume and viewing them on your host machine.

# Repository Structure

Here is a brief overview of the main directories and how the project is organized:

```
Fisher-Kolmogorov-Methods/
├── Meshes/          # Mesh generation, brain STL files, and conversion utilities
├── Models/          # Core solvers grouped by spatial-temporal discretization
├── Plots/           # Scripts to visualize simulation results and convergence charts
├── Utilities/       # Auxiliary routines and shared helper functions
└── config.py        # Global path configurations
```

### Core Components

* **`Models/`**: This is the heart of the repository, built upon a modular object-oriented design. It features a core base solver from which specialized BDF and $\theta$-method solvers inherit. From these, the 8 distinct numerical models are derived (combining the 4 spatial discretizations with the 2 temporal schemes). Each model has its own subfolder containing its specific solver implementation and the `mainTest<N>.py` executable scripts.
* **`Meshes/`**: Contains the complete pipeline for geometric domains, handling mesh generation, conversion, and visualization for all test cases. The directory components are all located at the root of this folder:
  * `ImportMeshes.py`: Contains the centralized `mesh_factory` function used to load and manage meshes across the repository.
  * `MeshPlotUtilities.py`: Provides the underlying grid-plotting functions, including `plot_mesh_grid`.
  * `RunPlotMesh.py`: The main executable script used to plot and visualize the computational meshes.
  * `PlotMeshConfig/`: Stores reference mesh configurations (specifically for spatial convergence studies and brain simulations) used during visualization.
  * `BrainMeshes/`: Handles the generation of 2D brain slices from raw `.stl` data, interactive slicing scripts, and tools to convert meshes into FEniCS-compatible `.xdmf` formats.
* **`Plots/`**: Houses the configuration and plotting scripts used to post-process simulation data, evaluate error norms, and format final figures. 
  * `ModelConfig/`: Contains template scripts for storing model-specific data (with `SP-LDG-BDF` model configuration currently fully populated).
  * `PlotUtilities.py` provides the underlying plotting functions executed by the main entry point `RunPlots.py`.
* **`Utilities/`**: Contains auxiliary scripts organized by task. It encapsulates specific classes, configuration dictionaries, and shared helper functions used throughout the codebase to ensure modularity and clean structure.
* **`config.py`**: Defines the global path and directory structures used across the repository to automatically standardize where output files are stored and organized.

---

# How to run

## 1 — Run a simulation

The models are organized in the `Models/` directory. Each subfolder corresponds to a specific spatial-temporal discretization scheme:

```
Models/
├── DG-BDF/         ├── DG-THETA/
├── LDG-BDF/        ├── LDG-THETA/
├── PP-DG-BDF/      ├── PP-DG-THETA/
├── SP-LDG-BDF/     └── SP-LDG-THETA/
```

Launch the test corresponding to the model of your choice by using the following commands(replace the model folder with the desired one):

```bash
python3 Models/DG-BDF/mainTest1.py     # Test 1: Cosine (convergence analysis)
python3 Models/DG-BDF/mainTest2.py     # Test 2: Waves (robustness on sharp fronts / saturation effect)
python3 Models/DG-BDF/mainTest3.py     # Test 3: Brain Application
```

Each `mainTest<N>.py` script executes the full simulation followed by a post-processing phase, which varies depending on the test case:

* **Tests 1 & 2 (`ConvergenceTest` method):** Computes convergence rates and generates the relative plots. Plot saving is disabled by default but can be enabled by setting the `save` flag to `True`.
* **Test 3 (`Solve` method):** Stores the solution data by exporting the corresponding `.xdmf` and `.h5` files.
  
> **Note:** Before running `mainTest3.py`, make sure the brain meshes are available. See step 3 below.

---

## 2 — Plot the meshes

To visualize the computational meshes used in all three tests, run the dedicated script from the project root:

```bash
python3 Meshes/RunPlotMesh.py
```

By default, the script plots the meshes used for the spatial convergence analysis in Tests 1 and 2, as well as the specific brain meshes used for the simulations illustrated in the report.

---

## 3 — Brain meshes for Test 3

Test 3 utilizes 2D brain section meshes. The repository **already includes** the pre-generated `.msh` files inside `Meshes/BrainMeshes/MshFiles/`. 

* **Standard Run:** No manual preparation is required. Launching `mainTest3.py` will automatically trigger the conversion from `.msh` to `.xdmf` if the latter files are missing.
* **Custom Mesh Generation:** If you wish to customize the cutting sections or regenerate the meshes from scratch, you can follow the pipeline described below.

### Step 3a — Generate the `.msh` files from the STL

Download the brain STL model from [this source](https://cults3d.com/en/3d-model/tool/brain-bygonza), rename it `brain.stl`, and place it at:

```
Meshes/BrainMeshes/StlFiles/brain.stl
```

Create the `StlFiles/` folder if it does not exist. Then run the mesh generator from the project root:

```bash
python3 Meshes/BrainMeshes/MeshGenerators/meshGenerator2D.py
```

The script is interactive: it will prompt you to select the anatomical section (sagittal, coronal, or horizontal) and the cutting offset. The output `.msh` file is saved in `Meshes/BrainMeshes/MshFiles/`. Repeat for each section you need.

### Step 3b — Convert the `.msh` files to `.xdmf`

FEniCS requires meshes in XDMF format. Run the converter from the project root:

```bash
python3 Meshes/BrainMeshes/Converter/Msh_to_xdmf.py
```

The script is interactive: it will list the available `.msh` files and ask you to select one. The converted `.xdmf` file is saved in `Meshes/BrainMeshes/MeshSrc/`. Repeat for each section you need. Once the XDMF files are in place, `mainTest3.py` can be executed normally.

---

# Discontinuous Galerkin - based methods for Fisher-Kolmogorov

This repository contains various Discontinuous Galerkin (DG) methods for solving the Fisher-Kolmogorov equation. We provide four different spatial discretizations for the diffusive term, each paired with two temporal schemes: a BDF (Backward Differentiation Formula) scheme for higher-order time accuracy, and a $\theta$-method (which includes the Crank-Nicolson scheme).

# Mathematical Model

The Fisher-Kolmogorov equation (also known as the Fisher-KPP equation) is a reaction-diffusion model described as:
$$
\frac{\partial c}{\partial t} - \nabla \cdot (D \nabla c) = \alpha c (1 - c) + f
$$
This repository focuses on the numerical solution of this equation using Discontinuous Galerkin (DG) methods.

# Space discretization

## DG Methods

### - Interior Penalty Discontinuous Galerkin (DG)
Based on [this reference](https://www.mate.polimi.it/biblioteca/add/qmox/10-2023.pdf) and implementing a parameter for choosing between different interior penalty strategies (namely SIP, NIP, IIP), this method provides a standard Discontinuous Galerkin discretization of the diffusion term. It is the computationally cheapest model since it involves a single variable and no transformations. Nevertheless, it is also the least robust one. This lack of robustness can be easily observed by running a convergence analysis with respect to the polynomial degree.

### - Positivity Preserving Discontinuous Galerkin (PP-DG)
Based on [this reference](https://www.mate.polimi.it/biblioteca/add/qmox/59-2023.pdf), this scheme employs the following exponential change of variable, 
$$
c = e^{\lambda}
$$
the method analytically guarantees positivity throughout the simulation. While highly effective at preserving the lower physical bound, the exponential formulation can occasionally lead to challenges in accurately capturing the upper saturation limits of the concentration.

**Note on Penalization:** 
In this implementation, a simplification has been adopted for the penalization term $\eta$, compared to the one provided in the paper. Specifically, we use the modulus of $\lambda$ instead of the $\lVert \lambda \rVert_{L^{\infty}(K^{*})}$ norm. Numerical evidence suggests that this choice does not significantly impact the results compared to an approximation using the $\lVert \lambda \rVert_{L^{p}(K^{*})}$ norm via a Lagrange multiplier approach. Alternative strategies could be explored, such as: 
* implementing a warm start for the polynomial degree $p$;
* using the penalization from the previous step, $\eta(\lambda^{k})$, for the current step $k+1$. Although this partially unaligns the $\theta$-method, it ensures the use of a known, exact quantity.

## LDG Methods 

### - Local Discontinuous Galerkin (LDG)
This implementation follows the standard LDG approach for spatial discretization as described in [this reference](https://www.dam.brown.edu/people/tqin/Files/ReadingGroup2016Summer/err_LDG_Poisson.pdf). It offers an optimal balance between computational efficiency and accuracy. Since this is not a structure-preserving method, it is best suited for applications where minor undershooting or overshooting is acceptable. It is worth noting that at low polynomial approximation degrees, even structure-preserving methods may occasionally exceed physical bounds due to numerical oscillations.

### - Structure Preserving Local Discontinuous Galerkin (SP-LDG)
Based on [this reference](https://www.mate.polimi.it/biblioteca/add/qmox/18-2025.pdf), this  method combines the rigorous mass conservation of the LDG framework with the following sigmoid-based change of variable
$$
c = u(w) = \frac{e^{w}}{1 + e^{w}}
$$
designed to preserve the entire physical range $[0, 1]$. By enforcing these bounds at the structural level, it provides the most robust and accurate results among the implemented schemes. Although it is the most computationally intensive option, it is the recommended choice for simulations requiring high precision and strict adherence to physical constraints.

**Note on reduced models:** 
Since the computational cost of this particular model is significantly higher than the previous ones, we have also implemented two reduced models:
* a two-equation model, eliminating the two linear variables at the continuous level;
* a one-equation model (**TODO**), which reduces everything to a single variable, fundamentally corresponding to a SIP-DG scheme with a structure-preserving change of variables.
We remark that a static condensation implementation would be more appropriate for this purpose, and we may address this specific implementation in the future.

# Time Discretization

## - Backward Differentiation Formula ($\text{BDF}\nu$)
The BDF time integration enables high-order temporal accuracy. This implementation is particularly suited for scenarios where the initial solution is smooth, as it does not currently include a self-starting procedure (**TODO**). However, by providing a well-defined time-dependent initial condition, the solvers can retrieve the starting BDF history from negative times (which can be interpreted as past measurements), thereby retaining the optimal $\text{BDF}\nu$ order of accuracy.

## - $\theta$-Method
The $\theta$-method provides a flexible temporal framework. Depending on the choice of $\theta$, it can recover the first-order Backward Euler scheme ($\theta=1$) or the second-order Crank-Nicolson method ($\theta=0.5$). Unlike the BDF approach, it requires only a single initial condition, making it more straightforward for general applications.

*Note: An exception occurs if a second-order explicit extrapolation is combined with the Crank-Nicolson scheme, which may alter the required historical steps.*

# Tests

## - Test 1: Cosine
The first test case employs a cosine-based analytical solution to evaluate the formal convergence properties of the models. Specifically, we perform a spatial convergence analysis with respect to both the mesh size ($h$) and the polynomial degree ($p$) separately, alongside a temporal convergence analysis to validate the accuracy of the BDF and $\theta$-method schemes.

## - Test 2: Waves 
This test is designed to assess the robustness of the spatial discretizations. It demonstrates the ability of the schemes to maintain stability and accuracy even when capturing sharp fronts and high gradients, which are characteristic of the Fisher-KPP equation. Moreover, we exploit this test to perform a saturation study of the time discretization when doing a spatial convergence analysis, as can be observed in the `Plot/ModelConfig/*` scripts. 

## - Test 3: Simple Brain Application
The final test case evaluates the performance of the models within complex geometries and under non-homogeneous diffusion and reaction coefficients. The simulation is conducted on three anatomical brain sections (sagittal, coronal, and horizontal) derived from the [following STL model](https://cults3d.com/en/3d-model/tool/brain-bygonza). This test highlights the methods' applicability to realistic biological scenarios involving heterogeneous media.
