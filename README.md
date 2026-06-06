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

```
fisher-kolmogorov/
├── config.py                            # Output directory paths
├── pyproject.toml                       # Package metadata and CLI entry points
│
├── fisher_kolmogorov/                   # Main package
│   ├── models/                          # Solver implementations
│   │   ├── solver_base.py               # Abstract base class
│   │   ├── solver_bdf.py                # BDF time integrator (orders 1–6)
│   │   ├── solver_theta.py              # θ-method time integrator
│   │   ├── solver_rk.py                 # Explicit RK (BDF startup + stand-alone)
│   │   ├── solver_dg_bdf.py             # Interior penalty DG — BDF
│   │   ├── solver_dg_theta.py           # Interior penalty DG — θ
│   │   ├── solver_ldg_bdf.py            # LDG — BDF
│   │   ├── solver_ldg_theta.py          # LDG — θ
│   │   ├── solver_spldg_bdf.py          # Structure-preserving LDG — BDF
|   |   ├── solver_spldg_bdf_red2.py     # Structure-preserving LDG — BDF (reduced to 2 equations)
│   │   ├── solver_spldg_theta.py        # Structure-preserving LDG — θ
|   |   ├── solver_spldg_theta_red2.py   # Structure-preserving LDG — θ   (reduced to 2 equations)
│   │   ├── solver_ppdg_bdf.py           # Positivity-preserving DG — BDF
│   │   └── solver_ppdg_theta.py         # Positivity-preserving DG — θ
│   │
│   ├── runners/                         # Convergence and simulation orchestration
│   │   ├── spatial_runner.py
│   │   ├── polynomial_runner.py
│   │   ├── temporal_runner.py
│   │   └── brain_runner.py
│   │
│   ├── cli/                             # CLI entry points (installed as system commands)
│   │   ├── main_convergence_cli.py      # → fk-convergence
│   │   └── main_brain_cli.py            # → fk-brain
│   │
│   ├── configs/
│   │   ├── model_configs/               # Solver parameter dataclasses
│   │   │   ├── base.py                  # ModelParams base dataclass
│   │   │   ├── dg_configs.py            # DgParams
│   │   │   └── ldg_configs.py           # LdgParams, SpLdgParams, PpDgParams
│   │   └── test_configs/                # Convergence test specifications
│   │       ├── base.py                  # TestConfig, ConvergenceParams, BrainConfig
│   │       ├── cosine.py                # Cosine test factories
│   │       ├── waves.py                 # Wave (tanh) test factories
│   │       └── brain.py                 # Brain simulation factory
│   │
│   ├── meshes/                          # Mesh generation and loading utilities
│   │
│   ├── plots/                           # Plot utilities and per-model convergence data
│   │   ├── _primitives.py               # add_ref_line, add_slope_triangle, save_plot
│   │   ├── _axes.py                     # finalize_ax, finalize_ax_pair
│   │   ├── _curves.py                   # Curve-drawing helpers per study type
│   │   ├── plot_utilities.py            # Public API — one function per plot type
│   │   └── models_config/               # Per-model numerical results for plotting
│   │       ├── __init__.py              # CONFIG_REGISTRY + load_config
│   │       ├── ldg_bdf.py
│   │       └── ...
│   │
│   └── utilities/                       # Shared utilities
│       ├── enum_utilities.py            # All enums (SpaceMethod, TimeMethod, ...)
│       ├── fenics_utilities.py
│       ├── io_utilities.py
│       ├── math_utilities.py
│       ├── transform_utilities.py       # Identity, Sigmoid, Exponential transforms
│       ├── explicit_extrapolations.py   # Extrapolation table for linearized solvers
│       └── dictionary_utilities.py      # Label maps for plots
│
├── reproduce/                           # Reproducible paper results
│   ├── README.md                        # Paper reference, hardware, expected runtimes
│   └── sh/                              # Bash scripts — one per study type
│       ├── spatial_all.sh
│       ├── temporal_all.sh
│       ├── polynomial.sh
│       ├── spatial_saturation.sh
│       └── polynomial_saturation.sh
│
├── local/                               # Personal quick-launch scripts (git-ignored)
└── tests/                               # Smoke and convergence tests
```

---

## Usage

After installation, two commands are available system-wide:

### Convergence studies — `fk-convergence`

```bash
fk-convergence --solver ldg_bdf --test cosine --conv spatial --l 2
fk-convergence --solver spldg_bdf --test wave  --conv temporal \
    --tol 1e-11 --max-it 200
# Linearized reaction term (dg_bdf / ldg_bdf only):
fk-convergence --solver dg_bdf --test cosine --conv temporal --linearize
```

Available solvers: `dg_bdf`, `dg_theta`, `ldg_bdf`, `ldg_theta`, `ppdg_bdf`, `ppdg_theta`, `spldg_bdf`, `spldg_bdf_red2`, `spldg_theta`, `spldg_theta_red2`.

Run `fk-convergence --help` for the full option list.

### Brain simulation — `fk-brain`

```bash
fk-brain --solver ldg_theta --section sagittal
fk-brain --solver dg_bdf --section coronal \
    --l 2 --T 50.0 --dt 0.25 --scheme bdf2 --eta0 10.0 --linearize
```

Run `fk-brain --help` for the full option list.

### Reproducing paper results

See `reproduce/README.md` for the paper reference, hardware details, and expected runtimes. Each script loops over the relevant solvers and parameters:

```bash
bash reproduce/sh/spatial_all.sh
bash reproduce/sh/temporal_all.sh
```

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
This test is designed to assess the robustness of the spatial discretizations. It demonstrates the ability of the schemes to maintain stability and accuracy even when capturing sharp fronts and high gradients, which are characteristic of the Fisher-KPP equation. Moreover, we exploit this test to perform a saturation study of the time discretization when doing a spatial convergence analysis, as can be observed in the `fisher_kolmogorov/plots/model_config/*` scripts. 

## - Test 3: Simple Brain Application
The final test case evaluates the performance of the models within complex geometries and under non-homogeneous diffusion and reaction coefficients. The simulation is conducted on three anatomical brain sections (sagittal, coronal, and horizontal) derived from the [following STL model](https://cults3d.com/en/3d-model/tool/brain-bygonza). This test highlights the methods' applicability to realistic biological scenarios involving heterogeneous media.