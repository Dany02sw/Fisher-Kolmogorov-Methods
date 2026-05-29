## Installation & Setup

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
docker build -t fisher_kolmogorov .
```

### 2 - Run the container interactively with a shared volume:
```bash
docker run -ti -v "$(pwd):/home/fenics/shared" fisher_kolmogorov
```
Note: The -v flag mounts your current directory inside the container in real-time. You can edit the code on your host machine using your favorite IDE, and the container will see the updates instantly. You do not need to run pip install -e . inside Docker.

# ---

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