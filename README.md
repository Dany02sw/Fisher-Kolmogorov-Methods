# Discontinuous Galerkin methods for Fisher-Kolmogorov
This repository contains various Discontinuous Galerkin (DG) methods for solving the Fisher-Kolmogorov equation. We provide three different spatial discretizations for the diffusive term, each paired with two temporal schemes: a BDF (Backward Differentiation Formula) scheme for higher-order time accuracy, and a $\theta$-method (which includes the Crank-Nicolson scheme).

# Mathematical Model
The Fisher-Kolmogorov equation (also known as the Fisher-KPP equation) is a reaction-diffusion model described as:
$$
\frac{\partial c}{\partial t} - \nabla \cdot (D \nabla c) = \alpha c (1 - c) + f
$$
represents the nonlinear reaction term. This repository focuses on the numerical solution of this equation using Discontinuous Galerkin (DG) methods.

# Space discretization
## - Local Discontinuous Galerkin (LDG)
This implementation follows the standard LDG approach for spatial discretization as described in This implementation follows the standard **LDG approach** as described in [this reference](https://www.dam.brown.edu/people/tqin/Files/ReadingGroup2016Summer/err_LDG_Poisson.pdf). It offers an optimal balance between computational efficiency and accuracy. Since this is not a structure-preserving method, it is best suited for applications where minor undershooting or overshooting is acceptable. It is worth noting that at low polynomial approximation degrees, even structure-preserving methods may occasionally exceed physical bounds due to numerical oscillations.

## - Positivity Preserving Discontinuous Galerkin (PP-DG)
Based on [this reference](https://www.mate.polimi.it/biblioteca/add/qmox/59-2023.pdf), this scheme minimizes computational overhead by solving only a single equation. By employing the following exponential change of variable, 
$$
c = e^{\lambda}
$$
the method analytically guarantees positivity throughout the simulation. While highly effective at preserving the lower physical bound, the exponential formulation can occasionally lead to challenges in accurately capturing the upper saturation limits of the concentration.

**Note on Penalization:** 
In this implementation, a simplification has been adopted for the penalization term $\eta$. Specifically, we use the modulus of $\lambda$ instead of the $\lVert \lambda \rVert_{L^{\infty}(K^{*})}$ norm. Numerical evidence suggests that this choice does not significantly impact the results compared to an approximation using the $\lVert \lambda \rVert_{L^{p}(K^{*})}$ norm via a Lagrange multiplier approach. Alternative strategies could be explored, such as: 
* implementing a warm start for the polynomial degree $p$;
* using the penalization from the previous step, $\eta(\lambda^{k})$, for the current step $k+1$. Although this partially unaligns the $\theta$-method, it ensures the use of a known, exact quantity.

## - Structure Preserving Local Discontinuous Galerkin (SP-LDG)
Based on [this reference](https://www.mate.polimi.it/biblioteca/add/qmox/18-2025.pdf), this  method combines the rigorous mass conservation of the LDG framework with the following sigmoid-based change of variable
$$
c = u(w) = \frac{e^{w}}{1 + e^{w}}
$$
designed to preserve the entire physical range $[0, 1]$. By enforcing these bounds at the structural level, it provides the most robust and accurate results among the implemented schemes. Although it is the most computationally intensive option, it is the recommended choice for simulations requiring high precision and strict adherence to physical constraints.

# Time Discretization
## - Backward Differentiation Formula (BDF)
The BDF time integration enables high-order temporal accuracy. This implementation is particularly suited for scenarios where the initial diffusion is slow or smooth, as it does not currently include a self-starting procedure.

## - $\theta$-Method
The $\theta$-method provides a flexible temporal framework. Depending on the choice of $\theta$, it can recover the first-order Backward Euler scheme ($\theta=1$) or the second-order Crank-Nicolson method ($\theta=0.5$). Unlike the BDF approach, it requires only a single initial condition, making it more straightforward for general applications.

# Tests
## - Test 1: cosine
The first test case employs a cosine-based analytical solution to evaluate the formal convergence properties of the models. Specifically, we perform a spatial convergence analysis with respect to both(separately) the mesh size ($h$) and the polynomial degree ($p$), alongside a temporal convergence analysis to validate the accuracy of the BDF and $\theta$-method schemes.

## - Test 2: waves 
This test is designed to assess the robustness of the spatial discretizations. It demonstrates the ability of the schemes to maintain stability and accuracy even when capturing sharp fronts and high gradients, which are characteristic of the Fisher-KPP equation.

## - Test 3: simple brain application
The final test case evaluates the performance of the models within complex geometries and under non-homogeneous diffusion and reaction coefficients. The simulation is conducted on three anatomical brain sections(sagittal, coronal, and horizontal) derived from the [following stl model](...TODO LINK...). This test highlights the methods' applicability to realistic biological scenarios involving heterogeneous media.