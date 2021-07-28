---
title: 'GGLasso - a Python package for General Graphical Lasso computation'
tags:
  - Python
  - graphical lasso
  - structured sparsity
  - convex optimization
  - ADMM
authors:
  - name: Fabian Schaipp
    affiliation: 1
  - name: Oleg Vlasovets
    affiliation: 2,3
  - name: Michael Ulbrich
    affiliation: 1
  - name: Christian L. Müller
    orcid: 0000-0002-3821-7083
    affiliation: "2,3,4"

affiliations:
  - name: Technische Universität München
    index: 1
  - name: Institute of Computational Biology, Helmholtz Zentrum München
    index: 2
  - name: Department of Statistics, Ludwig-Maximilians-Universität München
    index: 3
  - name: Center for Computational Mathematics, Flatiron Institute, New York
    index: 4
date: 13 May 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:

---

# Summary

We introduce `GGLasso`, a Python package that solves General Graphical Lasso problems. Given a multivariate Gaussian $\mathcal{X} \sim \mathcal{N}(\mu, \Sigma) \in \mathbb{R}^p$, statisticians are naturally interested in estimating conditional independencies of $\mathcal{X}$. A fundamental result of graphical models [@Lauritzen1996] states that for a multivariate Gaussian two variables $\mathcal{X}_{i}$ and $\mathcal{X}_j$ are independent -- conditional on all other variables -- if and only if $\Sigma^{-1}_{ij}=0$.
Hence, estimating the inverse covariance matrix -- also called \textit{precision matrix} -- is sufficient in order to infer the conditional dependence structure of $\mathcal{X}$. Initially proposed by [@Friedman2007] and [@Yuan2007], *Graphical Lasso* translates this into a nonsmooth, convex optimization problem given by

$$
\min_{\Theta \in \mathbb{S}^p_{++}} \quad - \log \det \Theta + \langle S,  \Theta \rangle+ \lambda \|\Theta\|_{1,od}.
$$

In the above, $\lambda >0$ is a regularization parameter and $\|\Theta\|_{1,od} := \sum_{i\neq j} |\Theta_{ij}|$ denotes the off-diagonal $\ell_1$-norm of a matrix.

Multiple Graphical Lasso is given by the problem

$$
\label{prob:mgl}
\min_{\Theta \in \mathbb{S}_{++}^K }\quad \sum_{k=1}^{K} \left(-\log\det(\Theta^{(k)}) + \langle S^{(k)},  \Theta^{(k)} \rangle \right)+ \mathcal{P}(\Theta).
$$

where $\mathcal{P}$ is a regularization function depending whether we do GGL or FGL. 



# Statement of need 

Currently, there is no Python package for solving Group Graphical Lasso problems. The Single Graphical Lasso problem is implemented in `scikit-learn` [@Pedregosa2011], however with no extension for latent variables. The package `regain` [@Tomasi2018] contains solvers for Single and Fused Graphical Lasso problems, with and without latent variables. With `GGLasso`, we make the following contributions:

* Proposing a uniform framework for solving Graphical Lasso problems. 
* Providing solvers for Group Graphical Lasso problems (with and without latent variables).
* Providing a solver for -- what we call -- *nonconforming GGL* problems where not all variables are contained in every instance. We demonstrate a usecase of such a formulation in the context of microbial consensus networks. 
* Implementing a block-wise ADMM solver for SGL problems following [@Witten2011] as well as proximal point solvers FGL and GGL problems.

In the table below we give an overview of existing functionalities and the `GGLasso` package.

|       | scikit-learn |  regain |  GGLasso | comment |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| SGL              | **YES**    | **YES**       | **YES**       | new: block-wise solver           |
| SGL + latent     | **NO**       | **YES**       | **YES**       |             |
| GGL              | **NO**       | **NO**          | **YES**       |             |
| GGL + latent     | **NO**       | **NO**          | **YES**       |             |
| FGL              | **NO**       | **YES**       | **YES**       | new: proximal point solver            |
| FGL + latent     | **NO**       | **YES**       | **YES**       |             |
| GGL nonconforming  (+latent)    | **NO**       | **NO**       | **YES**       |             |



# Functionalities

## Installation and problem initiation

`GGLasso` can be easily installed via `pip`.

```shell
pip install gglasso
```

The central object of `GGLasso` is the class `glasso_problem`, which then calls the correct solvers depending on the problem formulation. This class streamlines the solving or model selection procedure for SGL, GGL, FGL problems with or without latent variables.

We instantiate a Graphical Lasso problem: for this, we need the emirical covariance matrix/matrices `S` and the number of samples `N`. We can choose to model latent variables and set the regularization parameters via the other input arguments. 

```python
# Import the main class of the package
from gglasso.problem import glasso_problem

# Define a Graphical Lasso problem instance with default setting, 
# given data X, y, and constraints C.
problem  = glasso_problem(S, N, reg = "GGL", reg_params = None, latent = False)
```
The problem formulation is automatically derived from the input arguments of `glasso_problem`: the shape of the input `S` which determines whether we face a Single or Multiple Graphical Lasso problem. The problem has a representation method which prints the derived problem formulation:

```python
print(problem)
```
We refer to the [detailled documentation](https://gglasso.readthedocs.io/en/latest/problem-object.html) for further instructions.

## Problem formulation



### *SGL* Single Graphical Lasso: {#SGL} 


### *GGL* Group Graphical Lasso: {#GGL}
We solve (\ref{prob:mgl}) with 

$$
\mathcal{P}(\Theta) = \lambda_1 \sum_{k=1}^{K} \sum_{i \neq j} |\Theta_{ij}^{(k)}| + \lambda_2  \sum_{i \neq j} \left(\sum_{k=1}^{K} |\Theta_{ij}^{(k)}|^2 \right)^{\frac{1}{2}}
$$

### *FGL* Fused Graphical Lasso: {#FGL}

$$
\mathcal{P}(\Theta) = \lambda_1 \sum_{k=1}^{K} \sum_{i \neq j} |\Theta_{ij}^{(k)}| + \lambda_2  \sum_{k=2}^{K}   \sum_{i \neq j} |\Theta_{ij}^{(k)} - \Theta_{ij}^{(k-1)}|
$$

For a detailled mathematical description of all problem formulations see [the documentation](https://gglasso.readthedocs.io/en/latest/math-description.html).


## Optimization algorithms

The `GGLasso` package implements several methods with provable convergence guarantees for solving the optimization problems formulated above. 

* **ADMM**: for all problem formulations we implemented the ADMM algorithm [@Boyd2011]. ADMM is a flexible and efficient optimization scheme which is specifically suited for Graphical Lasso problems as it only relies on efficient computation of the proximal operators of the involved functions [@Danaher2013; @Tomasi2018; @Ma2013].  
* **PPDNA**: for GGL and FGL problems without latent variables, we implemented the proximal point solver proposed in [@Zhang2019; @Zhang2020]. According to the numerical experiments in [@Zhang2020], PPDNA can be an efficient alternative to ADMM especiall for fast local convergence.

* **BLOCK-ADMM**: for SGL problems without latent variables, we implement a method which solves the problem blockwise, following the proposal in [@Witten2011]. This wrapper simply applies the ADMM solver to all connected components of the empirical covariance matrix after thresholding.

# Acknowledgements
 

# References


