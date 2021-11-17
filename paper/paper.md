---
title: 'GGLasso - a Python package for General Graphical Lasso computation'
tags:
  - Python
  - graphical lasso
  - latent graphical model
  - structured sparsity
  - convex optimization
  - ADMM
authors:
  - name: Fabian Schaipp
    orcid: 0000-0002-0673-9944
    affiliation: "1"
  - name: Oleg Vlasovets
    affiliation: "2,3"
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
date: 17 November 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:

---

# Summary

We introduce `GGLasso`, a Python package for solving General Graphical Lasso problems. The Graphical Lasso scheme, introduced by [@Friedman2007] (see also [@Yuan2007;@Banerjee2008]), estimates a sparse inverse covariance matrix $\Theta$ from multivariate Gaussian data $\mathcal{X} \sim \mathcal{N}(\mu, \Sigma) \in \mathbb{R}^p$. Originally proposed by [@Dempster:1972] under the name Covariance Selection, this estimation framework has been extended to include latent variables in [@Chandrasekaran2012]. Recent extensions also include the joint estimation of multiple inverse covariance matrices, see, e.g., in [@Danaher2013; @Tomasi2018]. The `GGLasso` package contains methods for solving the general problem formulation:

<div class="math">
\begin{align}
\label{eq:problem}
\min_{\Theta, L \in \mathbb{S}_{++}^K }\quad \sum_{k=1}^{K} \left(-\log\det(\Theta^{(k)} - L^{(k)}) + \langle S^{(k)},  \Theta^{(k)} - L^{(k)} \rangle \right)+ \mathcal{P}(\Theta) +\sum_{k=1}^{K} \mu_{1,k} \|L^{(k)}\|_{\star}.
\end{align}
</div>

Here, we denote with $\mathbb{S}_{++}^K$ the $K$-product of the space of symmetric, positive definite matrices. Moreover, we write $\Theta = (\Theta^{(1)},\dots,\Theta^{(K)})$ for the sparse component of the inverse covariances and $L = (L^{(1)},\dots,L^{(K)})$ for the low rank components, formed by potential latent variables. Here, $\mathcal{P}$ is a regularization function that induces a desired sparsity structure. The above problem formulation subsumes important special cases, including the single (latent variable) Graphical Lasso, the Group, and the Fused Graphical Lasso.

# Statement of need

Currently, there is no Python package available for solving general Graphical Lasso instances. The standard single Graphical Lasso problem (SGL) can be solved in `scikit-learn` [@Pedregosa2011]. The `skggm` package provides several algorithmic and model selection extensions for the single Graphical Lasso problem [@Laska2017]. The package `regain` [@Tomasi2018] comprises solvers for single and Fused Graphical Lasso problems, with and without latent variables. With `GGLasso`, we make the following contributions:

- Proposing a uniform framework for solving Graphical Lasso problems.
- Providing solvers for Group Graphical Lasso problems (with and without latent variables).
- Providing a solver for -- what we call -- *nonconforming GGL* problems where not all variables need to be present in every instance. We detail a use case of this novel extension on synthetic data.
- Implementing a block-wise ADMM solver for SGL problems following [@Witten2011] as well as proximal point solvers for FGL and GGL problems [@Zhang2021; @Zhang2020].

In the table below we give an overview of existing functionalities and the `GGLasso` package.


|                                 | scikit-learn |  regain     |  GGLasso | comment |
| -----------                     | -----------  | ----------- | ----------- | ----------- |
| SGL                             | **yes**      | **yes**       | **yes**       | new: block-wise solver           |
| SGL + latent                    | **no**       | **yes**       | **yes**       |             |
| GGL                             | **no**       | **no**          | **yes**       |             |
| GGL + latent                    | **no**       | **no**          | **yes**       |             |
| FGL                             | **no**       | **yes**       | **yes**       | new: proximal point solver            |
| FGL + latent                    | **no**       | **yes**       | **yes**       |             |
| GGL nonconforming  (+latent)    | **no**       | **no**       | **yes**       |             |



# Functionalities

## Installation and problem instantiation

`GGLasso` can be installed via `pip`.

```shell
pip install gglasso
```

The central object of `GGLasso` is the class `glasso_problem` which streamlines the solving or model selection procedure for SGL, GGL, and FGL problems with or without latent variables.

As an example, we instantiate a single Graphical Lasso problem (see the problem formulation below). We input the empirical covariance matrix `S` and the number of samples `N`. We can choose to model latent variables and set the regularization parameters via the other input arguments.

```python
# Import the main class of the package
from gglasso.problem import glasso_problem

# Define a SGL problem instance with given data S
problem  = glasso_problem(S, N, reg = None,
                          reg_params = {'lambda1': 0.01}, latent = False)
```

As a second example, we instantiate a Group Graphical Lasso problem with latent variables. Typically, the optimal choice of the regularization parameters are not known and are determined via model selection.

```python
# Define a GGL problem instance with given data S
problem  = glasso_problem(S, N, reg = "GGL", reg_params = None, latent = True)
```

Depending on the input arguments, `glasso_problem` comprises two main modes:

- if regularization parameters are specified, the problem-dependent default solver is called.
- if regularization parameters are *not* specified, `GGLasso` performs model selection via grid search and the extended BIC criterion [@Foygel2010]).

```python
problem.solve()
problem.model_selection()
```

For further information on the input arguments and methods, we refer to the [detailled documentation](https://gglasso.readthedocs.io/en/latest/problem-object.html).

![Illustration of the latent SGL: The estimated inverse covariance matrix $\hat \Omega$ decomposes into a sparse component $\hat \Theta$ (central) and a low-rank component $\hat L$ (right). \label{fig1}](../docs/source/pictures/SLRDecomp.pdf){width=90%}

## Problem formulation

We list important special cases of the problem formulation given in \autoref{eq:problem}. For a mathematical formulation of each special case, we refer to the [documentation](https://gglasso.readthedocs.io/en/latest/math-description.html).

### Single Graphical Lasso (*SGL*): {#SGL}
For $K=1$, the problem reduces to the single (latent variable) Graphical Lasso where
$$
\mathcal{P}(\Theta) = \lambda_1 \sum_{i \neq j} |\Theta_{ij}|.
$$
An illustration of the single latent variable Graphical Lasso model output is shown in \autoref{fig1}.

### Group Graphical Lasso (*GGL*): {#GGL}
For
$$
\mathcal{P}(\Theta) = \lambda_1 \sum_{k=1}^{K} \sum_{i \neq j} |\Theta_{ij}^{(k)}| + \lambda_2  \sum_{i \neq j} \left(\sum_{k=1}^{K} |\Theta_{ij}^{(k)}|^2 \right)^{\frac{1}{2}}
$$
we obtain the Group Graphical Lasso as formulated in [@Danaher2013].

### Fused Graphical Lasso (*FGL*): {#FGL}
For
$$
\mathcal{P}(\Theta) = \lambda_1 \sum_{k=1}^{K} \sum_{i \neq j} |\Theta_{ij}^{(k)}| + \lambda_2  \sum_{k=2}^{K}   \sum_{i \neq j} |\Theta_{ij}^{(k)} - \Theta_{ij}^{(k-1)}|
$$
we obtain Fused (also called Time-Varying) Graphical Lasso [@Danaher2013; @Tomasi2018; @Hallac2017].

### Nonconforming GGL:

Consider the GGL case in a situation where not all variables are observed in every instance $k=1,\dots,K$. `GGLasso` is able to solve these problems and include latent variables. We provide the mathematical details in the [documentation](https://gglasso.readthedocs.io/en/latest/math-description.html#ggl-the-nonconforming-case) and give an [example](https://gglasso.readthedocs.io/en/latest/auto_examples/plot_nonconforming_ggl.html#sphx-glr-auto-examples-plot-nonconforming-ggl-py).


## Optimization algorithms

The `GGLasso` package implements several methods with provable convergence guarantees for solving the optimization problems formulated above.

- *ADMM*: for all problem formulations we implemented the ADMM algorithm [@Boyd2011]. ADMM is a flexible and efficient optimization scheme which is specifically suited for Graphical Lasso problems as it only relies on efficient computation of the proximal operators of the involved functions [@Danaher2013; @Tomasi2018; @Ma2013].  

- *PPDNA*: for GGL and FGL problems without latent variables, we included the proximal point solver proposed in [@Zhang2021; @Zhang2020]. According to the numerical experiments in [@Zhang2020], PPDNA can be an efficient alternative to ADMM especially for fast local convergence.

- *block-ADMM*: for SGL problems without latent variables, we implemented a method which solves the problem block-wise, following the proposal in [@Witten2011]. This wrapper simply applies the ADMM solver to all connected components of the empirical covariance matrix after thresholding.

![Runtime comparison for SGL problems of varying dimension and sample size at three different $\lambda_1$ values. The left column shows the runtime at low accuracy, the right column at high accuracy. \label{fig2}](../docs/source/pictures/runtime_merged.pdf){width=90%}


## Benchmarks and applications

In our example gallery, we included benchmarks comparing the solvers in `GGLasso` to state-of-the-art software as well as illustrative examples explaining the usage and functionalities of the package. We want to emphasize the following examples:

- [Benchmarks](https://gglasso.readthedocs.io/en/latest/auto_examples/plot_benchmarks.html#sphx-glr-auto-examples-plot-benchmarks-py) for SGL problems: our solver is competitive with `scikit-learn` and `regain`. The newly implemented block-wise solver is highly efficient for large sparse networks (see \autoref{fig2} for runtime comparison at [low and high accuracy](https://gglasso.readthedocs.io/en/latest/auto_examples/plot_benchmarks.html#calculating-the-accuracy), respectively).

- [Soil microbiome application](https://gglasso.readthedocs.io/en/latest/auto_examples/plot_soil_example.html#sphx-glr-auto-examples-plot-soil-example-py): following [@Kurtz2019], we demonstrate how latent variables can be used to identify hidden confounders in microbial network inference.

- [Nonconforming GGL](https://gglasso.readthedocs.io/en/latest/auto_examples/plot_nonconforming_ggl.html#sphx-glr-auto-examples-plot-nonconforming-ggl-py): we illustrate how to use `GGLasso` for GGL problems with missing variables.


# Acknowledgements

We thank Prof. Dr. Michael Ulbrich, TU Munich, for supervising the Master's thesis of FS that led to the development of the software. We also thank Dr. Zachary D. Kurtz for helping with testing of the latent graphical model implementation.

# References
