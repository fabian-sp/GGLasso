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

We introduce `GGLasso`, a Python package that solves General Graphical Lasso problems. Given a multivariate Gaussian $\mathcal{X} \sim \mathcal{N}(\mu, \Sigma) \in \mathbb{R}^p$, statisticians are naturally interested in estimating conditional independencies of $\mathcal{X}$. A fundamental result of graphical models \cite{Lauritzen1996} states that for a multivariate Gaussian two variables $\mathcal{X}_{i}$ and $\mathcal{X}_j$ are independent -- conditional on all other variables -- if and only if $\Sigma^{-1}_{ij}=0$.
Hence, estimating the inverse covariance matrix -- also called \textit{precision matrix} -- is sufficient in order to infer the conditional dependence structure of $\mathcal{X}$. Initially proposed by [@Friedman2007] and [@Yuan2007], *Graphical Lasso* translates this into a nonsmooth, convex optimization problem given by

$$
\min_{\Theta \in \mathbb{S}^p_{++}} \quad - \log \det \Theta + \langle S,  \Theta \rangle+ \lambda \|\Theta\|_{1,od}.
$$
In the above, $\lambda >0$ is a regularization parameter and $\|\Theta\|_{1,od} := \sum_{i\neq j} |\Theta_{ij}|$ denotes the off-diagonal $\ell_1$-norm of a matrix.

Multiple Graphical Lasso

$$
\min_{\Theta \in \mathbb{S}_{++}^K }\quad \sum_{k=1}^{K} \left(-\log\det(\Theta^{(k)}) + \langle S^{(k)},  \Theta^{(k)} \rangle \right)+ \mathcal{P}(\Theta).
$$




# Statement of need 

Currently, there is no Python package for solving Group Graphical Lasso problems. The Single Graphical Lasso problem is implemented in `scikit-learn` [@Pedregosa2011], however with no extension for latent variables. The package `regain` [@Tomasi2018] contains solvers for Single and Fused Graphical Lasso problem, with and without latent variables. We propose a uniform framework for solving Graphical Lasso problems and provide solvers for Group Graphical Lasso problems (with and without latent variables). Moreover, we implement a block-wise ADMM solver for SGL problems following [@Witten2011] which is severly faster than current solvers for large, sparse problems. In the table below we give an overview of existing functionalities and the `GGLasso` package.

|       | scikit-learn |  regain |  GGLasso | comment |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| SGL              | :heavy_check_mark:    | :heavy_check_mark:       | :heavy_check_mark:       | new: block-wise solver           |
| SGL + latent     | :no_entry_sign:       | :heavy_check_mark:       | :heavy_check_mark:       |             |
| GGL              | :no_entry_sign:       | :no_entry_sign:          | :heavy_check_mark:       |             |
| GGL + latent     | :no_entry_sign:       | :no_entry_sign:          | :heavy_check_mark:       |             |
| FGL              | :no_entry_sign:       | :heavy_check_mark:       | :heavy_check_mark:       | new: Proximal point solver            |
| FGL + latent     | :no_entry_sign:       | :heavy_check_mark:       | :heavy_check_mark:       |             |
| GGL nonconforming  (+latent)    | :no_entry_sign:       | :no_entry_sign:       | :heavy_check_mark:       |             |



# Functionalities


# Acknowledgements
 

# References


