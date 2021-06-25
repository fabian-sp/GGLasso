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

We introduce `GGLasso`, a Python package that solves General Graphical Lasso problems.
The underlying statistical forward model is assumed to be of the following form:

$$
...
$$



$$
    \min_{\Omega \in \mathbb{S}^d, ect to} \qquad  C\beta = 0
$$

for several convex loss functions $f(\cdot,\cdot)$. This includes the constrained Lasso, the constrained scaled Lasso, sparse Huber M-estimators with linear equality constraints, and constrained (Huberized) Square Hinge Support Vector Machines (SVMs) for classification.

# Statement of need 

Currently, there is no Python package available that can solve these ubiquitous statistical estimation problems in a fast and efficient manner. 
`GGLasso` provides algorithmic strategies, including ADMM algorithms, to solve the underlying convex optimization problems with provable convergence guarantees. The `GGLasso` package is intended to fill the gap between popular Python tools such as [`scikit-learn`](https://scikit-learn.org/stable/) which <em>cannot</em> solve these constrained problems and general-purpose optimization solvers such as [`cvxpy`](https://www.cvxpy.org) that do not scale well for these problems and/or are inaccurate. `c-lasso` can solve the estimation problems at a single regularization level, across an entire regularization path, and includes three model selection strategies for determining the regularization parameter: a theoretically-derived fixed regularization, k-fold cross-validation, and stability selection. We show several use cases of the package, including an application of sparse log-contrast regression tasks for compositional microbiome data, and highlight the seamless integration into `R` via [`reticulate`](https://rstudio.github.io/reticulate/).

# Functionalities


# Acknowledgements
 

# References


