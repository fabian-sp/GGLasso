Mathematical description
=============================

The ``GGLasso`` package can solve several problem formulations related to Graphical Lasso. On this page, we aim to define the exact formulation for each problem.

Single Graphical Lasso problems (SGL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a multivariate Gaussian variable

.. math::
   X \sim \mathcal{N}(\mu, \Sigma)

Zeros in the precision matrix, :math:`\Sigma^{-1}` correspond to conditional independence of two components of :math:`X` which motivates a sparse estimation of :math:`\Sigma^{-1}`.
Typically, we are given a sample of :math:`N` independent samples of :math:`X` for which we can compute the empirical covariance matrix :math:`S`.
Even though :math:`S^{-1}` (if exists) is the maximum-likelihood estimator of the precision matrix, it is not guaranteed to be sparse. 

This leads to the nonsmooth convex optimization problem, known under the name of Graphical Lasso [1]_, given by

.. math::
   \min_{\Theta \in \mathbb{S}^p} - \log \det \Theta + \mathrm{Tr}(S\Theta) + \lambda \|\Theta\|_{1,od}

where :math:`\|\cdot\|_{1,od}` is the sum of absolute values of all off-diagonal elements and :math:`\mathrm{Tr}` is the trace of a matrix.

SGL with latent variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In presence of latent variables, the precision matrix of the marginal of observable variables turns out to have the structure *sparse - low rank* [4]_. The problem can then be formulated as  

.. math::
   \min_{\Theta, L \in \mathbb{S}^p} - \log \det (\Theta -L) + \mathrm{Tr}(S(\Theta-L)) + \lambda \|\Theta\|_{1,od} + \mu \|L\|_{\star}

where :math:`\|\cdot\|_{\star}` is the nuclear norm (sum of singular values). :math:`\Theta` represents the sparse part while :math:`L` encodes the low rank component.

Multiple Graphical Lasso problems (MGL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In many applications, compositional or temporal data is available. Hence, there has been an increased interest in estimating precision matrices for multiple instances jointly [2]_, [3]_. Group Graphical Lasso (GGL) describes the problem of estimating precision matrices across multiple instances of the same class under the assumption that the sparsity patterns are similar.
On the other hand, for time-varying data Fused Graphical Lasso was designed in order to get time-consistent estimates of the precision matrices.

More generally, the problem formulation of Multiple Graphical Lasso is given by

.. math::
   \min_{\Theta}\quad \sum_{k=1}^{K} \left(-\log\det(\Theta^{(k)}) + \langle S^{(k)},  \Theta^{(k)} \rangle \right)+ \mathcal{P}(\Theta).

In the above, the feasible set is the :math:`K`-fold product of :math:`\mathbb{S}^p_{++}`. We denote an element of this space :math:`\Theta =  (\Theta^{(1)}, \dots , \Theta^{(K)})`. As input, we have the empirical covariance matrices :math:`S =  (S^{(1)}, \dots , S^{(K)})` given.

Group Graphical Lasso (GGL)
""""""""""""""""""""""""""""""""""""""""""""""""""  

For the Group Graphical Lasso problem, the regularization function is given by 

.. math::
   \mathcal{P}(\Theta) = \lambda_1 \sum_{k=1}^{K} \sum_{i \neq j} |\Theta_{ij}^{(k)}| + \lambda_2  \sum_{i \neq j} \left(\sum_{k=1}^{K} |\Theta_{ij}^{(k)}|^2 \right)^{\frac{1}{2}}

with positive numbers :math:`\lambda_1, \lambda_2`. The first term promotes off-diagonal sparsity of the estimator while the second term -- similar to the classical group penalty -- induces that the non-zero entries are present for all instances :math:`\Theta^{(k)}`.

Fused Graphical Lasso (FGL)
"""""""""""""""""""""""""""""""""""""""""""""""""" 

For the Fused Graphical Lasso problem, the regularization function is given by 

.. math::
   \mathcal{P}(\Theta) = \lambda_1 \sum_{k=1}^{K} \sum_{i \neq j} |\Theta_{ij}^{(k)}| + \lambda_2  \sum_{k=2}^{K}   \sum_{i \neq j} |\Theta_{ij}^{(k)} - \Theta_{ij}^{(k-1)}|

with positive numbers :math:`\lambda_1, \lambda_2`. The first term promotes off-diagonal sparsity of the estimator while the second term -- also known as total-variation penalty -- induces that subsequent estimates of :math:`\Theta^{(k)}` are similar.

MGL with latent variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analogous to SGL, we can extend MGL problems with latent variables. For instance, and ADMM algorithm and software are available for FGL [3]_. The problem formulation then becomes 

.. math::
   \min_{\Theta, L}\quad \sum_{k=1}^{K} \left(-\log\det(\Theta^{(k)}- L^{(k)}) + \langle S^{(k)},  \Theta^{(k)} - L^{(k)} \rangle \right)+ \mathcal{P}(\Theta) +\sum_{k=1}^{K} \mu_k \|L^{(k)}\|_{\star}.



Optimization algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A popular algorithm for solving SGL and MGL problems is the ADMM algorithm [2]_, [3]_, [5]_. Alternatively, a proximal point algorithm with a semismooth Newton method for solving the subproblem was proposed (called PPDNA). 

The ``GGLasso`` package contains and ADMM solver for all problem formulations as well as the PPDNA solver for MGL problems without latent variables.

References
^^^^^^^^^^^

.. [1]  Friedman, J., Hastie, T., and Tibshirani, R. (2007).  Sparse inverse covariance estimation with the Graphical Lasso. Biostatistics, 9(3):432–441.
.. [2]  Danaher, P., Wang, P., and Witten, D. M. (2013).  The joint Graphical Lasso for inverse covariance estimation across multiple classes. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 76(2):373–397.
.. [3] Tomasi, F., Tozzo, V., Salzo, S., and Verri, A. (2018).  Latent variable time-varying network inference. InProceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM.
.. [4]  Chandrasekaran, V., Parrilo, P. A., and Willsky, A. S. (2012). Latent variable graphical model selection via convex optimization. The Annals of Statistics,40(4):1935–1967.
.. [5] Ma,  S., Xue,  L., and Zou, H.  (2013). Alternating direction methods for latent variable gaussian graphical model selection. Neural Computation, 25(8):2172–2198.




