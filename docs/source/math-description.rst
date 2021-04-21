Mathematical description
=============================

The ``GGLasso`` package can solve several problem formulations related to Graphical Lasso. On this page, we aim to define the exact formulation for each problem.

Single Graphical Lasso problems (SGL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a multivariate Gaussian variable

.. math::
   \mathcal{X} \sim \mathcal{N}(\mu, \Sigma) \in \mathbb{R}^p

Zeros in the precision matrix, :math:`\Sigma^{-1}` correspond to conditional independence of two components of :math:`X` which motivates a sparse estimation of :math:`\Sigma^{-1}`.
Typically, we are given a sample of :math:`N` independent samples of :math:`X` for which we can compute the empirical covariance matrix :math:`S`.
Even though :math:`S^{-1}` (if exists) is the maximum-likelihood estimator of the precision matrix, it is not guaranteed to be sparse. 

This leads to the nonsmooth convex optimization problem, known under the name of Graphical Lasso [ref1]_, given by

.. math::
   \min_{\Theta \in \mathbb{S}^p_{++}} - \log \det \Theta + \mathrm{Tr}(S\Theta) + \lambda \|\Theta\|_{1,od}

where :math:`\mathbb{S}^p_{++}` is the cone of symmetric positive definite matrices of size :math:`p \times p`, :math:`\|\cdot\|_{1,od}` is the sum of absolute values of all off-diagonal elements and :math:`\mathrm{Tr}` is the trace of a matrix.

SGL with latent variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In presence of latent variables, the precision matrix of the marginal of observable variables turns out to have the structure *sparse - low rank* [ref4]_. The problem can then be formulated as  

.. math::
   \min_{\Theta, L \in \mathbb{S}^p} - \log \det (\Theta -L) + \mathrm{Tr}(S(\Theta-L)) + \lambda_1 \|\Theta\|_{1,od} + \mu_1 \|L\|_{\star}

where :math:`\|\cdot\|_{\star}` is the nuclear norm (sum of singular values). :math:`\Theta` represents the sparse part while :math:`L` encodes the low rank component.

Multiple Graphical Lasso problems (MGL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In many applications, compositional or temporal data is available. Hence, there has been an increased interest in estimating precision matrices for multiple instances jointly [ref2]_, [ref3]_. Mathematically, we consider :math:`K` Gaussians

.. math::
   \mathcal{X}^{(k)} \sim \mathcal{N}(\mu^{(k)}, \Sigma^{(k)})\in \mathbb{R}^{p}


Group Graphical Lasso (GGL) describes the problem of estimating precision matrices across multiple instances of the same class under the assumption that the sparsity patterns are similar.
On the other hand, for time-varying data Fused Graphical Lasso was designed in order to get time-consistent estimates of the precision matrices, i.e. :math:`K` is the temporal index.

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

Analogous to SGL, we can extend MGL problems with latent variables.  The problem formulation then becomes 

.. math::
   \min_{\Theta, L}\quad \sum_{k=1}^{K} \left(-\log\det(\Theta^{(k)}- L^{(k)}) + \langle S^{(k)},  \Theta^{(k)} - L^{(k)} \rangle \right)+ \mathcal{P}(\Theta) +\sum_{k=1}^{K} \mu_{1,k} \|L^{(k)}\|_{\star}.

An ADMM algorithm and software is already available for FGL [ref3]_, however in [ref3]_ also the deviation of the low rank matrices is included in the penalty.

GGL - the nonconforming case
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So far, we have assumed that each component of :math:`\mathcal{X}^{(k)}` is present in each of the :math:`K` instances. However, in many practical situations this will not be the case. For example, assume that we have :math:`K` datasets of microbiome abundances but not every microbiome species (OTU) was measured in each dataset. Hence, we may want to estimate the association network but with a group sparsity penalty on all overlapping pairs of species. 

Consequently, assume that we have :math:`\mathcal{X}^{(k)} \sim \mathcal{N}(\mu^{(k)}, \Sigma^{(k)})\in \mathbb{R}^{p_k}` and that there exist groups of overlapping pairs of variables :math:`G_1, \dots, G_L` with 

.. math::
	G_l = \{(i_l^k, j_l^k) \in \mathbb{N}^2 \vert k \in K_l \}, \quad K_l \subset K

where :math:`k \in K_l` if and only if the pair of variables corresponding to :math:`G_l` exists in :math:`\mathcal{X}^{(k)}`. In that case :math:`(i_l^k, j_l^k)` are the indices of the relevant entry in :math:`\Theta^{(k)}` for group :math:`l`.

Now, the associated GGL regularizer becomes 

.. math::
	\mathcal{P}(\Theta) = \lambda_1 \sum_{i \neq j, k} |\Theta_{ij}^{(k)}| + \lambda_2 \sum_{l=1}^{L}\beta_l \|\Theta_{[l]}\|

where 
:math:`\Theta_{[l]}` is the vector with entries :math:`\{\Theta_{i_l^k j_l^k}^{(k)} \vert~ k \in K_l\} \in \mathbb{R}^{|K_l|}`. The scaling factor :math:`\beta_l > 0` is set to :math:`\beta_l = \sqrt{|K_l|}` in order to account for distinct group sizes.

In ``GGLasso`` we implemented an ADMM algorithm for the above described problem formulation, possibly extended with latent variables. 

Optimization algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All of the above problem formulations are instances of nonlinear, convex and nonsmooth optimization problems. See :ref:`Algorithms` for an overview of solvers which we implemented for these problems and a short guide on how to use them.

References
^^^^^^^^^^^

.. [ref1]  Friedman, J., Hastie, T., and Tibshirani, R. (2007).  Sparse inverse covariance estimation with the Graphical Lasso. Biostatistics, 9(3):432–441.
.. [ref2]  Danaher, P., Wang, P., and Witten, D. M. (2013). The joint graphical lasso for inverse covariance estimation across multiple classes. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 76(2):373–397.
.. [ref3] Tomasi, F., Tozzo, V., Salzo, S., and Verri, A. (2018). Latent Variable Time-varying Network Inference. InProceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM.
.. [ref4]  Chandrasekaran, V., Parrilo, P. A., and Willsky, A. S. (2012). Latent variable graphical model selection via convex optimization. The Annals of Statistics, 40(4):1935–1967.
.. [ref5] Ma,  S., Xue,  L., and Zou, H.  (2013). Alternating Direction Methods for Latent Variable Gaussian Graphical Model Selection. Neural Computation, 25(8):2172–2198.
.. [ref6] Zhang, Y., Zhang, N., Sun, D., and Toh, K.-C. (2020). A proximal point dual Newton algorithm for solving group graphical Lasso problems. SIAM J. Optim., 30(3):2197–2220.
.. [ref7] Zhang, N., Zhang, Y.,  Sun, D., and  Toh, K.-C. (2019). An efficient linearly convergent regularized proximal point algorithm for fused multiple graphical lasso problems.
.. [ref8] Boyd, S., Parikh, N., Chu, E., Peleato, B., and Eckstein, J. (2011). Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. Found. Trends Mach. Learn., 3(1):1–122.
.. [ref9] Witten, D. M., Friedman, J. H., and Simon, N. (2011). New Insights and Faster Computations for the Graphical Lasso. J. Comput. Graph. Statist., 20(4):892–900.
.. [ref10] Foygel, R. and Drton, M. (2010). Extended Bayesian Information Criteria for Gaussian Graphical Models. In Lafferty, J., Williams, C., Shawe-Taylor, J.,Zemel, R., and Culotta, A., editors, Advances in Neural Information Processing Systems, volume 23. Curran Associates, Inc.




