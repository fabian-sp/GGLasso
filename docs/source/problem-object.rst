Using the problem object
=============================

If you want to solve a (Multiple) Graphical Lasso problem, you can of course use the solvers we list in :ref:`Algorithms` directly. However, in most situations it is not clear how to choose the regularization parameters a priori and thus model selection becomes necessary. Below, we describe the model selection functionalities implemented in ``GGLasso``. In order to make its usage as simple as possible, we implemented a class ``glasso_problem`` which calls the solvers/model selection procedures internally and returns an estimator of the precision matrix/matrices in ``sklearn``-style.

Class glasso_problem
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gglasso.problem.glasso_problem
.. automethod:: gglasso.problem.glasso_problem.solve
.. automethod:: gglasso.problem.glasso_problem.model_selection

Other methods of glasso_problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automethod:: gglasso.problem.glasso_problem.set_modelselect_params
.. automethod:: gglasso.problem.glasso_problem.set_reg_params
.. automethod:: gglasso.problem.glasso_problem.set_start_point

Class GGLassoEstimator
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gglasso.problem.GGLassoEstimator	


Model selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Choosing the regularization parameters :math:`\lambda_1` and :math:`\lambda_2` (and :math:`\mu_1` in the latent variable case) has crucial impact how well the Graphical Lasso solution recovers true edges/ non-zero entries of the precision matrix.

``GGLasso`` contains model selection functionalities for each of the problem described in :ref:`Mathematical description`. Model selection is done via grid searches on the regularization parameters where the quality of a solution is assessed either with the AIC (Akaike Information Criterion) or the eBIC (Extended Bayesian Information Criterion).

Typically, the eBIC chooses sparser solutions and thus leads to less false discoveries. For a single precision matrix estimate :math:`\hat{\Theta}` of dimension :math:`p` with :math:`N` samples it is given by

.. math::
   eBIC_\gamma(\lambda_1) = N \left(-\log \det \hat{\Theta} + \langle S, \hat{\Theta}\rangle\right) + E \log N + 4 E \gamma \log p

where :math:`E` is the number of non-zero off-diagonal entries on the upper triangular of :math:`\hat{\Theta}` and :math:`\gamma \in [0,1]`. The larger you choose :math:`\gamma`, the sparser the solution will become. For MGL problems, we extend this (according to [ref2]_) by

.. math::
	eBIC_\gamma(\lambda_1, \lambda_2) := \sum_{k=1}^{K} N_k \left(-\log \det \hat{\Theta}^{(k)} + \langle S^{(k)}, \hat{\Theta}^{(k)}\rangle\right) + E_k \log N_k + 4 E_k \gamma \log p_k

where :math:`E_k` is the analogous of :math:`E` for :math:`\hat{\Theta}^{(k)}`. We refer to [ref10]_ for details on the eBIC.
