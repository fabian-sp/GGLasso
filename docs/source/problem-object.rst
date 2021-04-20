Using the problem object
=============================



Model selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Choosing the regularization parameters :math:`\lambda_1` and :math:`\lambda_2` (and :math:`\mu` in the latent variable case) has a crucial impact how well the Graphical Lasso solution recovers true edges/ non-zero entries of the precision matrix.

``GGLasso`` contains model selection functionalities for each of the problem described in :ref:`Mathematical description`. Model selection is done via grid searches on the regularization parameters where the quality of a solution is assessed either with the AIC (Akaike Information Criterion) or the eBIC (Extended Bayesian Information Criterion).

Typically, the eBIC chooses sparser solutions and thus leads to less false discoveries. We refer to [ref10]_ for details.

We describe the model selection procedure in detail for the main three cases:

* SGL: solve on a path of :math:`\lambda_1` values or on a grid of :math:`(\lambda_1, \mu)` values if ``latent=True``. Choose the grid point where the eBIC is minimal.
* MGL and ``latent=False``: solve on a grid of :math:`(\lambda_1, \lambda_2)` values. Choose the grid point where the eBIC is minimal.
* MGL and ``latent=True``: in a first stage, solve SGL on a :math:`(\lambda_1, \mu)` for each instance :math:`k=1,\dots,K` independently. Then, do a grid search on :math:`(\lambda_1, \lambda_2)` values and for each :math:`\lambda_1` and each instance :math:`k=1,\dots,K` pick the :math:`\mu` value which had minimal eBIC in stage one. Then, pick again the grid point with minimal eBIC.