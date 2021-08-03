Algorithms
=============================

The ``GGLasso`` package contains solvers for several (Multiple) Graphical Lasso problem formulations. See :ref:`Mathematical description` for an overview of problem formulations.
This page aims to give an overview of the functions you need to call for solving a certain problem.

A popular algorithm for solving SGL and MGL problems is the ADMM [ref8]_, [ref5]_, [ref2]_, [ref3]_. Alternatively, a proximal point dual Newton algorithm (PPDNA) was proposed for GGL [ref6]_ and FGL [ref7]_.

The ``GGLasso`` package contains and ADMM solver for all problem formulations as well as the PPDNA solver for MGL problems without latent variables. For a detailled documentation of all of the solvers described below, see :ref:`Detailled solver documentation`.


**Note**: all solvers need as input an empirical covariance matrix (or a collection of matrices for MGL) and a starting point. Using the identity matrix as starting point, if no better guess is available, typically works fine. 

SGL solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For SGL problems, the standard ADMM is ``from gglasso.solver.single_admm_solver import ADMM_SGL``. However, it is shown in [ref9]_ that if the true precision matrix is block-sparse, it is sufficient to solve Graphical Lasso independently on each block. This gives a huge performance if the number of nodes :math:`p` is large because ADMM scales in :math:`\mathcal{O}(p^3)`. We refer to [ref9]_ for details. The resulting algorithm is implemented in ``from gglasso.solver.single_admm_solver import block_SGL``.

``ADMM_SGL`` can also solve latent variable Graphcial Lasso problems via setting the option ``latent=True`` and specifying a positive value for the option ``mu1``, the penalty parameter for the nuclear norm.



MGL solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For MGL problems without latent variables two solvers are available, namely 

* ADMM: use ``from gglasso.solver.admm_solver import ADMM_MGL``
* PPDNA: use ``from gglasso.solver.ppdna_solver import PPDNA``. A two-stage version with ADMM for initial iterations and PPDNA for local convergence is implemented in ``warmPPDNA``.

Note that the formulation of the FGL regularizer differs sligtly in [ref2]_, [ref3]_ and our implementation.

Both ADMM and PPDNA have the option ``reg`` which can be set either to ``reg = 'GGL'`` for Group Graphical Lasso problems or ``reg = 'FGL'`` for Fused Graphical Lasso problems. 


For MGL problems with latent variables, only the ADMM solver is available. 

Nonconforming GGL 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the setting which we describe in :ref:`GGL - the nonconforming case`, we implemented an ADMM solver, see ``from gglasso.solver.ext_admm_solver import ext_ADMM_MGL``.
This solver is slightly more complicated to call as you have to tell the solver where the overlapping pairs of variables can be found in the repsective precision matrices. This can be done with the function argument ``G`` which can be seen as a bookeeping array: you should specify a ``(2,L,K)``-shaped array where :math:`L` is the number of groups. 

If your data samples are a list of ``pd.DataFrame`` objects where each Dataframe has the shape ``(n_variables,n_samples)`` and the index contains **unique identifiers** (preferably integers) for all variables, you can create ``G`` by simply calling the following two functions from ``gglasso.helper.ext_admm_helper``.

.. code-block:: python

     ix_exist, ix_location = construct_indexer(list_of_samples) 
     G = create_group_array(ix_exist, ix_location)

Here, ``list_of_samples`` stands for your list of data samples as described above.

Also have a look at the :ref:`Nonconforming Group Graphical Lasso experiment` in our example gallery.


Further Remarks - proximal operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ADMM for Graphical Lasso relies on the efficient computation of proximal operators, in particular for the log-determinant function and the regularization function. For a convex function :math:`f` its proximal operator is given by

.. math::
	\mathrm{prox}_f(x) = \arg \min_z f(z) + \frac{1}{2} \|z-x\|^2

For solving the FGL problem with ADMM, the proximal operator of the total variation norm (TV), i.e. :math:`x\mapsto \sum_{i=1}^{N-1} |x_{i+1} - x_i|`, needs to be computed. 
This is a case where the proximal operator is not known in closed form. An efficient algorithm for the TV-prox is Condat's algorithm [ref11]_. ``GGLasso`` contains an efficient implementation of this algorithm using ``numba``. This is implemented in ``prox_tv`` from ``gglasso.solver.ggl_helper``. Further, according to [ref7]_ and [ref6]_ we implemented the Clarke differential of this and other proximal operators (for example the :math:`\ell_1`-norm, :math:`\ell_2`-norm and a weighted sum of these norms). For details, see ``gglasso.solver.ggl_helper``.