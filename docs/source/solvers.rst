Algorithms
=============================

The ``GGLasso`` package contains solvers for several (Multiple) Graphical Lasso problem formulations. See :ref:`Mathematical description` for an overview of problem formulations.
This page aims to give an overview of the functions you need to call for solving a certain problem.

A popular algorithm for solving SGL and MGL problems is the ADMM algorithm. The ``GGLasso`` package contains and ADMM solver for all problem formulations as well as a proximal point solver for MGL problems without latent variables (called PPDNA).


All solvers need as input an empirical covariance matrix (or a collection of matrices for MGL) and a starting point. Using the identity matrix as strarting point, if no better guess is available, typically works fine.

SGL solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For SGL problems, use the solver . It can also solve latent variable Graphcial Lasso problems via setting the option ``latent=True`` and specifying a positive value for the option ``mu1``, the penalty parameter for the nuclear norm.


MGL solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For MGL problems, without latent variables two solvers are available, namely.
Both solvers have the option ``reg`` which can be set either to ``reg = 'GGL'`` for Group Graphical Lasso rpboelms or ``reg = 'FGL'`` for Fused Graphical Lasso problems.


If you want to model latent variables, only the ADMM solver works. 



