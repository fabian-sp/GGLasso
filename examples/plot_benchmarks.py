"""
Benchmarking
=================================

We compare the performance of the Graphical Lasso solvers implemented in ``GGLasso`` to two commonly used packages, i.e.

* `regain <https://github.com/fdtomasi/regain>`_ : contains an ADMM solver which is doing almost the same operations as ``ADMM_SGL``. For details, see the original paper. [ref3]_

* `sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html#sklearn.covariance.GraphicalLasso>`_: by default uses the coordinate descent algorithm which was originally proposed by Friedman et al. [ref1]_ 

The results can be generated using the notebook in ``benchmarks/benchmarks.ipynb`` in the Github repository.
From ``GGLasso`` we use the standard solver ``ADMM_SGL`` labeled by **gglasso** and the block-wise solver labeled by **gglasso-block**. For details, we refer to :ref:`SGL solver`.

"""

import pandas as pd
import numpy as np

from gallery_helper import plot_bm


#%%
#  Synthetic power-law networks
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We compare the solvers for a SGL problem using synthetic sparse powerlaw networks, which are generated as described in [ref2]_.
# The solvers are tested for different values of :math:`\lambda_1` and different dimensionalities. These values are printed below.
# We solve a SGL problem using each of the solvers independently, but using the same CPUs with 64GB of RAM in total.
#
# The results were generated on a machine equipped with `AMD Opteron(tm) 6378 @ 1.40GHz (max 2.40 GHz) (8 Cores per socket, hyper-threading)`.
#

df = pd.read_csv("../data/synthetic/bm5000.csv", index_col = 0)
df.reset_index(drop=True)

#%%

all_p_N= list(pd.unique(list(zip(df.p, df.N))))
print("Dimensionality and sample size: (p,N) =", all_p_N )

all_l1 = pd.unique(df.l1)
print("Values of lambda_1:  =", all_l1 )

#%%
#  Setup
# ^^^^^^^^^^^^^
# Each solver terminates after a given number of maximum iterations or when some optimality condition is met. Hence, the performance is difficult to compare as these optimality criteria may differ.
# Thus, we select a range of values for relative (rtol) and absolute (tol) tolerance (used in ADMM) and similarly tolerance values for ``sklearn``.


#%%
#  Calculating the accuracy 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# After solving each of the problems which each solver for different tolerance values, we compare the obtained solutions to a reference solution, denoted by :math:`Z^\ast`. 
#
# :math:`Z^\ast` is obtained by solving a SGL problem by one of the solvers for very small tolerance values (we used ``regain`` and set ``tol=rtol=1e-10``).
# Finally, for a solution :math:`Z`, we calculate its accuracy using the normalized Euclidean distance:
#
# .. math::
#   \text{accuracy}(Z) =  \frac{\|Z^\ast - Z \|}{ \| Z^\ast\| }.

# %%
# Runtime and accuracy with respect to :math:`\lambda_1`.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now, determine a maximal accuracy :math:`\epsilon`. For each solver, we now select the run with minimal runtime where :math:`\text{accuracy}(Z) \leq \epsilon` is fulfilled. 
# We plot the results for two values of :math:`\epsilon`:
# 

#%% 
# Accuracy of :math:`\epsilon=5\cdot10^{-3}`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


plot_bm(df, min_acc= 5e-3, lambda_list=all_l1)

#%% 
# Accuracy of :math:`\epsilon=5\cdot10^{-2}`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

plot_bm(df, min_acc = 5e-2, lambda_list=all_l1)
