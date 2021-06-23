"""
Benchmarking
=================================
We compare performance of Graphical lasso solvers implemented in ``GGLasso``,
`regain <https://github.com/fdtomasi/regain>`_ and
`sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html#sklearn.covariance.GraphicalLasso>`_.

"""

import pandas as pd
import numpy as np

from benchmarks.utilita import benchmark_parameters, load_dict, dict_shape
from benchmarks.plots import plot_bm


#%%
#  Synthetic power-law networks
# ^^^^^^^^^^^^
# We compare the solvers for a SGL problem using sparse powerlaw networks described in :ref:`Basic example`.
# To validate the performnace of solvers, we generate such big networks of maximum 5000 nodes :math:`(p_{max}=5000)`.
# We solve a SGL problem by each of the solvers independently, but using the same CPUs with 64GB of RAM in total.
#
# `CPU: AMD Opteron(tm) 6378 @ 1.40GHz (max 2.40 GHz) (8 Cores per socket, hyper-threading)`.


print("\n Shapes of empirical covariance matrix:", dict_shape(load_dict('S_dict')))
print("\n Shapes of the sample matrix:", dict_shape(load_dict('X_dict')))
print("\n Shapes of true precisin matrix:", dict_shape(load_dict('5K_Theta_dict')))

#%%
#  Hyperparameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We select default values of relative (rtol) and absolute (tol) tolerance rates for solver stopping crtireion.
# Using a greed search, we find the best tolerance rates which lead to minimum run time of the solver.
# Also, we compare the solvers in differnt setups of penalization hyperparameter :math:`\lambda_1` which is responsible the sparsity level in a final solution.

sk_params, rg_params, gglasso_params, lambda_list = benchmark_parameters()

#%%
#  Calculate the accuracy of the solver
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# It is not feasible to obtain the identical solutions by all three solvers due to the differences in optimization strategies.
# So, one can compare the solvers, we select hyperparameters which allow us to have approximatelly similar solutions.
#
# Firstly, we solve a SGL problem at a high level of precision by one of the solvers and call the solution a `model solution` (:math:`Z^{*}`).
# Then, we solve the same problem by each of the solver at the lower level of precision (:math:`Z_{i}`).
# Finally, we calculate the normalized Eucledean distance (`accuracy`) between :math:`Z^{*}` and solution :math:`Z_{i}`:
#
# :math:`\begin{align} \text{accuracy} = \frac{\lVert Z^{*} - Z_{i} \rVert}{\lVert Z^{*} \rVert} \end{align}`
df = pd.read_csv("../data/synthetic/bm2000.csv", sep=",")
df.iloc[:, 1:].tail()

# %%
# Time and accuracy with respect to :math:`\lambda_1`.
# ^^^^^^^^^^^^
#
plot_bm(df, lambda_list=np.unique(df["l1"].values))