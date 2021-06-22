"""
Benchmarking
=================================
We compare methods solving Graphical lasso problem.
The methods are implemented in sklearn, regain and gglasso librariries, respectively.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.contrib import tzip

from benchmarks.utilita import benchmark_parameters, load_dict, dict_shape
from benchmarks.plots import plot_bm

#%%
#  Synthetic power-law networks
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print("\n Shape of empirical covariance matrix:", dict_shape(load_dict('S_dict')))
print("\n Shape of the sample matrix:", dict_shape(load_dict('X_dict')))
print("\n Shape of true precisin matrix:", dict_shape(load_dict('Theta_dict')))

#%%
#  Model solution Z*
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
model_solution = load_dict('Z_dict')
print("\n Shape of empirical covariance matrix:", dict_shape(model_solution))
model_solution.keys()

#%%
#  Hyperparameters for methods comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We select default values of relative (rtol) and absolute (tol) tolerance rates for stopping crtireion in each method we compare.
# Using a greed search, we find the best hyperparameters which lead to minimum run time of the algorithm.

sk_params, rg_params, gglasso_params, lambda_list = benchmark_parameters()

#%%
#  Results of benchmarking
# ^^
df = pd.read_csv("data/synthetic/bm2000.csv", sep=",")
df.tail()


lambda_list = np.unique(df["l1"].values)

#%%
#  Benchmark methods execution time and accuracy
# ^^
# Main entry point
if __name__ == "__main__":

    plot_bm(df, lambda_list=lambda_list)
    plt.show()