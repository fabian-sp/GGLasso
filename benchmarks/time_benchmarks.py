
import os
os.chdir(os.path.pardir)

import numpy as np
import pandas as pd
import time

from gglasso.solver.single_admm_solver import ADMM_SGL
from gglasso.solver.single_admm_solver import block_SGL
from gglasso.helper.data_generation import time_varying_power_network, group_power_network, sample_covariance_matrix
from gglasso.helper.model_selection import single_grid_search

from benchmarks.benchmarks import models_to_dict, sklearn_time_benchmark, admm_time_benchmark, model_solution, benchmark_parameters
from benchmarks.benchmarks import time_benchmark, sparsity_benchmark
from benchmarks.benchmarks import sk_scaling, single_scaling, block_scaling

from benchmarks.plots import plot_accuracy, plot_scalability
from benchmarks.utils import network_generation, dict_shape, hamming_dict
from benchmarks.utils import benchmarks_dataframe, best_time_dataframe, drop_acc_duplicates

from regain.covariance import GraphicalLasso as rg_GL

from sklearn.covariance import GraphicalLasso as sk_GL
from sklearn import set_config
set_config(print_changed_only=False)


S_dict=dict()
X_dict=dict()
Theta_dict=dict()

# p_list=[100, 500, 1000, 2500, 5000, 10000]
# N_list=[200, 1000, 2000, 5000, 10000, 20000]
p_list=[100, 200]
N_list=[200, 400]

print(" Power network generation ".center(40, '-'))

for p, N in zip(p_list, N_list):
    try:
        start = time.perf_counter()
        S, X, Theta = network_generation(p, N, K=1, M=2)
        end = time.perf_counter()
        print("p: %5d, N : %5d, Time : %5.4f" % (p, N, end-start))
    except:
        print("Power network cannot be generated")
        print("Tip: increase the number of sub-blocks M")
        break

    S_dict[p, N] = S
    X_dict[p, N] = X
    Theta_dict[p, N] = Theta


print("\n Shape of S_i:", dict_shape(S_dict))
print("\n Shape of X_i:", dict_shape(X_dict))
print("\n Shape of Theta_i:", dict_shape(Theta_dict))


sk_params, rg_params, admm_params = benchmark_parameters(S_dict=S_dict, sk_tol_list=[0.5], enet_list=[0.5])
lambda_list = [0.5, 0.1, 0.05]


time_dict = dict()
accuracy_dict = dict()
Z_dict = dict()

for l1 in lambda_list:
    for X, S in zip(list(X_dict.values()), list(S_dict.values())):
        times, accs, precs = time_benchmark(X=X, S=S, Z_model="sklearn", lambda1=l1, n_iter=5,
                                            sk_params=sk_params, rg_params=rg_params, admm_params=admm_params)

        time_dict.update(times)
        accuracy_dict.update(accs)
        Z_dict.update(precs)


sparsity = hamming_dict(Theta_dict=Theta_dict, Z_dict=Z_dict, t_rounding=1e-4)

df = benchmarks_dataframe(times=time_dict, acc_dict=accuracy_dict, spars_dict=sparsity)
df = drop_acc_duplicates(df)
df.to_csv("time_df_v2.csv", encoding='utf-8', index=False)