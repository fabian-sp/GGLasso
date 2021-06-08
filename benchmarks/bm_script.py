#import os

import numpy as np
import pandas as pd
import time
#import itertools
from itertools import product


from regain_benchmark import regain_time
from sklearn_benchmark import sklearn_time
from gglasso_benchmark import gglasso_time

from utilita import network_generation, model_solution, benchmark_parameters 
from utilita import sparsity_benchmark, dict_shape, calc_hamming_dict
from utilita import benchmarks_dataframe,  drop_acc_duplicates

from plots import plot_accuracy, plot_scalability, plot_lambdas

from tqdm.contrib import tzip

pd.set_option('display.max_columns', None)

#%%

#from regain import datasets, utils


#%%
S_dict=dict()
X_dict=dict()
Theta_dict=dict()

p_list=[100, 200, 300]
N_list=[500, 1000, 1000]

print(" Network generation ".center(40, '-'))

for p, N in tzip(p_list, N_list):
    S, X, Theta = network_generation(p, N, M=10)
    print("p: %5d, N : %5d" % (p, N))

    S_dict[p, N] = S.copy()
    X_dict[p, N] = X.copy()
    Theta_dict[p, N] = Theta.copy()
    
#%%
print("\n Shape of S_i:", dict_shape(S_dict))
print("\n Shape of X_i:", dict_shape(X_dict))
print("\n Shape of Theta_i:", dict_shape(Theta_dict))


#%%

lambda_list = [0.1, 0.01, 0.001]
sk_params, rg_params, gglasso_params, lambda_list = benchmark_parameters(lambda_list = lambda_list)

#%%
model_time_dict = dict()
model_Z_dict = dict()
reference_solver = "regain"

print(f"Solving for a reference solution with solver {reference_solver}:")

for X, l1 in product(list(X_dict.values()), lambda_list):
    
    Z, Z_time, info = model_solution(solver=reference_solver, X=X, lambda1=l1)
    
    key = "p_" + str(X.shape[1]) + "_l1_" + str(l1)
    model_time_dict.update({key: Z_time})
    model_Z_dict.update({key: Z})


#%%
time_dict = dict()
accuracy_dict = dict()
Z_dict = dict()

n_iter = 2

#%%

S = list(S_dict.values())[0]
Omega_0 = np.eye(len(S))
method = 'single'
tol = 1e-5
rtol = 1e-5
l1 = 0.01
Z=model_Z_dict

for X, S in tzip(list(X_dict.values()), list(S_dict.values())):
    Omega_0 = np.eye(len(S))
    gg_time, gg_accuracy, Z_gg = gglasso_time(S=S, Omega_0=Omega_0, Z=model_Z_dict, lambda_list=lambda_list,
                                              n_iter=n_iter, gglasso_params=gglasso_params, warm_start=False)
    
    time_dict.update(gg_time)
    accuracy_dict.update(gg_accuracy)
    Z_dict.update(Z_gg)


        
#%%

for X, S in tzip(list(X_dict.values()), list(S_dict.values())):
    sk_time, sk_accuracy, Z_sk = sklearn_time(X=X, Z=model_Z_dict, sk_params=sk_params, lambda_list=lambda_list, n_iter=n_iter)
    
    time_dict.update(sk_time)
    accuracy_dict.update(sk_accuracy)
    Z_dict.update(Z_sk)
    

        
#%%

for X, S in tzip(list(X_dict.values()), list(S_dict.values())):
    rg_time, rg_accuracy, Z_rg = regain_time(X=X, Z=model_Z_dict, rg_params=rg_params, lambda_list=lambda_list, n_iter=n_iter)
    
    time_dict.update(rg_time)
    accuracy_dict.update(rg_accuracy)
    Z_dict.update(Z_rg)
    
         
#%%

hamming_dict = calc_hamming_dict(Theta_dict=Theta_dict, Z_dict=Z_dict, t_rounding=1e-8)


#%%

def benchmarks_dataframe(times=dict, acc=dict, hamming=dict):
    """
    Turn benchmark dictionaries into dataframes.
    :param times: dict
    Input dictionary where 'key' is the model and 'value' is its runtime.
    :param acc_dict: dict
    Input dictionary where 'key' is a model and 'value' is its corresponding accuracy given as:
    np.linalg.norm(Z - np.array(Z_i)) / np.linalg.norm(Z)
    where Z is our model solution and Z_i is the model.
    :param spars_dict: dict
    Input dictionary where 'key' is a model and 'value' is its corresponding sparsity measured by
    Hamming distance.
    :return: Pandas.DataFrame()
    """
    assert len(times) == len(acc) == len(hamming)

    all_dict = dict()
    for key in times.keys():
        all_dict[key] = {'time': times[key], 'accuracy': acc[key], 'hamming': hamming[key]}
    
    df = pd.DataFrame.from_dict(all_dict, orient = 'index')
    
    # split key into columns
    df['split'] = df.index.str.split('_')
    columns_names = ["method", "tol_str", "tol", "rtol_str", "rtol", "p_str", "p", "l1_str", "l1"]
    df[columns_names] = pd.DataFrame(df['split'].tolist(), index=df['split'].index)
    
    redundant_cols = ['split', "tol_str", "rtol_str", "p_str", "l1_str"]
    df = df.drop(redundant_cols, axis=1)
    
    convert_dict = {'tol': float, 'rtol': float, "p": int, "l1": float}
    df = df.astype(convert_dict)
    df = df.sort_values(['p', 'l1', 'method', 'time'])
    
    
    return df





#%% 

df = benchmarks_dataframe(times=time_dict, acc=accuracy_dict, hamming=hamming_dict)
df = drop_acc_duplicates(df)
df = df.reset_index(drop=True)
df.head()


