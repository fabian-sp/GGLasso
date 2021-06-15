"""
This script contains function helping to analyze the results of
time, sparsity and scaling benchmarks.
"""
import pandas as pd
import numpy as np
import pickle
import time
import os

from gglasso.helper.utils import hamming_distance
from gglasso.helper.data_generation import generate_precision_matrix, sample_covariance_matrix

from gglasso.solver.single_admm_solver import ADMM_SGL
from gglasso.solver.single_admm_solver import block_SGL, get_connected_components

from regain.covariance import GraphicalLasso as rg_GL

from sklearn.covariance import GraphicalLasso as sk_GL
from sklearn import set_config

set_config(print_changed_only=False)


def network_generation(p=int, N=int, M=10, style='powerlaw', gamma=2.8, prob=0.1, scale=False, nxseed=None):
    """
    Generates a law-power network with number of connected components bigger than 1.
    :param p: int
    Number of features.
    :param N: int
    Number of samples.
    :param M: int
    Number of subblock in each instance K.
    :return: S, X, Theta
    S - empirical covarince matrix.
    X - dual variable.
    Theta - true precision matrix.
    """
    Sigma, Theta = generate_precision_matrix(p=p, M=M, style=style, gamma=gamma, prob=prob, scale=scale, nxseed=nxseed)
    S, samples = sample_covariance_matrix(Sigma, N)
    X = samples.T

    return S, X, Theta


def benchmark_parameters(sk_tol_list=[1e-1,1e-2], enet_list=[1e-1,1e-2],
                         rg_tol_list=[1e-4, 1e-5, 5e-6, 1e-6], rg_rtol_list=[1e-4, 1e-5, 5e-6, 1e-6],
                         gglasso_tol_list=[5e-6, 1e-6, 1e-7, 5e-8], gglasso_rtol_list=[5e-6, 1e-6, 1e-7, 5e-8],
                         gglasso_stop='boyd', gglasso_method=['single', 'block'],
                         lambda_list=[0.5, 0.1, 0.05]):
    """
    Specify model hyperparameters.
    :param S_dict:
    :param sk_tol_list:
    :param enet_list:
    :param rg_tol_list:
    :param rg_rtol_list:
    :param admm_tol_list:
    :param admm_rtol_list:
    :param admm_stop:
    :param admm_method:
    :return:
    """

    # Sklearn params
    sk_params = {"tol": sk_tol_list, "enet": enet_list}

    # Regain params
    rg_params = {"tol": rg_tol_list, "rtol": rg_rtol_list}

    # ADMM params
    gglasso_params = {"tol": gglasso_tol_list, "rtol": gglasso_rtol_list, "stop": gglasso_stop,
                      "method": gglasso_method}

    print("\n Sklearn model parameters:", sk_params)
    print("\n Regain model parameters:", rg_params)
    print("\n ADMM model parameters:", gglasso_params)
    print("\n Lambda list:", lambda_list)

    return sk_params, rg_params, gglasso_params, lambda_list


def model_solution(solver="sklearn", X=np.array([]), lambda1=float, max_iter=1000, tol=1e-10, rtol=1e-10, enet=0.001):
    time_list = []
    if solver == "sklearn":

        start = time.perf_counter()
        Z = sk_GL(alpha=lambda1, max_iter=max_iter, tol=tol, enet_tol=enet, assume_centered=False).fit(X)
        Z.fit(X)
        end = time.perf_counter()

        time_list.append(end - start)

    elif solver == "regain":

        start = time.perf_counter()
        Z = rg_GL(alpha=lambda1, max_iter=max_iter, tol=tol, rtol=rtol,
                  assume_centered=False, verbose=True).fit(X)
        Z.fit(X)
        end = time.perf_counter()
        time_list.append(end - start)

    print(solver, Z)

    return Z.precision_, time_list, Z



def dict_shape(dict_=dict):
    shape_list = []

    for i in dict_.values():
        shape_list.append(np.array(i).shape)
    return shape_list


def calc_hamming_dict(Theta_dict=dict, Z_dict=dict, t_rounding=float):
    """
    Calculate Hamming distance between model solution Z and given solution Theta
    with a specified rounding accuracy t_rounding.
    :param Theta_dict: dict
    :param Z_dict: dict
    :param t_rounding: 1e-10, float
    :return: dict
    """
    sparsity_dict = dict()

    for keyT, Theta in Theta_dict.items():

        for key, Z in Z_dict.items():
            key_p = int(key.split('_p_')[1].split('_')[0])
            key_N = int(key.split('_N_')[1].split('_')[0])
            if (keyT[0] == key_p) and (keyT[1] == key_N):
                sparsity_dict[key] = hamming_distance(Theta, Z, t=t_rounding)

    return sparsity_dict


def sparsity_benchmark(df=pd.DataFrame(), upper_bound=float, lower_bound=float, lambda_filter=float):
    for i in ['method', 'accuracy', 'p', 'N', 'hamming']:
        assert i in df.columns

    df = df[df["l1"] == lambda_filter]

    spar_df = df[(df["accuracy"] < upper_bound) & (df["accuracy"] > lower_bound)]

    names = dict()
    frames = dict()
    for p in spar_df["p"].unique():
        for method in spar_df["method"].unique():
            names[method] = spar_df[(spar_df["p"] == p) & (spar_df["method"] == method)]["hamming"].min()
        frame = pd.DataFrame(names.items(), columns=['method', 'min_hamming'])
        frame = frame.sort_values(by='min_hamming', ascending=True)
        frames[p] = frame.reset_index(drop=True)

    return frames


def save_dict(D=dict, name=str):
    name = os.getcwd() + '/data/synthetic/' + str(name) + '.pickle'
    with open(os.path.expanduser(name), 'wb') as handle:
        pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(dict_name=str):
    name = os.getcwd() + '/data/synthetic/' + str(dict_name) + '.pickle'
    with open(os.path.expanduser(name), 'rb') as handle:
        D = pickle.load(handle)
    return D


def numba_warmup(S):
    print("######## NUMBA WARMUP ###########")
    Omega_0 = np.eye(len(S))
    _, _ = ADMM_SGL(S, lambda1=0.1, Omega_0=Omega_0,
                             max_iter=5, tol=1e-10, rtol=1e-10, stopping_criterion='boyd')

    _ = block_SGL(S, lambda1=0.1, Omega_0=Omega_0,
                        max_iter=5, tol=1e-10, rtol=1e-10, stopping_criterion='boyd')
    print("##################################")
    return


def benchmarks_dataframe(times=dict, acc=dict, hamming=dict, it = dict):
    """
    Turn benchmark dictionaries into dataframes.
    :param times: dict
    Input dictionary where 'key' is the model and 'value' is its runtime.
    :param acc_dict: dict
    Input dictionary where 'key' is a model and 'value' is its corresponding accuracy given as:
    np.linalg.norm(Z - np.array(Z_i)) / np.linalg.norm(Z)
    where Z is our model solution and Z_i is the model.
    :param spars_dict: dict
    Input dictionary where 'key' is a model and 'value' is its corresponding Hamming distance to the oracle precision matrix.
    :return: Pandas.DataFrame()
    """
    assert len(times) == len(acc) == len(hamming)

    all_dict = dict()
    for key in times.keys():
        all_dict[key] = {'time': times[key], 'accuracy': acc[key], 'hamming': hamming[key], 'iter': it[key]}

    df = pd.DataFrame.from_dict(all_dict, orient='index')

    # split key into columns
    df['split'] = df.index.str.split('_')
    columns_names = ["method", "tol_str", "tol", "rtol_str", "rtol", "p_str", "p", "N_str", "N", "l1_str", "l1"]
    df[columns_names] = pd.DataFrame(df['split'].tolist(), index=df['split'].index)

    redundant_cols = ['split', "tol_str", "rtol_str", "p_str", "N_str", "l1_str"]
    df = df.drop(redundant_cols, axis=1)

    convert_dict = {'tol': float, 'rtol': float, "p": int, "N": int, "l1": float}
    df = df.astype(convert_dict)
    df = df.sort_values(['p', 'l1', 'method', 'time'])

    return df