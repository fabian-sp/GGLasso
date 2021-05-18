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
from gglasso.helper.data_generation import group_power_network, sample_covariance_matrix

from gglasso.solver.single_admm_solver import ADMM_SGL
from gglasso.solver.single_admm_solver import block_SGL, get_connected_components

from regain.covariance import GraphicalLasso as rg_GL

from sklearn.covariance import GraphicalLasso as sk_GL
from sklearn import set_config

set_config(print_changed_only=False)


def network_generation(p=int, N=int, K=1, M=int):
    """
    Generates a law-power network with number of connected components bigger than 1.
    :param p: int
    Number of features.
    :param N: int
    Number of samples.
    :param K: int, default: ‘1’
    Number of instances with p features and N samples.
    :param M: int
    Number of subblock in each instance K.
    :return: S, X, Theta
    S - empirical covarince matrix.
    X - dual variable.
    Theta - true precision matrix.
    """
    Sigma, Theta = group_power_network(p, K=K, M=M)
    S, samples = sample_covariance_matrix(Sigma, N)

    S = S[0, :, :]
    Theta = Theta[0, :, :]
    X = samples[0, :, :].T

    return S, X, Theta


def benchmark_parameters(sk_tol_list=[0.5], enet_list=[0.5],
                         rg_tol_list=[1e-4, 1e-5, 1e-6], rg_rtol_list=[1e-3, 1e-4, 1e-5],
                         gglasso_tol_list=[1e-6, 1e-7, 1e-8], gglasso_rtol_list=[1e-5, 1e-6, 1e-7],
                         gglasso_stop=['boyd'], gglasso_method=['single', 'block'],
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
    sk_tol_list = sk_tol_list
    enet_list = enet_list
    sk_params = {"tol": sk_tol_list, "enet": enet_list}

    # Regain params
    rg_tol_list = rg_tol_list
    rg_rtol_list = rg_rtol_list
    rg_params = {"tol": rg_tol_list, "rtol": rg_rtol_list}

    # ADMM params
    gglasso_tol_list = gglasso_tol_list
    gglasso_rtol_list = gglasso_rtol_list
    gglasso_stop = gglasso_stop
    gglasso_method = gglasso_method
    gglasso_params = {"tol": gglasso_tol_list, "rtol": gglasso_rtol_list, "stop": gglasso_stop,
                      "method": gglasso_method}

    print("\n Sklearn model parameters:", sk_params)
    print("\n Regain model parameters:", rg_params)
    print("\n ADMM model parameters:", gglasso_params)
    print("\n Lambda list:", lambda_list)

    return sk_params, rg_params, gglasso_params, lambda_list


def model_solution(model="sklearn", X=np.array([]), lambda1=float, max_iter=50000, tol=1e-10, rtol=1e-4, enet=0.001):
    time_list = []
    if model == "sklearn":

        start = time.perf_counter()
        Z = sk_GL(alpha=lambda1, max_iter=max_iter, tol=tol, enet_tol=enet, assume_centered=False).fit(X)
        info = sk_GL(alpha=lambda1, max_iter=max_iter, tol=tol, enet_tol=enet, assume_centered=False, verbose=True)
        end = time.perf_counter()

        time_list.append(end - start)

    elif model == "regain":

        start = time.perf_counter()
        Z = rg_GL(alpha=lambda1, init=np.eye(X.shape[1]), max_iter=max_iter, tol=tol, rtol=rtol,
                  assume_centered=False).fit(X)
        info = rg_GL(alpha=lambda1, init=np.eye(X.shape[1]), max_iter=max_iter, tol=tol, rtol=rtol,
                     assume_centered=False, verbose=True)
        end = time.perf_counter()

        time_list.append(end - start)

    print(model, info)

    return Z.precision_, time_list, info


def benchmarks_dataframe(times=dict, acc_dict=dict, spars_dict=dict):
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
    assert len(times) == len(acc_dict) == len(spars_dict)

    # The time measured during the grid search of best hyperparameters for the models
    df = pd.DataFrame(data={'name': list(times.keys()),
                            'time': list(times.values()),
                            "accuracy": list(acc_dict.values()),
                            "hamming": list(spars_dict.values())})

    df['split'] = df['name'].str.split('_')
    columns_names = ["method", "tol_str", "tol", "rtol_str", "rtol", "p_str", "p", "l1_str", "l1"]
    df[columns_names] = pd.DataFrame(df['split'].tolist(), index=df['split'].index)

    redundant_cols = ['split', "tol_str", "rtol_str", "p_str", "l1_str"]
    df = df.drop(redundant_cols, axis=1)

    convert_dict = {'tol': float, 'rtol': float, "p": int, "l1": float}
    df = df.astype(convert_dict)
    df['method_str'] = df['method'].str.replace('\d+', '')
    df = df.sort_values(by=['time'])

    return df


def best_time_dataframe(best_time=dict):
    """
    Turn dictionaries of the model best running times into dataframes.
    :param best_time: dict
    Input dictionary where 'key' is the selected best model and 'value' is its runtime.
    :return: Pandas.DataFrame()
    """
    # The time measured during the scalability benchmark
    time_df = pd.DataFrame.from_dict(best_time, orient='index', columns=['time'])
    time_df.reset_index(level=0, inplace=True)

    time_df['split'] = time_df['index'].str.split('_')
    columns_names = ["method", "p_str", "p", "N_str", "N"]
    time_df[columns_names] = pd.DataFrame(time_df['split'].tolist(), index=time_df['split'].index)

    redundant_cols = ["p_str", "N_str"]
    time_df = time_df.drop(redundant_cols, axis=1)

    convert_dict = {"p": int, "N": int}
    time_df = time_df.astype(convert_dict)
    time_df.sort_values(by=['time'])

    return time_df


def drop_acc_duplicates(df):
    """
    Drop duplicates of the models showing the same accuracy.
    :param df: pd.DataFrame()
    :return: pd.DataFrame()
    """
    assert 'method' in df.columns
    assert 'accuracy' in df.columns

    unique_acc_df = df[:1]
    for method in df.method.unique():
        filtered = df[df['method'] == method]
        filtered = filtered.drop_duplicates(subset='accuracy', keep='first')
        unique_acc_df = pd.concat([filtered, unique_acc_df])

    return unique_acc_df[:-1]


def dict_shape(dict_=dict):
    shape_list = []

    for i in dict_.values():
        shape_list.append(np.array(i).shape)
    return shape_list


def hamming_dict(Theta_dict=dict, Z_dict=dict, t_rounding=float):
    """
    Calculate Hamming distance between model solution Z and given solution Theta
    with a specified rounding accuracy t_rounding.
    :param Theta_dict: dict
    :param Z_dict: dict
    :param t_rounding: 1e-10, float
    :return: dict
    """
    sparsity_dict = dict()

    for Theta in Theta_dict.values():

        for key, Z in Z_dict.items():
            if Theta.shape == Z.shape:
                sparsity_dict[key] = hamming_distance(Theta, Z, t=t_rounding)

    return sparsity_dict


def sparsity_benchmark(df=pd.DataFrame()):
    for i in ['method', 'accuracy', 'p', 'hamming']:
        assert i in df.columns

    spar_df = df[(df["accuracy"] < 0.01) & (df["accuracy"] > 0.0001)]

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


def numba_warmup(S=np.array([]), Omega_0=np.array([]), l1=0.5, tol=1e-1, rtol=1e-1, stopping_criterion='boyd',
                 n_iter=5, max_iter=100):
    for i in range(0, n_iter):
        Z_s, info = ADMM_SGL(S, lambda1=l1, Omega_0=Omega_0, Theta_0=Omega_0, X_0=Omega_0,
                             max_iter=max_iter, tol=tol, rtol=rtol, stopping_criterion=stopping_criterion)
        Z_b = block_SGL(S, lambda1=l1, Omega_0=Omega_0, Theta_0=Omega_0, X_0=Omega_0,
                        max_iter=max_iter, tol=tol, rtol=rtol, stopping_criterion='boyd')
    result = "Numba has been succesfully initiated"
    return result
