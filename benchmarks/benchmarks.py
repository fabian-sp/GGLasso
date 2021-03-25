import sys
import time
import numpy as np
import pandas as pd
import itertools
import plotly.express as px
from tqdm import trange

from sklearn.covariance import GraphicalLasso as sk_GL
from sklearn.covariance import empirical_covariance
from sklearn import set_config

set_config(print_changed_only=False)

# sys.path.append('..')
from gglasso.solver.single_admm_solver import ADMM_SGL
from gglasso.solver.single_admm_solver import block_SGL
from gglasso.helper.data_generation import time_varying_power_network, group_power_network, sample_covariance_matrix
from gglasso.helper.model_selection import single_grid_search
from gglasso.helper.basic_linalg import adjacency_matrix

from regain.covariance import GraphicalLasso as rg_GL


def network_generation(p_list=list, N_list=list, K=int, M=int, S_dict=dict(), Theta_dict=dict(), X_dict=dict()):
    for p, N in zip(p_list, N_list):
        Sigma, Theta = group_power_network(p, K=K, M=M)  # Theta is true precision matrix
        S, samples = sample_covariance_matrix(Sigma, N)

        S = S[0, :, :]
        Theta = Theta[0, :, :]  # true precision matrix
        X = samples[0, :, :].T

        S_dict[p, N] = S
        X_dict[p, N] = X
        Theta_dict[p, N] = Theta

    return S_dict, X_dict, Theta_dict


def models_to_dict(models=None, lambda1=0.01, tol_list=list, rtol_list=list, enet_list=list, max_iter=50000):
    if models is None:
        models = list(str)
    models_dict = dict()

    for model in models:
        for tol, rtol, enet_tol in itertools.product(tol_list, rtol_list, enet_list):

            if model == "regain":
                models_dict[str(model) + "_tol_" + str(tol) + "_rtol_" + str(rtol)] \
                    = rg_GL(alpha=lambda1, tol=tol, rtol=rtol, max_iter=max_iter,
                            assume_centered=True)

            elif model == "sklearn":
                models_dict[str(model) + "_tol_" + str(tol) + "_enet_" + str(enet_tol)] \
                    = sk_GL(alpha=lambda1, tol=tol, enet_tol=enet_tol, max_iter=max_iter,
                            assume_centered=True)

    return models_dict


def model_solution(model=str, X=np.array([]), lambda1=0.01, max_iter=50000, tol=float, rtol=float, enet=float):
    time_list = []
    if model == "sklearn":

        start = time.time()
        Z = sk_GL(alpha=lambda1, max_iter=max_iter, tol=tol, enet_tol=enet, assume_centered=True).fit(X)
        info = sk_GL(alpha=lambda1, max_iter=max_iter, tol=tol, enet_tol=enet, assume_centered=True)
        end = time.time()

        time_list.append(end - start)

    elif model == "regain":

        start = time.time()
        Z = rg_GL(alpha=lambda1, max_iter=max_iter, tol=tol, rtol=rtol, assume_centered=True).fit(X)
        info = rg_GL(alpha=lambda1, max_iter=max_iter, tol=tol, enet_tol=enet, assume_centered=True)
        end = time.time()

        time_list.append(end - start)

    print(model, info)

    return Z.precision_, time_list


def sklearn_time_benchmark(models=dict, X=np.array([]), Z=np.array([]), n_iter=10,
                           cov_dict=dict(), precision_dict=dict(),
                           time_dict=dict(), accuracy_dict=dict()):
    for model, model_instant in models.items():

        time_list = []
        for _ in trange(n_iter, desc=model, leave=True):
            start = time.time()
            Z_i = model_instant.fit(X)
            end = time.time()

            time_list.append(end - start)

        time_dict[model] = np.mean(time_list)

        cov_dict["cov_" + model] = Z_i.covariance_
        precision_dict["precision_" + model] = Z_i.precision_

    for model, Z_i in precision_dict.items():
        accuracy = np.linalg.norm(Z - np.array(Z_i)) / np.linalg.norm(Z)
        accuracy_dict["accuracy_not" + model] = accuracy

    return time_dict, accuracy_dict, precision_dict


def admm_time_benchmark(S=np.array([]), Omega_0=np.array([]), Z=np.array([]), lambda1=0.01, n_iter=10, max_iter=50000,
                        method_list=list, stop_list=list, tol_list=list, rtol_list=list,
                        cov_dict=dict(), precision_dict=dict(),
                        time_dict=dict(), accuracy_dict=dict()):
    for method in method_list:
        for tol, rtol, stop in itertools.product(tol_list, rtol_list, stop_list):

            time_list = []
            key = method + "-" + str(stop) + "_tol_" + str(tol) + "_rtol_" + str(rtol)

            for _ in trange(n_iter, desc=key, leave=True):
                if method == "single":
                    start = time.time()
                    Z_i, info = ADMM_SGL(S, lambda1=lambda1, Omega_0=Omega_0, max_iter=max_iter,
                                         tol=tol, rtol=rtol, stopping_criterion=stop)
                    end = time.time()
                    time_list.append(end - start)

                elif method == "block":
                    start = time.time()
                    Z_i = block_SGL(S, lambda1=lambda1, Omega_0=Omega_0, max_iter=max_iter,
                                    tol=tol, rtol=rtol, stopping_criterion=stop)
                    end = time.time()
                    time_list.append(end - start)

            # mean time in "n" iterations, the first iteration we skip because of numba init
            time_dict[key] = np.mean(time_list[1:])

            cov_dict["cov_" + key] = Z_i["Omega"]
            precision_dict["precision_" + key] = Z_i["Theta"]

            accuracy = np.linalg.norm(Z - np.array(Z_i["Theta"])) / np.linalg.norm(Z)
            accuracy_dict["accuracy_" + key] = accuracy

    return time_dict, accuracy_dict, precision_dict


def time_benchmark(X_dict=dict, S_dict=dict, lambda1=0.01, Z_model=str, Z_tol=1e-10, Z_rtol=1e-4, Z_enet=0.1,
                   sk_models=["sklearn", "regain"], admm_models=["single", "block"], admm_stop=['boyd'],
                   tol_list=list, rtol_list=list, enet_list=list, max_iter=50000,
                   t_dict=dict(), acc_dict=dict(), prec_dict=dict()):
    assert Z_model in ('sklearn', 'regain')

    for X, S in zip(X_dict.values(), S_dict.values()):

        # Sklearn and regain benchmarking
        Z, Z_time = model_solution(model=Z_model, X=X, lambda1=lambda1,
                                   tol=Z_tol, rtol=Z_rtol, enet=Z_enet,
                                   max_iter=max_iter)

        models = models_to_dict(models=sk_models, lambda1=lambda1, tol_list=tol_list,
                                rtol_list=rtol_list, enet_list=enet_list, max_iter=max_iter)

        sk_time, sk_accuracy, Z_sk = sklearn_time_benchmark(models, X=X, Z=Z, n_iter=5)

        # ADMM benchmarking
        Omega_0 = np.eye(len(S))

        admm_time, admm_accuracy, Z_admm = admm_time_benchmark(S=S, Omega_0=Omega_0, Z=Z, lambda1=lambda1,
                                                               method_list=admm_models, stop_list=admm_stop,
                                                               tol_list=tol_list, rtol_list=rtol_list, n_iter=1 + 5)

        # Join results in a single dictionary
        times = sk_time.copy()
        times.update(admm_time)
        for key, value in times.items():
            t_dict[key + "_p_" + str(len(S)) + "_N_" + str(len(X))] = value

        accs = sk_accuracy.copy()
        accs.update(admm_accuracy)
        for key, value in accs.items():
            acc_dict[key + "_p_" + str(len(S)) + "_N_" + str(len(X))] = value

        precs = Z_sk.copy()
        precs.update(Z_admm)
        for key, value in precs.items():
            prec_dict[key + "_p_" + str(len(S)) + "_N_" + str(len(X))] = value

    return t_dict, acc_dict, prec_dict


def hamming_distance(X, Z, t=1e-10):
    A = adjacency_matrix(X, t=t)
    B = adjacency_matrix(Z, t=t)
    return (A + B == 1).sum()


def sparsity_benchmark(Theta_dict=dict, Z_dict=dict, t_rounding=float, sparsity_dict=dict()):
    for Theta in Theta_dict.values():

        for key, Z in Z_dict.items():
            if Theta.shape == Z.shape:
                sparsity_dict[key] = hamming_distance(Theta, Z, t=t_rounding)

    return sparsity_dict


def dict_to_dataframe(times=dict, acc_dict=dict, spars_dict=dict):

    assert len(times) == len(acc_dict) == len(spars_dict)

    df = pd.DataFrame(data={'name': times.keys(),
                            'time': times.values(),
                            "accuracy": acc_dict.values(),
                            "hamming": spars_dict.values()})

    df['split'] = df['name'].str.split('_')

    columns_names = ["method", "tol_str", "tol", "rtol_str", "rtol", "p_str", "p", "N_str", "N"]
    df[columns_names] = pd.DataFrame(df['split'].tolist(), index=df['split'].index)

    redundant_cols = ['split', "tol_str", "rtol_str", "p_str", "N_str"]
    df = df.drop(redundant_cols, axis=1)

    return df


# def gini(array):
#     """Calculate the Gini coefficient of a numpy array."""
#
#     # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
#     # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
#     array = array.flatten()  # all values are treated equally, arrays must be 1d
#     if np.amin(array) < 0:
#         array -= np.amin(array)  # values cannot be negative
#     array += 1e-20  # values cannot be 0
#     array = np.sort(array)  # values must be sorted
#     index = np.arange(1, array.shape[0] + 1)  # index per array element
#     n = array.shape[0]  # number of array elements
#
#     gini_coef = (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))  # Gini coefficient
#
#     return gini_coef

# sparsity_sk = list(map(gini, Z_sk.values()))
# sparsity_admm =  list(map(gini, Z_admm.values()))
