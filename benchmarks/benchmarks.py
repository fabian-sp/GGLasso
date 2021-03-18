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

from regain.covariance import GraphicalLasso as rg_GL


def models_to_dict(models=None, lambda1=float, tol_list=list, rtol_list=list, enet_list=list, max_iter=50000):
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
        accuracy_dict[model] = accuracy

    return time_dict, accuracy_dict, precision_dict


def admm_time_benchmark(S=np.array([]), Omega_0=np.array([]), Z=np.array([]), lambda1=float, n_iter=10, max_iter=50000,
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

            # mean time in "n" iterations
            time_dict[key] = np.mean(time_list)

            cov_dict["cov_" + key] = Z_i["Omega"]
            precision_dict["precision_" + key] = Z_i["Theta"]

            accuracy = np.linalg.norm(Z - np.array(Z_i["Theta"])) / np.linalg.norm(Z)
            accuracy_dict["accuracy_" + key] = accuracy

    return time_dict, accuracy_dict, precision_dict


def model_solution(model=str, X=np.array([]), lambda1=float, max_iter=50000, tol=float, rtol=float, enet=float):

    if model == "sklearn":

        start = time.time()
        Z = sk_GL(alpha=lambda1, max_iter=max_iter, tol=tol, enet_tol=enet, assume_centered=True).fit(X)
        info = sk_GL(alpha=lambda1, max_iter=max_iter, tol=tol, enet_tol=enet, assume_centered=True)
        end = time.time()

    elif model == "regain":

        start = time.time()
        Z = rg_GL(alpha=lambda1, max_iter=max_iter, tol=tol, rtol=rtol, assume_centered=True).fit(X)
        info = rg_GL(alpha=lambda1, max_iter=max_iter, tol=tol, enet_tol=enet, assume_centered=True)
        end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    Z_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

    print(model, info)

    return Z.precision_, Z_time


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""

    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten()  # all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative
    array += 1e-20  # values cannot be 0
    array = np.sort(array)  # values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements

    gini_coef = (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))  # Gini coefficient

    return gini_coef