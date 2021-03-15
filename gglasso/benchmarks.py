import sys
import time
import numpy as np
import pandas as pd
import itertools
import plotly.express as px

from sklearn.covariance import GraphicalLasso as sk_GL
from sklearn.covariance import empirical_covariance

# sys.path.append('..')
from gglasso.solver.single_admm_solver import ADMM_SGL
from gglasso.solver.single_admm_solver import block_SGL
from gglasso.helper.data_generation import time_varying_power_network, group_power_network, sample_covariance_matrix
from gglasso.helper.model_selection import single_grid_search

from regain.covariance import GraphicalLasso as rg_GL

def models_to_dict(models=None, lambda1=0.1, tol_list=list, rtol_list=list, enet_list=list, max_iter=50000):
    if models is None:
        models = list(str)
    models_dict = dict()

    for model in models:
        for tol, rtol, enet_tol in itertools.product(tol_list, rtol_list, enet_list):

            if model == "regain":
                models_dict[str(model) + "_tol_" + str(tol) + "_rtol_" + str(rtol)] \
                    = rg_GL(alpha=lambda1, tol=tol, rtol=rtol, max_iter=max_iter)

            elif model == "sklearn":
                models_dict[str(model) + "_tol_" + str(tol) + "_enet_" + str(enet_tol)] \
                    = sk_GL(alpha=lambda1, tol=tol, enet_tol=enet_tol, max_iter=max_iter)

    return models_dict


def sklearn_time_benchmark(models=dict, X=np.array([]), Z=np.array([]),
                   cov_dict=dict(), precision_dict=dict(),
                   time_dict=dict(), accuracy_dict=dict()):
    for model, model_instant in models.items():
        start = time.time()
        Z_i = model_instant.fit(X)
        end = time.time()

        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        time_dict[model] = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

        cov_dict["cov_" + model] = Z_i.covariance_
        precision_dict["precision_" + model] = Z_i.precision_

    for model, Z_i in precision_dict.items():
        accuracy = np.linalg.norm(Z - np.array(Z_i)) / np.linalg.norm(Z)
        accuracy_dict[model] = accuracy

    return time_dict, accuracy_dict


def admm_time_benchmark(S=np.array([]), Omega_0=np.array([]), Z=np.array([]), lambda1=float, max_iter=50000,
                        method_list=list, stop_list=list, tol_list=list, rtol_list=list,
                        cov_dict=dict(), precision_dict=dict(),
                        time_dict=dict(), accuracy_dict=dict()):
    for method in method_list:
        for tol, rtol, stop in itertools.product(tol_list, rtol_list, stop_list):
            if method == "single":
                start = time.time()
                Z_i, info = ADMM_SGL(S, lambda1=lambda1, Omega_0=Omega_0, max_iter=max_iter,
                                     tol=tol, rtol=rtol, stopping_criterion=stop)
                end = time.time()

            elif method == "block":
                start = time.time()
                Z_i = block_SGL(S, lambda1=lambda1, Omega_0=Omega_0, max_iter=max_iter,
                                tol=tol, rtol=rtol, stopping_criterion=stop)
                end = time.time()

            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)
            key = method + "-" + str(stop) + "_tol_" + str(tol) + "_rtol_" + str(rtol)
            time_dict[key] = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

            cov_dict["cov_" + key] = Z_i["Omega"]
            precision_dict["precision_" + key] = Z_i["Theta"]

            accuracy = np.linalg.norm(Z - np.array(Z_i["Theta"])) / np.linalg.norm(Z)
            accuracy_dict["accuracy_" + key] = accuracy

    return time_dict, accuracy_dict
