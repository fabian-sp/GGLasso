import time
import numpy as np
import pandas as pd
import itertools

from tqdm import trange
from sklearn.covariance import GraphicalLasso as sk_GL
from sklearn import set_config
set_config(print_changed_only=False)

from gglasso.solver.single_admm_solver import ADMM_SGL
from gglasso.solver.single_admm_solver import block_SGL

from regain.covariance import GraphicalLasso as rg_GL


def benchmark_parameters(S_dict=dict, sk_tol_list=[0.5, 0.25, 0.1], enet_list=[0.5, 0.25, 0.1],
                         rg_tol_list=[1e-4, 1e-5, 1e-6], rg_rtol_list=[1e-3, 1e-4, 1e-5],
                         admm_tol_list=[1e-6, 1e-7, 1e-8], admm_rtol_list=[1e-5, 1e-6, 1e-7],
                         admm_stop=['boyd'], admm_method=['single', 'block']):
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

    init_shape_list = []
    for i in S_dict.values():
        init_shape_list.append(len(i))
    rg_params = {"tol": rg_tol_list, "rtol": rg_rtol_list, "init_shape": init_shape_list}

    # ADMM params
    admm_tol_list = admm_tol_list
    admm_rtol_list = admm_rtol_list
    admm_stop = admm_stop
    admm_method = admm_method
    admm_params = {"tol": admm_tol_list, "rtol": admm_rtol_list, "stop": admm_stop, "method": admm_method}

    print("\n Sklearn model parameters:", sk_params)
    print("\n Regain model parameters:", rg_params)
    print("\n ADMM model parameters:", admm_params)

    return sk_params, rg_params, admm_params


def models_to_dict(models=None, lambda1=0.1, max_iter=50000, sk_params=dict, rg_params=dict):
    if models is None:
        models = list(str)
    models_dict = dict()

    for model in models:

        if model == "regain":

            for key in rg_params.keys():
                if key == "tol":
                    tol_list = rg_params[key]
                elif key == "rtol":
                    rtol_list = rg_params[key]
                elif key == "init_shape":
                    init_shape_list = rg_params[key]

            for shape in init_shape_list:
                for tol, rtol in itertools.product(tol_list, rtol_list):
                    key = str(model) + "_tol_" + str(tol) + "_rtol_" + str(rtol) + "_p_" + str(shape)
                    models_dict[key] = rg_GL(alpha=lambda1, tol=tol, rtol=rtol, max_iter=max_iter,
                                             assume_centered=False, init=np.eye(shape))

        elif model == "sklearn":

            for key in sk_params.keys():
                if key == "tol":
                    tol_list = sk_params[key]
                elif key == "enet":
                    enet_list = sk_params[key]

            for tol, enet in itertools.product(tol_list, enet_list):
                key = str(model) + "_tol_" + str(tol) + "_enet_" + str(enet)
                models_dict[key] = sk_GL(alpha=lambda1, tol=tol, enet_tol=enet, max_iter=max_iter,
                                         assume_centered=False)

    return models_dict


def model_solution(model="sklearn", X=np.array([]), lambda1=0.01, max_iter=50000, tol=1e-10, rtol=1e-4, enet=0.001):
    time_list = []
    if model == "sklearn":

        start = time.time()
        Z = sk_GL(alpha=lambda1, max_iter=max_iter, tol=tol, enet_tol=enet, assume_centered=False).fit(X)
        info = sk_GL(alpha=lambda1, max_iter=max_iter, tol=tol, enet_tol=enet, assume_centered=False)
        end = time.time()

        time_list.append(end - start)

    elif model == "regain":

        start = time.time()
        Z = rg_GL(alpha=lambda1, max_iter=max_iter, tol=tol, rtol=rtol, assume_centered=False).fit(X)
        info = rg_GL(alpha=lambda1, max_iter=max_iter, tol=tol, rtol=rtol, assume_centered=False)
        end = time.time()

        time_list.append(end - start)

    print(model, info)

    return Z.precision_, time_list


def sklearn_time_benchmark(models=dict, X=np.array([]), Z=np.array([]), n_iter=10):
    cov_dict = dict()
    precision_dict = dict()
    time_dict = dict()
    accuracy_dict = dict()

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


def admm_time_benchmark(S=np.array([]), Omega_0=np.array([]), Z=np.array([]), lambda1=0.1, n_iter=int, max_iter=50000,
                        admm_params=dict):
    cov_dict = dict()
    precision_dict = dict()
    accuracy_dict = dict()
    time_dict = dict()

    tol_list = admm_params["tol"]
    rtol_list = admm_params["rtol"]
    method_list = admm_params["method"]
    stop_list = admm_params["stop"]

    for method in method_list:

        for tol in tol_list:

            i = 0
            for rtol in rtol_list:

                if i == 0:
                    Omega_0 = Omega_0
                else:
                    # to reduce the convergence time we use the starting point from previous iterations of rtol
                    Omega_0 = Z_i["Omega"]

                time_list = []
                key = method + "-" + str(stop_list[0]) + "_tol_" + str(tol) + "_rtol_" + str(rtol)

                for _ in trange(n_iter, desc=key, leave=True):
                    if method == "single":
                        start = time.time()
                        Z_i, info = ADMM_SGL(S, lambda1=lambda1, Omega_0=Omega_0, max_iter=max_iter, tol=tol, rtol=rtol,
                                             stopping_criterion=stop_list[0])
                        end = time.time()
                        time_list.append(end - start)

                    elif method == "block":
                        start = time.time()
                        Z_i = block_SGL(S, lambda1=lambda1, Omega_0=Omega_0, max_iter=max_iter, tol=tol, rtol=rtol,
                                        stopping_criterion=stop_list[0])
                        end = time.time()
                        time_list.append(end - start)

                # mean time in "n" iterations, the first iteration we skip because of numba init
                time_dict[key] = np.mean(time_list[1:])

                cov_dict["cov_" + key] = Z_i["Omega"]
                precision_dict["precision_" + key] = Z_i["Theta"]

                accuracy = np.linalg.norm(Z - np.array(Z_i["Theta"])) / np.linalg.norm(Z)
                accuracy_dict["accuracy_" + key] = accuracy

                i += 1

    return time_dict, accuracy_dict, precision_dict


def time_benchmark(X=list, S=list, lambda1=0.1, max_iter=50000, Z_model=str, sk_models=["sklearn", "regain"],
                   n_iter=int, sk_params=dict, rg_params=dict, admm_params=dict):
    assert Z_model in ('sklearn', 'regain')

    accuracy_dict = dict()
    precision_dict = dict()
    time_dict = dict()

    # Model solution Z
    Z, Z_time = model_solution(model=Z_model, X=X, lambda1=lambda1)

    # Sklearn and regain benchmarking
    models = models_to_dict(models=sk_models, lambda1=lambda1, max_iter=max_iter,
                            sk_params=sk_params, rg_params=rg_params)

    sk_time, sk_accuracy, Z_sk = sklearn_time_benchmark(models, X=X, Z=Z, n_iter=n_iter)

    # ADMM benchmarking
    Omega_0 = np.eye(len(S))

    admm_time, admm_accuracy, Z_admm = admm_time_benchmark(S=S, Omega_0=Omega_0, Z=Z, lambda1=lambda1,
                                                           n_iter=1 + n_iter,
                                                           admm_params=admm_params)

    # Join results in a single dictionary
    times = sk_time.copy()
    times.update(admm_time)
    for key, value in times.items():
        if "regain" in key.split("_"):
            time_dict[key + "_N_" + str(len(X))] = value
        else:
            time_dict[key + "_p_" + str(len(S)) + "_N_" + str(len(X))] = value

    accs = sk_accuracy.copy()
    accs.update(admm_accuracy)
    for key, value in accs.items():
        if "regain" in key.split("_"):
            accuracy_dict[key + "_N_" + str(len(X))] = value
        else:
            accuracy_dict[key + "_p_" + str(len(S)) + "_N_" + str(len(X))] = value

    precs = Z_sk.copy()
    precs.update(Z_admm)
    for key, value in precs.items():
        if "regain" in key.split("_"):
            precision_dict[key + "_N_" + str(len(X))] = value
        else:
            precision_dict[key + "_p_" + str(len(S)) + "_N_" + str(len(X))] = value

    return time_dict, accuracy_dict, precision_dict


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


def sk_scaling(X, model, k_iter=5, time_list=None):
    if time_list is None:
        time_list = []

    for _ in trange(k_iter, desc=str(model), leave=True):
        sk_start = time.time()
        Z_i = model.fit(X)
        sk_end = time.time()

        sk_time = sk_end - sk_start
        time_list.append(sk_time)

    if model.n_iter_ == model.max_iter:
        status = True
    else:
        status = False

    return time_list, status


def single_scaling(S, lambda1, Omega_0, max_iter, tol, rtol, n_iter, time_list=[]):
    for _ in trange(n_iter, leave=True):
        single_start = time.time()
        Z_i, info = ADMM_SGL(S, lambda1=lambda1, Omega_0=Omega_0, max_iter=max_iter,
                             tol=tol, rtol=rtol, stopping_criterion="boyd")
        single_end = time.time()

        single_time = single_end - single_start
        time_list.append(single_time)

    return time_list


def block_scaling(S, lambda1, Omega_0, max_iter, tol, rtol, n_iter, time_list=[]):
    for _ in trange(n_iter):
        block_start = time.time()
        Z_i = block_SGL(S, lambda1=lambda1, Omega_0=Omega_0, max_iter=max_iter,
                        tol=tol, rtol=rtol, stopping_criterion="boyd")
        block_end = time.time()

        block_time = block_end - block_start
        time_list.append(block_time)

    return time_list

