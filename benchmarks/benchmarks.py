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


def network_generation(p=int, N=int, K=1, M=int, S_dict=dict(), Theta_dict=dict(), X_dict=dict()):
    Sigma, Theta = group_power_network(p, K=K, M=M)  # Theta is true precision matrix
    S, samples = sample_covariance_matrix(Sigma, N)

    S = S[0, :, :]
    Theta = Theta[0, :, :]  # true precision matrix
    X = samples[0, :, :].T

    return S, X, Theta


def dict_shape(dict_=dict):
    shape_list = []

    for i in dict_.values():
        shape_list.append(np.array(i).shape)
    return shape_list


def benchmark_parameters(sk_tol_list=[0.5, 0.25, 0.1], enet_list=[0.5, 0.25, 0.1],
                         rg_tol_list=[1e-4, 1e-5, 1e-6], rg_rtol_list=[1e-3, 1e-4, 1e-5],
                         admm_tol_list=[1e-6, 1e-7, 1e-8], admm_rtol_list=[1e-5, 1e-6, 1e-7],
                         admm_stop=['boyd'], admm_method=['single', 'block']):
    # Sklearn params
    sk_tol_list = sk_tol_list
    enet_list = enet_list
    sk_params = {"tol": sk_tol_list, "enet": enet_list}

    # Regain params
    rg_tol_list = rg_tol_list
    rg_rtol_list = rg_rtol_list
    rg_params = {"tol": rg_tol_list, "rtol": rg_rtol_list}

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

            for tol, rtol in itertools.product(tol_list, rtol_list):
                key = str(model) + "_tol_" + str(tol) + "_rtol_" + str(rtol)
                models_dict[key] = rg_GL(alpha=lambda1, tol=tol, rtol=rtol, max_iter=max_iter, assume_centered=False)

        elif model == "sklearn":

            for key in sk_params.keys():
                if key == "tol":
                    tol_list = sk_params[key]
                elif key == "enet":
                    enet_list = sk_params[key]

            for tol, enet in itertools.product(tol_list, enet_list):
                key = str(model) + "_tol_" + str(tol) + "_enet_" + str(enet)
                models_dict[key] = sk_GL(alpha=lambda1, tol=tol, enet_tol=enet, max_iter=max_iter, assume_centered=False)

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
        info = rg_GL(alpha=lambda1, max_iter=max_iter, tol=tol, enet_tol=enet, assume_centered=False)
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
        for tol, rtol, stop in itertools.product(tol_list, rtol_list, stop_list):

            time_list = []
            key = method + "-" + str(stop) + "_tol_" + str(tol) + "_rtol_" + str(rtol)

            for _ in trange(n_iter, desc=key, leave=True):
                if method == "single":
                    start = time.time()
                    Z_i, info = ADMM_SGL(S, lambda1=lambda1, Omega_0=Omega_0, max_iter=max_iter, tol=tol, rtol=rtol,
                                         stopping_criterion=stop)
                    end = time.time()
                    time_list.append(end - start)

                elif method == "block":
                    start = time.time()
                    Z_i = block_SGL(S, lambda1=lambda1, Omega_0=Omega_0, max_iter=max_iter, tol=tol, rtol=rtol,
                                    stopping_criterion=stop)
                    end = time.time()
                    time_list.append(end - start)

            # mean time in "n" iterations, the first iteration we skip because of numba init
            time_dict[key] = np.mean(time_list[1:])

            cov_dict["cov_" + key] = Z_i["Omega"]
            precision_dict["precision_" + key] = Z_i["Theta"]

            accuracy = np.linalg.norm(Z - np.array(Z_i["Theta"])) / np.linalg.norm(Z)
            accuracy_dict["accuracy_" + key] = accuracy

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

    admm_time, admm_accuracy, Z_admm = admm_time_benchmark(S=S, Omega_0=Omega_0, Z=Z, lambda1=lambda1, n_iter=1+n_iter,
                                                           admm_params=admm_params)

    # Join results in a single dictionary
    times = sk_time.copy()
    times.update(admm_time)
    for key, value in times.items():
        time_dict[key + "_p_" + str(len(S)) + "_N_" + str(len(X))] = value

    accs = sk_accuracy.copy()
    accs.update(admm_accuracy)
    for key, value in accs.items():
        accuracy_dict[key + "_p_" + str(len(S)) + "_N_" + str(len(X))] = value

    precs = Z_sk.copy()
    precs.update(Z_admm)
    for key, value in precs.items():
        precision_dict[key + "_p_" + str(len(S)) + "_N_" + str(len(X))] = value

    return time_dict, accuracy_dict, precision_dict


def hamming_distance(X, Z, t=1e-10):
    A = adjacency_matrix(X, t=t)
    B = adjacency_matrix(Z, t=t)
    return (A + B == 1).sum()


def hamming_dict(Theta_dict=dict, Z_dict=dict, t_rounding=float):
    sparsity_dict = dict()

    for Theta in Theta_dict.values():

        for key, Z in Z_dict.items():
            if Theta.shape == Z.shape:
                sparsity_dict[key] = hamming_distance(Theta, Z, t=t_rounding)

    return sparsity_dict


def dict_to_dataframe(times=dict, acc_dict=dict, spars_dict=dict):
    assert len(times) == len(acc_dict) == len(spars_dict)

    df = pd.DataFrame(data={'name': list(times.keys()),
                            'time': list(times.values()),
                            "accuracy": list(acc_dict.values()),
                            "hamming": list(spars_dict.values())})

    df['split'] = df['name'].str.split('_')

    columns_names = ["method", "tol_str", "tol", "rtol_str", "rtol", "p_str", "p", "N_str", "N"]
    df[columns_names] = pd.DataFrame(df['split'].tolist(), index=df['split'].index)

    redundant_cols = ['split', "tol_str", "rtol_str", "p_str", "N_str"]
    df = df.drop(redundant_cols, axis=1)

    convert_dict = {'tol': float, 'rtol': float, "p": int, "N": int}
    df = df.astype(convert_dict)

    df = df.sort_values(by=['time'])

    return df


def drop_duplicates(df):
    assert 'method' in df.columns
    assert 'accuracy' in df.columns

    new_df = df[:1]
    for method in df.method.unique():
        filtered = df[df['method'] == method]
        filtered = filtered.drop_duplicates(subset='accuracy', keep='first')
        new_df = pd.concat([filtered, new_df])

    return new_df[:-1]


def plot_log_distance(df=pd.DataFrame(), upper_bound=float, lower_bound=float):
    fig = px.scatter(df[(df["accuracy"] < upper_bound) & (df["accuracy"] > lower_bound)],
                     x="time", y="accuracy", text="name", color="method",
                     log_y=True, facet_col='p', facet_col_wrap=3,
                     labels={
                         "time": "Time, s",
                         "accuracy": "Log_distance",
                         "method": "method"
                     },
                     template="plotly_white",
                     title="Log-distance between Z and Z' with respect to ADMM convergence rates")

    fig.update_traces(mode='markers+lines', marker_line_width=1, marker_size=10)
    fig.update_xaxes(matches=None)
    fig.update_yaxes(exponentformat="power")

    return fig


def sparsity_benchmark(df):
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
