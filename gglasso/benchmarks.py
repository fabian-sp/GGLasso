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
                    = rg_GL(alpha=lambda1, tol=tol, rtol=rtol, max_iter=max_iter, assume_centered=True)

            elif model == "sklearn":
                models_dict[str(model) + "_tol_" + str(tol) + "_enet_" + str(enet_tol)] \
                    = sk_GL(alpha=lambda1, tol=tol, enet_tol=enet_tol, max_iter=max_iter, assume_centered=True)

    return models_dict


def sklearn_time_benchmark(models=dict, X=np.array([]), Z=np.array([]), n_iter=int,
                           cov_dict=dict(), precision_dict=dict(),
                           time_dict=dict(), accuracy_dict=dict()):
    for model, model_instant in models.items():

        time_list = []
        for _ in np.arange(n_iter):
            start = time.time()
            Z_i = model_instant.fit(X)
            end = time.time()

            time_list.append(end - start)

        time_dict[model] = np.mean(time_list)

        cov_dict["cov_" + model] = Z_i.covariance_
        precision_dict["_precision_" + model] = Z_i.precision_

    for model, Z_i in precision_dict.items():
        accuracy = np.linalg.norm(Z - np.array(Z_i)) / np.linalg.norm(Z)
        accuracy_dict[model] = accuracy

    return time_dict, accuracy_dict


def admm_time_benchmark(S=np.array([]), Omega_0=np.array([]), Z=np.array([]), lambda1=float, n_iter=int, max_iter=50000,
                        method_list=list, stop_list=list, tol_list=list, rtol_list=list,
                        cov_dict=dict(), precision_dict=dict(),
                        time_dict=dict(), accuracy_dict=dict()):
    for method in method_list:
        for tol, rtol, stop in itertools.product(tol_list, rtol_list, stop_list):

            time_list = []
            for _ in np.arange(n_iter):
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

            key = "_" + method + "-" + str(stop) + "_tol_" + str(tol) + "_rtol_" + str(rtol)
            # mean time in "n" iterations
            time_dict[key] = np.mean(time_list)

            cov_dict["cov_" + key] = Z_i["Omega"]
            precision_dict["precision_" + key] = Z_i["Theta"]

            accuracy = np.linalg.norm(Z - np.array(Z_i["Theta"])) / np.linalg.norm(Z)
            accuracy_dict["accuracy_" + key] = accuracy

    return time_dict, accuracy_dict

# p = 100
# N = 200
# Omega_0 = np.eye(p)
# lambda1 = 0.01
# tol_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
# enet_list = [1]
#
# Sigma, Theta = group_power_network(p, K=5, M=2)  # Theta is true precision matrix
#
# S, samples = sample_covariance_matrix(Sigma, N)
#
# S = S[0, :, :]
# Theta = Theta[0, :, :]  # true precision matrix
#
# Z = rg_GL(alpha=lambda1, max_iter=50000, tol=1e-10, assume_centered=True).fit(samples[0, :, :].T)
# Z = Z.precision_
#
# admm_time, admm_accuracy = admm_time_benchmark(S=S, Omega_0=Omega_0, Z=Z, lambda1=lambda1,
#                                                method_list=["single", "block"],
#                                                stop_list=['boyd', "kkt"],
#                                                tol_list=[0.1],
#                                                rtol_list=[0.1],
#                                                n_iter=100)
# #
# #
#
# models = models_to_dict(models=["sklearn", "regain"], lambda1=lambda1, tol_list=[0.1], rtol_list=[0.1],
#                         enet_list=[1])
#
# sk_time, sk_accuracy = sklearn_time_benchmark(models=models, X=samples[0, :, :].T, Z=Z, n_iter=10)



# df = pd.DataFrame(data={'name': admm_time.keys(), 'time': admm_time.values(), "accuracy": admm_accuracy.values()})
# df.head()
#
# df['split'] = df['name'].str.split('_')
# df[["iter", "method", "tol_str", "tol", "rtol_str", "rtol"]] = pd.DataFrame(df['split'].tolist(), index=df['split'].index)
# df = df.drop(["name", 'split', "tol_str", "rtol_str"], axis=1)
# df.head()
#
# # df = pd.DataFrame(admm_time.items())
# # df[['iter', 'method', 'tol', 'rtol']] = pd.DataFrame(df[0].tolist(), index=df.index)
# # df = df.rename(columns={0: 'model', 1: 'time'})
#
#
#
# empty = []
# for i in np.arange(10):
#     start = time.time()
#     Z_i, info = ADMM_SGL(S, lambda1=lambda1, Omega_0=Omega_0, max_iter=100,
#                          tol=0.01, rtol=0.01, stopping_criterion='boyd')
#     end = time.time()
#     empty.append(end - start)
#     time.sleep(1)
# empty
#
# empty = []
# for i in np.arange(10):
#     start = time.time()
#     Z_i = sk_GL(alpha=lambda1, tol=0.01, enet_tol=0.01, max_iter=100, assume_centered=True).fit(samples[0, :, :].T)
#     end = time.time()
#     empty.append(end-start)
# empty
#
#
#
#
# start = time.time()
# Z = rg_GL(alpha=lambda1, max_iter=50000, tol=1e-10).fit(samples[0, :, :].T)
# end = time.time()
#
# hours, rem = divmod(end - start, 3600)
# minutes, seconds = divmod(rem, 60)
# Z_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
# Z = Z.precision_
# print(Z_time)
#
# admm_time, admm_accuracy = admm_time_benchmark(S=S, Omega_0=Omega_0, Z=Z, lambda1=lambda1,
#                                                method_list=["single", "block"],
#                                                stop_list=['boyd', 'kkt'],
#                                                tol_list=[0.1, 0.01],
#                                                rtol_list=[0.1, 0.01])
#
# len(admm_time)
#
# # for letter in 'Python':     # First Example
# #    if letter == 'P':
# #       continue
# #    print('Current Letter :', letter)
