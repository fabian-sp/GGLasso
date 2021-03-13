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

# def sampling(N, p, K, M):
#     Sigma, Theta = group_power_network(p, K=K, M=M)  # Theta is true precision matrix
#     S, samples = sample_covariance_matrix(Sigma, N)
#
#     S = S[0, :, :]
#     Theta = Theta[0, :, :]  # true precision matrix
#     return S, Theta

p = 100
N = 200
Sigma, Theta = group_power_network(p, K=5, M=2)  # Theta is true precision matrix

S, samples = sample_covariance_matrix(Sigma, N)

S = S[0, :, :]
Theta = Theta[0, :, :]  # true precision matrix

lambda1 = 0.01
Omega_0 = np.eye(p)

model_1 = rg_GL(alpha=0.01, max_iter=100, tol=1e-4).fit(samples[0, :, :].T)
model_2 = sk_GL(alpha=0.01, max_iter=100, tol=0.01).fit(samples[0, :, :].T)
sol, info = ADMM_SGL(S, lambda1, Omega_0, max_iter=100, tol=1e-4, rtol=1e-4,
                     stopping_criterion="boyd",
                     verbose=False, latent=False)
type(model_2.precision_)

model = sk_GL()
model(alpha=0.01).fit(samples[0, :, :].T)

dict_models = {"sklearn": sk_GL(), "regain": rg_GL()}
for model, model_instant in dict_models.items():
    print(model, model_instant)

dict_models = dict()
for tol in [0.01, 0.02]:
    for rtol in [0.01, 0.02]:
        dict_models['regain_' + "tol_" + str(tol) + "_rtol_" + str(rtol)] = rg_GL(alpha=lambda1, max_iter=100, tol=tol,
                                                                                  rtol=rtol)
    for enet_tol in [1, 0.01]:
        dict_models['sklearn' + "tol_" + str(tol) + "_enet_tol_" + str(rtol)] = sk_GL(alpha=lambda1, max_iter=100,
                                                                                      tol=tol,
                                                                                      enet_tol=enet_tol)


def models_to_dict(models=None, tol_list=list, rtol_list=list, enet_list=list, max_iter=int):
    if models is None:
        models = list(str)
    models_dict = dict()

    for model in models:
        for tol, rtol, enet_tol in itertools.product(tol_list, rtol_list, enet_list):

            if model == "regain":
                models_dict[str(model) + "_tol_" + str(tol) + "_rtol_" + str(rtol)] \
                    = rg_GL(alpha=lambda1, max_iter=max_iter, tol=tol, rtol=rtol)

            elif model == "sklearn":
                models_dict[str(model) + "_tol_" + str(tol) + "_enet_tol_" + str(enet_tol)] \
                    = sk_GL(alpha=lambda1, max_iter=max_iter, tol=tol, enet_tol=enet_tol)

            # if model == "admm":

    return models_dict


def time_benchmark(models=dict, Z=np.array([]), cov_dict=dict(), precision_dict=dict(),
                   time_dict=dict(), accuracy_dict=dict()):

    for model, model_instant in models.items():
        start = time.time()
        Z_n = model_instant.fit(samples[0, :, :].T)
        end = time.time()

        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        time_dict[model] = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

        cov_dict[model] = Z_n.covariance_
        precision_dict[model] = Z_n.precision_

    for model, Z_n in precision_dict.items():
        accuracy = np.linalg.norm(Z - np.array(Z_n)) / np.linalg.norm(Z)
        accuracy_dict[model] = accuracy

    return accuracy_dict


b = time_benchmark(models=a, Z=model_1.precision_)

len(b)

str_tol = [str(x) for x in tol_list]
method_list = ["sklearn"] * len(accuracy_list)
df_sk = pd.DataFrame(data={'time': time_list,
                           'distance': accuracy_list,
                           'method': method_list,
                           'tol_rate': tol_list,
                           'str_tol': str_tol})
