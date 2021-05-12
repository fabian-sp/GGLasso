import numpy as np
import itertools
import time

from tqdm import trange

from gglasso.solver.single_admm_solver import ADMM_SGL
from gglasso.solver.single_admm_solver import block_SGL

from benchmarks.utils import benchmark_parameters, save_dict, load_dict


def gglasso_time(S=np.array([]), Omega_0=np.array([]), Z=dict, lambda_list=list, n_iter=int, max_iter=50000,
                 gglasso_params=dict):
    cov_dict = dict()
    precision_dict = dict()
    accuracy_dict = dict()
    time_dict = dict()

    tol_list = gglasso_params["tol"]
    rtol_list = gglasso_params["rtol"]
    method_list = gglasso_params["method"]
    stop_list = gglasso_params["stop"]

    for method, tol, rtol in itertools.product(method_list, tol_list, rtol_list):

        i = 0
        t = dict()  # dictionary for keeping the time from the previous iteration
        for l1 in lambda_list:

            if i == 0:
                Omega_0 = Omega_0
                Theta_0 = Omega_0.copy()
                X_0 = np.zeros((S.shape[0], S.shape[0]))
                time_list = np.array([0])
            else:
                # to reduce the convergence time we use the results from previous iterations
                Omega_0 = Z_i["Omega"]
                Theta_0 = Z_i["Theta"]
                X_0 = Z_i["X"]
                time_list = t[i - 1]

            pars = "_tol_" + str(tol) + "_rtol_" + str(rtol) + "_p_" + str(S.shape[0]) + "_l1_" + str(l1)
            key = method + "-" + str(stop_list[0]) + pars

            for _ in trange(n_iter, desc=key, leave=True):
                if method == "single":
                    start = time.perf_counter()
                    Z_i, info = ADMM_SGL(S, lambda1=l1,
                                         Omega_0=Omega_0, Theta_0=Theta_0, X_0=X_0,
                                         max_iter=max_iter, tol=tol, rtol=rtol,
                                         stopping_criterion=stop_list[0])
                    end = time.perf_counter()
                    time_list = np.append(time_list, end - start)

                elif method == "block":
                    start = time.perf_counter()
                    Z_i = block_SGL(S, lambda1=l1,
                                    Omega_0=Omega_0, Theta_0=Theta_0, X_0=X_0,
                                    max_iter=max_iter, tol=tol, rtol=rtol,
                                    stopping_criterion=stop_list[0])
                    end = time.perf_counter()
                    time_list = np.append(time_list, end - start)
                    print("{0}: {1} connected components.".format(key, Z_i["numC"]))

            if method == "block":
                # show number of connected components
                key = str(Z_i["numC"]) + key

            # mean time in "n" iterations, the first iteration we skip because of numba init
            time_dict[key] = float(time_list[-n_iter - 1: -n_iter] + np.mean(time_list[-n_iter:]))
            t[i] = time_list[-n_iter - 1: -n_iter] + np.mean(time_list[-n_iter:])

            cov_dict["cov_" + key] = Z_i["Omega"]
            precision_dict["precision_" + key] = Z_i["Theta"]

            model_key = "p_" + str(S.shape[0]) + "_l1_" + str(l1)
            accuracy = np.linalg.norm(Z[model_key] - np.array(Z_i["Theta"])) / np.linalg.norm(Z[model_key])
            accuracy_dict["accuracy_" + key] = accuracy

            i += 1

    return time_dict, accuracy_dict, precision_dict


def run_gglasso(X_dict=dict, S_dict=dict, model_Z_dict=dict, lambda_list=list, n_iter=int, gglasso_params=dict):
    time_dict = dict()
    accuracy_dict = dict()
    Z_dict = dict()
    trace_dict = dict()

    for X, S in zip(list(X_dict.values()), list(S_dict.values())):
        Omega_0 = np.eye(len(S))
        gg_time, gg_accuracy, Z_gg = gglasso_time(S=S, Omega_0=Omega_0, Z=model_Z_dict, lambda_list=lambda_list,
                                                  n_iter=n_iter, gglasso_params=gglasso_params)
        time_dict.update(gg_time)
        accuracy_dict.update(gg_accuracy)
        Z_dict.update(Z_gg)

        for key, item in Z_dict.items():
            trace_dict.update({key: {"Z": item, "X": X, "S": S}})  # add time for each lambda

    return time_dict, accuracy_dict, trace_dict


# Main entry point
if __name__ == "__main__":
    _, _, gglasso_params, lambda_list = benchmark_parameters()
    S_dict = load_dict(dict_name="S_dict")
    X_dict = load_dict(dict_name="X_dict")
    Z_dict = load_dict(dict_name="Z_dict")

    time_dict, accuracy_dict, trace_dict = run_gglasso(X_dict=X_dict, S_dict=S_dict, model_Z_dict=Z_dict,
                                                       lambda_list=lambda_list, n_iter=1 + 1,
                                                       gglasso_params=gglasso_params)

    save_dict(D=time_dict, name="gglasso_time_dict")
    save_dict(D=accuracy_dict, name="gglasso_acc_dict")
    save_dict(D=trace_dict, name="gglasso_trace_dict")
