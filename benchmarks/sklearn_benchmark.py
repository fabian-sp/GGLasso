import numpy as np
import itertools
import time

from tqdm import trange

from sklearn.covariance import GraphicalLasso as sk_GL
from sklearn import set_config

from benchmarks.utils import benchmark_parameters, save_dict, load_dict

set_config(print_changed_only=False)


def sklearn_time(X=np.array([]), Z=dict, sk_params=dict, lambda_list=list, n_iter=int, max_iter=50000):
    cov_dict = dict()
    precision_dict = dict()
    time_dict = dict()
    accuracy_dict = dict()

    tol_list = sk_params["tol"]
    enet_list = sk_params["enet"]

    for tol, enet, l1 in itertools.product(tol_list, enet_list, lambda_list):
        key = "sklearn" + "_tol_" + str(tol) + "_enet_" + str(enet) + "_p_" + str(X.shape[1]) + "_l1_" + str(l1)

        model = sk_GL(alpha=l1, tol=tol, enet_tol=enet, max_iter=max_iter, assume_centered=False, verbose=True)

        time_list = []

        for _ in trange(n_iter, desc=key, leave=True):
            start = time.perf_counter()
            Z_i = model.fit(X)
            end = time.perf_counter()

            time_list.append(end - start)

        time_dict[key] = np.mean(time_list)

        cov_dict["cov_" + key] = Z_i.covariance_
        precision_dict["precision_" + key] = Z_i.precision_

        model_key = "p_" + str(X.shape[1]) + "_l1_" + str(l1)
        accuracy = np.linalg.norm(Z[model_key] - np.array(Z_i.precision_)) / np.linalg.norm(Z[model_key])
        accuracy_dict["accuracy_" + key] = accuracy

    return time_dict, accuracy_dict, precision_dict


def run_sklearn(X_dict=dict, S_dict=dict, model_Z_dict=dict, lambda_list=list, n_iter=int, sklearn_params=dict):
    time_dict = dict()
    accuracy_dict = dict()
    Z_dict = dict()
    trace_dict = dict()

    for X, S in zip(list(X_dict.values()), list(S_dict.values())):
        sk_time, sk_accuracy, Z_sk = sklearn_time(X=X, Z=model_Z_dict, sk_params=sklearn_params,
                                                  lambda_list=lambda_list, n_iter=n_iter)

        time_dict.update(sk_time)
        accuracy_dict.update(sk_accuracy)
        Z_dict.update(Z_sk)

        for key, item in Z_dict.items():
            trace_dict.update({key: {"Z": item, "X": X, "S": S}})

    return time_dict, accuracy_dict, trace_dict


# Main entry point
if __name__ == "__main__":
    sk_params, _, _, lambda_list = benchmark_parameters()
    S_dict = load_dict(dict_name="S_dict")
    X_dict = load_dict(dict_name="X_dict")
    Z_dict = load_dict(dict_name="Z_dict")

    time_dict, accuracy_dict, trace_dict = run_sklearn(X_dict=X_dict, S_dict=S_dict, model_Z_dict=Z_dict,
                                                       lambda_list=lambda_list, n_iter=2, sklearn_params=sk_params)

    save_dict(D=time_dict, name="sklearn_time_dict")
    save_dict(D=accuracy_dict, name="sklearn_acc_dict")
    save_dict(D=trace_dict, name="sklearn_trace_dict")
