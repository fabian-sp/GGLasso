import numpy as np
import itertools
import time

from tqdm import trange

from sklearn.covariance import GraphicalLasso as sk_GL
from sklearn import set_config

from benchmarks.utilita import benchmark_parameters, save_dict, load_dict

set_config(print_changed_only=False)


def sklearn_time(X=np.array([]), Z=dict, sk_params=dict, lambda_list=list, n_iter=int, max_iter=50):
    precision_dict = dict()
    time_dict = dict()
    accuracy_dict = dict()
    iter_dict = dict()

    tol_list = sk_params["tol"]
    enet_list = sk_params["enet"]
    
    tol_enet_list = list(zip(tol_list, enet_list))

    p = X.shape[1]
    N = X.shape[0]

    for tol_enet, l1 in itertools.product(tol_enet_list, lambda_list):
        tol = tol_enet[0]
        enet = tol_enet[1]
        
        key = "sklearn" + "_tol_" + str(tol) + "_enet_" + str(enet) + "_p_" + str(p) + "_N_" + str(N) + "_l1_" + str(l1)

        time_list = list()
        iter_list = list()

        for _ in trange(n_iter, desc=key, leave=True):
            start = time.perf_counter()
            model = sk_GL(alpha=l1, tol=tol, enet_tol=enet, max_iter=max_iter, assume_centered=False, verbose=False)
            Z_i = model.fit(X)
            end = time.perf_counter()

            time_list.append(end - start)
            iter_list.append(model.n_iter_)

        time_dict[key] = np.mean(time_list)
        iter_dict[key] = int(np.mean(iter_list))

        precision_dict[key] = Z_i.precision_

        model_key = "p_" + str(p) + "_N_" + str(N) + "_l1_" + str(l1)
        accuracy = np.linalg.norm(Z[model_key] - np.array(Z_i.precision_)) / np.linalg.norm(Z[model_key])
        accuracy_dict[key] = accuracy

    result = {'time': time_dict, 'accuracy': accuracy_dict, 'sol': precision_dict, 'iter': iter_dict}

    return result


def run_sklearn(X_dict=dict, S_dict=dict, model_Z_dict=dict, lambda_list=list, n_iter=int, sklearn_params=dict):
    time_dict = dict()
    accuracy_dict = dict()
    Z_dict = dict()
    iter_dict = dict()

    for X, S in zip(list(X_dict.values()), list(S_dict.values())):
        sk_result = sklearn_time(X=X, Z=model_Z_dict, sk_params=sklearn_params,
                                                  lambda_list=lambda_list, n_iter=n_iter)

        time_dict.update(sk_result['time'])
        accuracy_dict.update(sk_result['accuracy'])
        Z_dict.update(sk_result['sol'])
        iter_dict.update(sk_result['iter'])

    return time_dict, accuracy_dict, Z_dict, iter_dict


# Main entry point
if __name__ == "__main__":
    sk_params, _, _, lambda_list = benchmark_parameters()
    S_dict = load_dict(dict_name="S_dict")
    X_dict = load_dict(dict_name="X_dict")
    Z_dict = load_dict(dict_name="Z_dict")

    time_dict, accuracy_dict, Z_dict, iter_dict = run_sklearn(X_dict=X_dict, S_dict=S_dict, model_Z_dict=Z_dict,
                                                       lambda_list=lambda_list, n_iter=2, sklearn_params=sk_params)

    save_dict(D=time_dict, name="sklearn_time_dict")
    save_dict(D=accuracy_dict, name="sklearn_acc_dict")
    save_dict(D=Z_dict, name="sklearn_sol_dict")
    save_dict(D=iter_dict, name="sklearn_iter_dict")
