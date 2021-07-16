import numpy as np
import itertools
import time

from tqdm import trange

from gglasso.solver.single_admm_solver import ADMM_SGL
from gglasso.solver.single_admm_solver import block_SGL, get_connected_components

from benchmarks.utilita import benchmark_parameters, save_dict, load_dict, numba_warmup


def call_gglasso(S=np.array([]), Omega_0=np.array([]), Theta_0=np.array([]), X_0=np.array([]),
                 method=str, n_iter=int, tol=float, rtol=float, l1=float, stop_crit=str,
                 key=str, max_iter=5000):
    
    all_times = list()
    all_iter = list()
    
    for _ in trange(n_iter, desc=key, leave=True):

        if method == "gglasso":
            start = time.perf_counter()
            sol, info = ADMM_SGL(S, lambda1=l1,
                                 Omega_0=Omega_0, Theta_0=Theta_0, X_0=X_0,
                                 max_iter=max_iter, tol=tol, rtol=rtol,
                                 stopping_criterion=stop_crit, measure=True)
            end = time.perf_counter()
            
            all_times.append(end - start)
            all_iter.append(len(info['residual']))

        elif method == "gglasso-block":
            start = time.perf_counter()
            sol = block_SGL(S, lambda1=l1,
                            Omega_0=Omega_0, Theta_0=Theta_0, X_0=X_0,
                            max_iter=max_iter, tol=tol, rtol=rtol,
                            stopping_criterion=stop_crit)
            end = time.perf_counter()
            
            all_times.append(end - start)
            all_iter.append(0)
            
            numC, _ = get_connected_components(S, l1)
            print("{0}: {1} connected components.".format(key, numC))

    return sol, all_times, all_iter


def gglasso_time(S=np.array([]), X=np.array([]), Omega_0=np.array([]), Z=dict, lambda_list=list, n_iter=int,
                 gglasso_params=dict, warm_start=False):
    
    precision_dict = dict()
    accuracy_dict = dict()
    time_dict = dict()
    iter_dict = dict()

    tol_list = gglasso_params["tol"]
    rtol_list = gglasso_params["rtol"]
    method_list = gglasso_params["method"]
    stop_crit = gglasso_params["stop"]

    # Initialize numba
    numba_warmup(S=S)
    p = X.shape[1]
    N = X.shape[0]
    
    tol_rtol_list = list(zip(tol_list, rtol_list))

    for method, tol_rtol in itertools.product(method_list, tol_rtol_list):
        
        tol = tol_rtol[0]
        rtol = tol_rtol[1]

        addon_time = 0.
        Theta_0 = Omega_0.copy();
        X_0 = np.zeros((S.shape[0], S.shape[0]))

        for l1 in lambda_list:

            pars = "_tol_" + str(tol) + "_rtol_" + str(rtol) + "_p_" + str(p) + "_N_" + str(N) + "_l1_" + str(l1)
            key = method + pars

            # Run GGLasso
            Z_i, all_times, all_iter = call_gglasso(S=S, Omega_0=Omega_0, Theta_0=Theta_0, X_0=X_0,
                                                    method=method, n_iter=n_iter, tol=tol, rtol=rtol, l1=l1,
                                                    stop_crit=stop_crit, key=key)

            # update starting points
            if warm_start:
                Omega_0, Theta_0 = Z_i["Omega"], Z_i["Theta"]

            time_dict[key] = np.mean(all_times) + addon_time
            iter_dict[key] = int(np.mean(all_iter))
            
            # add needed time to addon
            if warm_start:
                addon_time += np.mean(all_times)

            precision_dict[key] = Z_i["Theta"]

            model_key = "p_" + str(p) + "_N_" + str(N) + "_l1_" + str(l1)
            accuracy = np.linalg.norm(Z[model_key] - np.array(Z_i["Theta"])) / np.linalg.norm(Z[model_key])
            accuracy_dict[key] = accuracy
            
    result = {'time': time_dict, 'accuracy': accuracy_dict, 'sol': precision_dict, 'iter': iter_dict}

    return result


def run_gglasso(X_dict=dict, S_dict=dict, model_Z_dict=dict, lambda_list=list, n_iter=int, gglasso_params=dict):
    time_dict = dict()
    accuracy_dict = dict()
    Z_dict = dict()
    iter_dict = dict()

    for X, S in zip(list(X_dict.values()), list(S_dict.values())):
        Omega_0 = np.eye(len(S))
        gg_result = gglasso_time(S=S, X=X, Omega_0=Omega_0, Z=model_Z_dict, lambda_list=lambda_list,
                                                  n_iter=n_iter, gglasso_params=gglasso_params)
        time_dict.update(gg_result['time'])
        accuracy_dict.update(gg_result['accuracy'])
        Z_dict.update(gg_result['sol'])
        iter_dict.update(gg_result['iter'])

    return time_dict, accuracy_dict, Z_dict, iter_dict


# Main entry point
if __name__ == "__main__":
    _, _, gglasso_params, lambda_list = benchmark_parameters()
    S_dict = load_dict(dict_name="S_dict")
    X_dict = load_dict(dict_name="X_dict")
    Z_dict = load_dict(dict_name="Z_dict")

    time_dict, accuracy_dict, Z_dict, iter_dict = run_gglasso(X_dict=X_dict, S_dict=S_dict, model_Z_dict=Z_dict,
                                                       lambda_list=lambda_list, n_iter=2,
                                                       gglasso_params=gglasso_params)

    save_dict(D=time_dict, name="gglasso_time_dict")
    save_dict(D=accuracy_dict, name="gglasso_acc_dict")
    save_dict(D=Z_dict, name="gglasso_sol_dict")
    save_dict(D=iter_dict, name="gglasso_iter_dict")