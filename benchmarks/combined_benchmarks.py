import numpy as np

from benchmarks.regain_benchmark import regain_time
from benchmarks.sklearn_benchmark import sklearn_time
from benchmarks.gglasso_benchmark import gglasso_time

from benchmarks.utilita import model_solution


def time_benchmarks(X=list, S=list, lambda_list=list, max_iter=50000, Z_model=str,
                   n_iter=int, sk_params=dict, rg_params=dict, gglasso_params=dict):
    assert Z_model in ('sklearn', 'regain')

    # Model solution Z
    Z, Z_time = model_solution(model=Z_model, X=X, lambda1=lambda_list)

    # Sklearn
    sk_time, sk_accuracy, Z_sk = sklearn_time(X=X, Z=Z, sk_params=sk_params, lambda_list=lambda_list, n_iter=n_iter,
                                              max_iter=max_iter)

    # Regain
    rg_time, rg_accuracy, Z_rg = regain_time(X=X, Z=Z, rg_params=rg_params, lambda_list=lambda_list, n_iter=n_iter,
                                             max_iter=max_iter)

    # GGLasso
    Omega_0 = np.eye(len(S))

    gg_time, gg_accuracy, Z_gg = gglasso_time(S=S, Omega_0=Omega_0, Z=Z, gglasso_params=gglasso_params,
                                              lambda_list=lambda_list, n_iter=1 + 1, max_iter=max_iter)

    # Join results in a single dictionary
    times = sk_time.copy()
    times.update(rg_time)
    times.update(gg_time)

    accs = sk_accuracy.copy()
    accs.update(rg_accuracy)
    accs.update(gg_accuracy)

    precs = Z_sk.copy()
    precs.update(Z_rg)
    precs.update(Z_gg)

    return times, accs, precs
