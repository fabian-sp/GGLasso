import time
import itertools

from benchmarks.utilita import network_generation, save_dict, model_solution


def build_datasets(p_list=list, N_list=list):
    S_dict = dict()
    X_dict = dict()
    Theta_dict = dict()

    print(" Power network generation ".center(40, '-'))

    for p, N in zip(p_list, N_list):
        try:
            start = time.perf_counter()
            S, X, Theta = network_generation(p, N, M=10)
            end = time.perf_counter()
            print("p: %5d, N : %5d, Time : %5.4f" % (p, N, end-start))
        except:
            print("Power network cannot be generated")
            print("Tip: increase the number of sub-blocks M")
            break

        S_dict[p, N] = S
        X_dict[p, N] = X
        Theta_dict[p, N] = Theta

    return S_dict, X_dict, Theta_dict


def Z_solution(X_dict=dict, lambda_list=list, model="sklearn"):
    model_time_dict = dict()
    model_Z_dict = dict()

    for X, l1 in itertools.product(list(X_dict.values()), lambda_list):
        Z, Z_time, info = model_solution(solver=model, X=X, lambda1=l1)

        N = X.shape[0]
        p = X.shape[1]

        key = "p_" + str(p) + "_N_" + str(N) + "_l1_" + str(l1)
        model_time_dict.update({key: Z_time})
        model_Z_dict.update({key: Z})
    print("Model solution({0}): {1}".format(model, info))
    return model_time_dict, model_Z_dict


# Main entry point
if __name__ == "__main__":
    p_list = [100, 200]
    N_list = [200, 400]
    lambda_list = [0.5, 0.1, 0.05]

    S_dict, X_dict, Theta_dict = build_datasets(p_list=p_list, N_list=N_list)

    save_dict(D=S_dict, name="S_dict")
    save_dict(D=X_dict, name="X_dict")
    save_dict(D=Theta_dict, name="Theta_dict")

    model_time_dict, model_Z_dict = Z_solution(X_dict=X_dict, lambda_list=lambda_list, model="regain")

    save_dict(D=model_time_dict, name="Z_time_dict")
    save_dict(D=model_Z_dict, name="Z_dict")
