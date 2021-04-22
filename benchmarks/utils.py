"""
This script contains function helping to analyze the results of
time, sparsity and scaling benchmarks.
"""
import pandas as pd
import numpy as np

from gglasso.helper.utils import hamming_distance
from gglasso.helper.data_generation import group_power_network, sample_covariance_matrix


def network_generation(p=int, N=int, K=1, M=int):
    """
    Generates a law-power network with number of connected components bigger than 1.
    :param p: int
    Number of features.
    :param N: int
    Number of samples.
    :param K: int, default: ‘1’
    Number of instances with p features and N samples.
    :param M: int
    Number of subblock in each instance K.
    :return: S, X, Theta
    S - empirical covarince matrix.
    X - dual variable.
    Theta - true precision matrix.
    """
    Sigma, Theta = group_power_network(p, K=K, M=M)
    S, samples = sample_covariance_matrix(Sigma, N)

    S = S[0, :, :]
    Theta = Theta[0, :, :]
    X = samples[0, :, :].T

    return S, X, Theta


def benchmarks_dataframe(times=dict, acc_dict=dict, spars_dict=dict):
    """
    Turn benchmark dictionaries into dataframes.
    :param times: dict
    Input dictionary where 'key' is the model and 'value' is its runtime.
    :param acc_dict: dict
    Input dictionary where 'key' is a model and 'value' is its corresponding accuracy given as:
    np.linalg.norm(Z - np.array(Z_i)) / np.linalg.norm(Z)
    where Z is our model solution and Z_i is the model.
    :param spars_dict: dict
    Input dictionary where 'key' is a model and 'value' is its corresponding sparsity measured by
    Hamming distance.
    :return: Pandas.DataFrame()
    """
    assert len(times) == len(acc_dict) == len(spars_dict)

    # The time measured during the grid search of best hyperparameters for the models
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


def best_time_dataframe(best_time=dict):
    """
    Turn dictionaries of the model best running times into dataframes.
    :param best_time: dict
    Input dictionary where 'key' is the selected best model and 'value' is its runtime.
    :return: Pandas.DataFrame()
    """
    # The time measured during the scalability benchmark
    time_df = pd.DataFrame.from_dict(best_time, orient='index', columns=['time'])
    time_df.reset_index(level=0, inplace=True)

    time_df['split'] = time_df['index'].str.split('_')
    columns_names = ["method", "p_str", "p", "N_str", "N"]
    time_df[columns_names] = pd.DataFrame(time_df['split'].tolist(), index=time_df['split'].index)

    redundant_cols = ["p_str", "N_str"]
    time_df = time_df.drop(redundant_cols, axis=1)

    convert_dict = {"p": int, "N": int}
    time_df = time_df.astype(convert_dict)
    time_df.sort_values(by=['time'])

    return time_df


def drop_acc_duplicates(df):
    """
    Drop duplicates of the models showing the same accuracy.
    :param df: pd.DataFrame()
    :return: pd.DataFrame()
    """
    assert 'method' in df.columns
    assert 'accuracy' in df.columns

    unique_acc_df = df[:1]
    for method in df.method.unique():
        filtered = df[df['method'] == method]
        filtered = filtered.drop_duplicates(subset='accuracy', keep='first')
        unique_acc_df = pd.concat([filtered, unique_acc_df])

    return unique_acc_df[:-1]


def dict_shape(dict_=dict):
    shape_list = []

    for i in dict_.values():
        shape_list.append(np.array(i).shape)
    return shape_list


def hamming_dict(Theta_dict=dict, Z_dict=dict, t_rounding=float):
    """
    Calculate Hamming distance between model solution Z and given solution Theta
    with a specified rounding accuracy t_rounding.
    :param Theta_dict: dict
    :param Z_dict: dict
    :param t_rounding: 1e-10, float
    :return: dict
    """
    sparsity_dict = dict()

    for Theta in Theta_dict.values():

        for key, Z in Z_dict.items():
            if Theta.shape == Z.shape:
                sparsity_dict[key] = hamming_distance(Theta, Z, t=t_rounding)

    return sparsity_dict
