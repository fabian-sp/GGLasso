"""
This script generates the plots for benchmark analysis.
"""
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from benchmarks.utilita import benchmarks_dataframe


def plot_accuracy(df=pd.DataFrame(), upper_bound=float, lower_bound=float, lambda_filter=float, sortby=list):
    """
    Plot how accurate the solution of a model with particular hyperparameters
    comparing with the model solution Z.
    :param df: pandas.DataFrame()
    :param upper_bound: float
    Specify the upper bound for the solution accuracy.
    :param lower_bound: float
    Specify the lower bound for the solution accuracy.
    :return: px.scatter()
    """

    df = df.sort_values(by=sortby)

    # filter by lambda
    df = df[df["l1"] == lambda_filter]
    df["dim"] = list(zip(df.p, df.N))

    df_2 = df.copy()
    for method in df.method.unique():
        filtered = df[df['method'] == method]
        filtered = filtered.drop_duplicates(subset='accuracy', keep='first')
        df_2 = df_2.append(filtered, ignore_index=False)

    df = df_2.iloc[df.shape[0] + 1:, :]

    color_discrete_map = {'block-boyd': '#FF0000', 'regain': '#32CD32',
                          'single-boyd': '#FF8C00', 'sklearn': '#0000FF'}

    fig = px.scatter(df[(df["accuracy"] < upper_bound) & (df["accuracy"] > lower_bound)],
                     y="time", x="accuracy", text="l1", color="method",
                     log_y=True, log_x=True, facet_col='dim', facet_col_wrap=3, color_discrete_map=color_discrete_map,
                     labels={
                         "time": "Time, s",
                         "accuracy": "Accuracy",
                         "method": "method"
                     },
                     template="plotly_white",
                     title="ADMM performance benchmark at lambda={0}.<br>"
                           "Accuracy is between {1} and {2}".format(upper_bound, lower_bound, lambda_filter))

    fig.update_traces(mode='markers+lines', marker_line_width=1, marker_size=10)
    fig.update_xaxes(matches=None)
    fig.update_yaxes(exponentformat="power")

    return fig


def plot_scalability(df=pd.DataFrame()):
    """
    Plot how well the implementations of ADMM scale according to a different choice of lambda.
    :param df: pandas.DataFrame()
    :return: px.scatter()
    """
    color_discrete_map = {'block-boyd': '#FF0000', 'regain': '#32CD32',
                          'single-boyd': '#FF8C00', 'sklearn': '#0000FF'}

    fig = px.scatter(df, x="p", y="time", text="N", color="method",
                     log_y=True, facet_col='l1', facet_col_wrap=3, color_discrete_map=color_discrete_map,
                     labels={
                         "time": "Time, s",
                         "p": "Number of features, p",
                         "method": "method"
                     },
                     template="plotly_white",
                     title="Scalability of ADMM with different lambdas")

    fig.update_traces(mode='markers+lines', marker_line_width=1, marker_size=10)

    return fig


def plot_lambdas(df=pd.DataFrame(), upper_bound=float, lower_bound=float):

    df = df[(df["accuracy"] < upper_bound) & (df["accuracy"] > lower_bound)]

    df = df.groupby(['method', "l1", "p", 'N'], as_index=False)['time'].min()

    color_discrete_map = {'block-boyd': '#FF0000', 'regain': '#32CD32',
                          'single-boyd': '#FF8C00', 'sklearn': '#0000FF'}

    fig = px.scatter(df, x="p", y="time", text="N", color="method",
                     log_y=True, facet_col='l1', facet_col_wrap=3, color_discrete_map=color_discrete_map,
                     labels={
                         "time": "Time, s",
                         "p": "Number of features, p",
                         "N": "Number of samples, N"
                     },
                     template="plotly_white",
                     title="Scalability of ADMM with different lambdas.<br>"
                           "Accuracy is between {0} and {1}".format(upper_bound, lower_bound))

    fig.update_traces(mode='markers+lines', marker_line_width=1, marker_size=10)

    return fig


def plot_bm(df=pd.DataFrame(), lambda_list=list, min_acc=1e-2, log_scale=True):
    
    col_dict = {"gglasso": '#264F73', "gglasso-block": "#3F7EA6", 'regain': '#F2811D', 'sklearn': '#C0C0C0'}
    
    fig, axs = plt.subplots(len(lambda_list), 1, figsize=(6,10))
    j = 0
    for l1 in lambda_list:
        ax = axs[j]
        df_sub = df[(df.l1 == l1) & (df.accuracy <= min_acc)]
        tmp = df_sub.groupby(["p", "N", "method"])["time"].min()

        tmp.unstack().plot(ls='-', marker='o', xlabel="(p,N)", ylabel="runtime [sec]", ax=ax, color = col_dict)
        
        ax.set_title(rf"$\lambda_1$ = {l1}")
        ax.grid(linestyle='--')
        ax.legend(loc='upper left')
        
        if log_scale:
            ax.set_yscale('log')
            # ax.set_xscale('log')

        j += 1

    fig.tight_layout()
    return


# Main entry point
if __name__ == "__main__":
    df = pd.read_csv("data/synthetic/bm5000.csv", sep="\t")

    lambda_list = np.unique(df["l1"].values)
    plot_bm(df, lambda_list=lambda_list)

    plt.savefig('examples/plots/ggl_runtime/bm_{0}.png'.format(time.strftime("%Y%m%d")))
    plt.show()

