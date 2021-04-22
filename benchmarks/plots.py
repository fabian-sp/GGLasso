"""
This script generates the plots for benchmark analysis.
"""
import plotly.express as px
import pandas as pd


def plot_accuracy(df=pd.DataFrame(), upper_bound=float, lower_bound=float):
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
    fig = px.scatter(df[(df["accuracy"] < upper_bound) & (df["accuracy"] > lower_bound)],
                     x="time", y="accuracy", text="name", color="method",
                     log_y=True, facet_col='p', facet_col_wrap=3,
                     labels={
                         "time": "Time, s",
                         "accuracy": "Log_distance",
                         "method": "method"
                     },
                     template="plotly_white",
                     title="Log-distance between Z and Z' with respect to ADMM convergence rates")

    fig.update_traces(mode='markers+lines', marker_line_width=1, marker_size=10)
    fig.update_xaxes(matches=None)
    fig.update_yaxes(exponentformat="power")

    return fig


def plot_scalability(df=pd.DataFrame()):
    """
    Plot how well different implementations of ADMM scale.
    :param df: pandas.DataFrame()
    :return: px.scatter()
    """
    fig = px.scatter(df, x="time", y="p", text="N", color="method",
                     labels={
                         "time": "Time, s",
                         "p": "Number of features, p",
                         "method": "method"
                     },
                     template="plotly_white",
                     title="Scalability plot")

    fig.update_traces(mode='markers+lines', marker_line_width=1, marker_size=10)

    return fig
