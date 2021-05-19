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
    df = df.sort_values(by=['p', 'time'])

    # filter by lambda
    df = df[df["l1"] == 0.1]

    color_discrete_map = {'block-boyd': '#FF0000', 'regain': '#32CD32',
                          'single-boyd': '#FF8C00', 'sklearn': '#0000FF'}

    fig = px.scatter(df[(df["accuracy"] < upper_bound) & (df["accuracy"] > lower_bound)],
                     y="time", x="accuracy", text="l1", color="method_str",
                     log_y=True, log_x=True, facet_col='p', facet_col_wrap=3, color_discrete_map=color_discrete_map,
                     labels={
                         "time": "Time, s",
                         "accuracy": "Accuracy",
                         "method": "method"
                     },
                     template="plotly_white",
                     title="ADMM performance benchmark.<br>Accuracy is between 0.01 and 0.0001")

    fig.update_traces(mode='markers+lines', marker_line_width=1, marker_size=10)
    fig.update_xaxes(matches=None)
    fig.update_yaxes(exponentformat="power")

    return fig

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
    df = df.groupby(['method_str', "p", "l1"], as_index=False)['time'].mean()
    color_discrete_map = {'block-boyd': '#FF0000', 'regain': '#32CD32',
                          'single-boyd': '#FF8C00', 'sklearn': '#0000FF'}

    fig = px.scatter(df, x="p", y="time", text="p", color="method_str",
                     log_y=True, facet_col='l1', facet_col_wrap=3, color_discrete_map=color_discrete_map,
                     labels={
                         "time": "Time, s",
                         "p": "Number of features, p",
                         "method": "method"
                     },
                     template="plotly_white",
                     title="Scalability of ADMM with different lambdas.<br>Accuracy is between 0.01 and 0.0001")

    fig.update_traces(mode='markers+lines', marker_line_width=1, marker_size=10)
    # fig.update_annotations(text="Accuracy is between 0.01 and 0.0001")

    return fig
