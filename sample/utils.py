"""Utils

This file defines some functions useful throughout the entire project that do not fit anywhere else.

This file can also be imported as a module and contains the following
functions and classes:

    * HiddenPrints - blocks printing
    * save_pandas_table - function to save a pandas table in a specific directory
    * compact_dict_print - creates a string defining a dictionary without any illegal characters
    * select_features - selects only features from a pandas dataframe
    * generate_coverage_of_model_graph - generates a plot of coverage over first two features based on model outputs
"""

import sys
import os

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
from typing import *


class HiddenPrints:
    """
    Class to block printing.
    Taken from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def save_pandas_table(dir: str, df: pd.DataFrame):
    plt.clf()
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table(ax, df)
    plt.savefig(dir)
    df.to_csv(dir + '.csv')


def compact_dict_print(dict: Dict[str, Any]):
    result = ''
    for index, key in enumerate(dict):
        result += f'{key}={dict[key]}{"," if index < len(dict) - 1 else ""}'.replace(' ', '_').replace(':', '-')
    return result


def select_features(df: pd.DataFrame, dim: int=-1):
    if dim == -1:
        return df[[name for name in df.columns if 'feature' in name]]
    return df[[f'feature_{i}' for i in range(dim)]]


def generate_coverage_of_model_graph(model, df: pd.DataFrame, save_dir: str):
    plt.clf()
    feature_one = df['feature_0']
    feature_two = df['feature_1']
    predictions = model.estimate_causal_effect(select_features(df))
    maximal = df['treatment_effect'].max()
    minimal = df['treatment_effect'].min()
    color_function = lambda i: [0,
                                max(0, min(1, (predictions[i] - minimal) / (maximal - minimal + 0.01))),
                                max(0, min(1, 1 - (predictions[i] - minimal) / (maximal - minimal + 0.01)))]
    plt.scatter(feature_one, feature_two, c=[color_function(i) for i in df.index])
    plt.savefig(save_dir)
