import sys
import os

import matplotlib.pyplot as plt
from pandas.plotting import table
from typing import *


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def save_pandas_table(dir, df):
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
        result += f'{key}={dict[key]}{"," if index < len(dict) - 1 else  ""}'.replace(' ', '_').replace(':', '-')
    return result

def select_features(df, dim=-1):
    if dim == -1:
        return df[[name for name in df.columns if 'feature' in name]]
    return df[[f'feature_{i}' for i in range(dim)]]

def generate_coverage_of_model_graph(model, df, save_dir):
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
