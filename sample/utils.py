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
