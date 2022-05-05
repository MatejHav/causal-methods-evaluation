import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import *
from builders import Experiment
from session import Session
from utils import *


class Parameterizer:

    def __init__(self, parameter_function: Callable[[Dict[str, float]], Callable[[], Experiment]],
                 params: List[Dict[str, float]]):
        self.parameter_function = parameter_function
        self.parameters = params
        os.makedirs('parameterization', exist_ok=True)
        self.directory = f'parameterization/params_{len([a for a in os.scandir("parameterization")])}'
        os.makedirs(self.directory, exist_ok=True)

    def run(self, save_graphs: bool = True):
        res = []
        for param in self.parameters:
            print(f'Testing parameters {compact_dict_print(param)}')
            experiment_function = self.parameter_function(param)
            session = Session(experiment_function, f'session_{compact_dict_print(param)}')
            results = session.run(save_graphs=save_graphs)
            res.append((param, results))
            save_pandas_table(self.directory + f'/table_of_{compact_dict_print(param)}', results)
        if save_graphs:
            self.generate_graphs(res)

    def run_specific(self, test_set=pd.DataFrame, truth_set=pd.DataFrame, save_graphs: bool = True):
        res = []
        for param in self.parameters:
            print(f'Testing parameters {compact_dict_print(param)}')
            experiment_function = self.parameter_function(param)
            session = Session(experiment_function, f'session_{compact_dict_print(param)}')
            results = session.run_specific(test_set, truth_set, save_graphs=save_graphs)
            res.append((param, results))
            save_pandas_table(self.directory + f'/table_of_{compact_dict_print(param)}', results)
        if save_graphs:
            self.generate_graphs(res)

    def generate_graphs(self, res):
        plots = {}
        x = {}
        for params, results in res:
            for key in params:
                if key not in plots:
                    plots[key] = {}
                    x[key] = []
                x[key].append(params[key])
                for model_name, scores in results.iterrows():
                    for score_name in scores.index:
                        new_key = f'{model_name}_{score_name}'
                        if new_key not in plots[key]:
                            plots[key][new_key] = []
                        plots[key][new_key].append(scores[score_name])
        for key in plots:
            plt.clf()
            directory = self.directory + f'/graph_{key}.png'
            plt.title(f'Changing of parameter {key}')
            plt.xlabel(key)
            plt.ylabel('metrics')
            for model in plots[key]:
                plt.plot(x[key], plots[key][model], label=model)
            plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
            plt.savefig(directory, bbox_inches="tight")
            plt.clf()
