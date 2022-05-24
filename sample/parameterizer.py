import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import *
from experiment import Experiment
from session import Session
from utils import *


class Parameterizer:
    """
    Tests an experiment with different inputs. All generated data important to this session is stored in the
    'sample/parameterization' directory.
    """

    def __init__(self, parameter_function: Callable[[Dict[str, float]], Callable[[], Experiment]],
                 params: List[Dict[str, float]], name: str = None):
        """
        Initialization of the Parameterizer class
        :param parameter_function: Function that takes in a dictionary that contains the necessary parameters
        to test and returns a function that outputs a freshly defined experiment based on the parameters formed in the
         dictionary
        :param params: List of dictionaries to apply on parameter_function
        :param name: name for storing purposes
        """
        self.parameter_function = parameter_function
        self.parameters = params
        os.makedirs('parameterization', exist_ok=True)
        self.directory = f'parameterization/params_{len([a for a in os.scandir("parameterization")]) if name is None else name}'
        os.makedirs(self.directory, exist_ok=True)

    def run(self, save_graphs: bool = True, epochs: int = 10):
        res = {'mean_results': [], 'var_results': []}
        self.epochs = epochs
        for param in self.parameters:
            print(f'Testing parameters {compact_dict_print(param)}')
            experiment_function = self.parameter_function(param)
            session = Session(experiment_function, f'session_{compact_dict_print(param)}')
            results, var_res = session.run(save_graphs=False, epochs=epochs)
            res['mean_results'].append((param, results))
            res['var_results'].append((param, var_res))
            save_pandas_table(self.directory + f'/table_of_{compact_dict_print(param)}', results)
        if save_graphs:
            self.generate_graphs(res)

    def run_specific(self, test_set=pd.DataFrame, truth_set=pd.DataFrame, epochs: int = 10, save_graphs: bool = True):
        res = {'mean_results': [], 'var_results': []}
        self.epochs = epochs
        for param in self.parameters:
            print(f'Testing parameters {compact_dict_print(param)}')
            experiment_function = self.parameter_function(param)
            session = Session(experiment_function, f'session_{compact_dict_print(param)}')
            results, var_res = session.run_specific(test_set, truth_set, save_graphs=False, epochs=epochs)
            res['mean_results'].append((param, results))
            res['var_results'].append((param, var_res))
            save_pandas_table(self.directory + f'/table_of_{compact_dict_print(param)}', results)
        if save_graphs:
            self.generate_graphs(res)

    def generate_graphs(self, res):
        self.generate_param_graph_over_mean(res['mean_results'])
        self.generate_param_graph_with_variance(res['var_results'])


    def generate_param_graph_with_variance(self, res):
        plots = {}
        x = {}
        # Split the results into sections per parameters
        for params, results in res:
            for key in params:
                if key not in plots:
                    plots[key] = {}
                    x[key] = []
                x[key].append(params[key])
                # Parameters are now separated one by one
                # Iterate over all inter-results and extract the scores
                # plots[parameter][metric of model] = [[results for param1], [results for param2]...]
                new = {}
                for index, table in enumerate(results):
                    for model_name, scores in table.iterrows():
                        for score_name in scores.index:
                            new_key = f'{score_name} of {model_name}'
                            if new_key not in new:
                                new[new_key] = True
                            if new_key not in plots[key]:
                                plots[key][new_key] = [[]]
                                new[new_key] = False
                            if new[new_key]:
                                plots[key][new_key].append([])
                                new[new_key] = False
                            plots[key][new_key][-1].append(scores[score_name])
        for key in plots:
            plt.clf()
            for model in plots[key]:
                directory = self.directory + f'/variance_graph_{model}_{key}.png'
                plt.title(f'Boxplot of {self.epochs} replications measuring {model} while changing parameter \"{key}\"')
                plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
                metric_name = model.split(' of ')[0]
                plt.xlabel(key)
                plt.ylabel(metric_name)
                flierprops = dict(marker='+', markerfacecolor='r', markersize=10,
                                  linestyle='none', markeredgecolor='r')
                plt.boxplot(plots[key][model], flierprops=flierprops, meanline=True)
                plt.xticks(range(1, len(x[key]) + 1), x[key])
                plt.savefig(directory, bbox_inches="tight")
                plt.clf()

    def generate_param_graph_over_mean(self, res):
        """
        Generates graphs for each parameter that was changed and each metric and model that was tested.
        :param res: Results from the parameterization
        """
        plots = {}
        x = {}
        # Split the results into sections per parameters
        for params, results in res:
            for key in params:
                if key not in plots:
                    plots[key] = {}
                    x[key] = []
                x[key].append(params[key])
                for model_name, scores in results.iterrows():
                    for score_name in scores.index:
                        new_key = f'{score_name} of {model_name}'
                        if new_key not in plots[key]:
                            plots[key][new_key] = []
                        plots[key][new_key].append(scores[score_name])
        for key in plots:
            plt.clf()
            directory = self.directory + f'/graph_{key}.png'
            plt.title(f'Performance whilst changing parameter \"{key}\"')
            plt.xlabel(key)
            metric_name = None
            for model in plots[key]:
                metric_name = model.split(' of ')[0]
                plt.plot(x[key], plots[key][model], label=model)
            plt.ylabel(metric_name)
            plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
            plt.savefig(directory, bbox_inches="tight")
            plt.clf()
