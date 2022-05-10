import os
import numpy as np
import pandas as pd

from typing import *
from utils import HiddenPrints, save_pandas_table
from alive_progress import alive_bar
from builders import Experiment

class Session:

    def __init__(self, experiment: Callable[[], Experiment], name: str = None):
        if name is None:
            name = str(self.__hash__())
        self.experiment_function = experiment
        self.directory = f'sessions/session_{name}/'
        os.makedirs(self.directory, exist_ok=True)

    def run(self, epochs: int = 10, save_graphs: bool=True):
        results = None
        model_names = None
        metric_names = None
        with alive_bar(epochs) as bar:
            for _ in range(epochs):
                with HiddenPrints():
                    experiment = self.experiment_function()
                if model_names is None and metric_names is None:
                    model_names = [str(model) for model in experiment.models]
                    metric_names = list(experiment.metrics.keys())
                experiment.run(save_graphs=save_graphs)
                if results is None:
                    results = np.zeros((len(experiment.models), len(experiment.metrics)))
                results = results + experiment.results[-1].to_numpy()
                bar()
        results = results / epochs
        results = pd.DataFrame(results, columns=metric_names, index=model_names)
        save_pandas_table(self.directory + '/results', results)
        return results

    def run_specific(self, test_set=pd.DataFrame, truth_set=pd.DataFrame, epochs: int = 50, save_graphs: bool=True):
        results = None
        model_names = None
        metric_names = None
        with alive_bar(epochs) as bar:
            for _ in range(epochs):
                with HiddenPrints():
                    experiment = self.experiment_function()
                if model_names is None and metric_names is None:
                    model_names = [str(model) for model in experiment.models]
                    metric_names = list(experiment.metrics.keys())
                experiment.run(save_graphs=save_graphs)
                experiment.test_specific_set(test_set, truth_set)
                if results is None:
                    results = np.zeros((len(experiment.models), len(experiment.metrics)))
                results = results + experiment.results[-1].to_numpy()
                bar()
        results = results / epochs
        results = pd.DataFrame(results, columns=metric_names, index=model_names)
        save_pandas_table(self.directory + '/results', results)
        return results

