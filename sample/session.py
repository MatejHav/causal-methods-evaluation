import os
import threading
import numpy as np
import pandas as pd

from typing import *
from utils import HiddenPrints, save_pandas_table
from alive_progress import alive_bar
from experiment import Experiment

class Session:
    """
    To test a model an experiment is replicated multiple times to get the average performance. This class facilitates
    that with its run method. It stores the result in the 'sample/sessions' directory.
    If running in an IDE make sure to turn on emulating console printing for progress bars.
    """

    def __init__(self, experiment: Callable[[], Experiment], name: str = None):
        """
        Initialization of the Session class
        :param experiment: A lambda function that takes to input and generates a freshly defined experiment class
        :param name: Name of the session for directory storage
        """
        if name is None:
            name = str(self.__hash__())
        self.experiment_function = experiment
        self.directory = f'sessions/session_{name}/'
        os.makedirs(self.directory, exist_ok=True)

    def batch(self, experiments, results, var_results):
        for experiment in experiments:
            experiment.run(save_graphs=False, save_data=True)
            results.append(experiment.results[-1].to_numpy())
            var_results.append(experiment.results[-1])

    def run(self, epochs: int = 10, save_graphs: bool=True):
        results_array = []
        variance_results = []
        model_names = None
        metric_names = None
        # with alive_bar(epochs) as bar:
        #     for _ in range(epochs):
        #         with HiddenPrints():
        #             experiment = self.experiment_function()
        #         if model_names is None and metric_names is None:
        #             model_names = [str(model) for model in experiment.models]
        #             metric_names = list(experiment.metrics.keys())
        #         experiment.run(save_graphs=False, save_data=True)
        #         if results is None:
        #             results = np.zeros((len(experiment.models), len(experiment.metrics)))
        #         results = results + experiment.results[-1].to_numpy()
        #         variance_results.append(experiment.results[-1])
        #         bar()
        threads = []
        # Create 6 threads eah having epoch // 6 experiments
        number_of_threads = 6
        for i in range(number_of_threads):
            experiments = [self.experiment_function() for _ in range(epochs // number_of_threads)]
            if model_names is None and metric_names is None:
                model_names = [str(model) for model in experiments[0].models]
                metric_names = list(experiments[0].metrics.keys())
            threads.append(threading.Thread(target=self.batch, daemon=True, args=(experiments, results_array, variance_results)))
            threads[-1].start()
        with alive_bar(number_of_threads) as bar:
            for i in range(number_of_threads):
                threads[i].join()
                bar()
        results = sum(results_array) / epochs
        results = pd.DataFrame(results, columns=metric_names, index=model_names)
        save_pandas_table(self.directory + '/results', results)
        return results, variance_results

    def run_specific(self, test_set=pd.DataFrame, truth_set=pd.DataFrame, epochs: int = 10, save_graphs: bool=True):
        results = None
        variance_results = []
        model_names = None
        metric_names = None
        with alive_bar(epochs) as bar:
            for _ in range(epochs):
                with HiddenPrints():
                    experiment = self.experiment_function()
                if model_names is None and metric_names is None:
                    model_names = [str(model) for model in experiment.models]
                    metric_names = list(experiment.metrics.keys())
                experiment.run(save_graphs=False, save_data=True)
                experiment.test_specific_set(test_set, truth_set)
                if results is None:
                    results = np.zeros((len(experiment.models), len(experiment.metrics)))
                results = results + experiment.results[-1].to_numpy()
                variance_results.append(experiment.results[-1])
                bar()
        results = results / epochs
        results = pd.DataFrame(results, columns=metric_names, index=model_names)
        save_pandas_table(self.directory + '/results', results)
        return results, variance_results

