import numpy as np
import pandas as pd

from data_generator import *
from causal_effect_methods import *
from compare import *
from typing import *


class Experiment:

    def __init__(self, seed: int = None):
        self.generators: List[(Generator, int)] = []
        self.models: List[CausalMethod] = []
        self.metrics: Dict[str, Callable[[List[float]], float]] = {}
        self.seed = np.random.seed(seed)
        self.directory = f'experiments/experiment_{seed if seed is not None else f"randomized{self.__hash__()}"}'
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def add_custom_generator(self, generator: Generator, sample_size: int = 500):
        self.generators.append((generator, sample_size))
        return self

    def add_custom_model(self, model: CausalMethod):
        self.models.append(model)
        return self

    def add_custom_metric(self, name: str, scoring_function: Callable[[List[float]], float]):
        self.metrics[name] = scoring_function
        return self

    # RUN EXPERIMENT

    def run(self, save_data: bool = True, save_graphs: bool = True, show_graphs: bool = False):
        model_dictionary = {}
        for model in self.models:
            model_dictionary[str(model)] = model
        metric_dictionary = self.metrics
        results = np.array([0 for _ in metric_dictionary])
        for generator, sample_size in self.generators:
            generator.directory = self.directory + "/" + generator.directory[5:]
            generator.generate_data(sample_size, save_data=save_data, save_graphs=save_graphs, show_graphs=show_graphs)
            result = run(model_dictionary, metric_dictionary,
                         data_file=generator.directory + generator.generated_files['data'][-1],
                         samples=sample_size, save_table=save_data,
                         dir=generator.directory + f'/inter_table_{len(results)}').to_numpy()
            results = results + result[:, 1:]

        results = results / len(self.generators)
        final_results = []
        for index, result in enumerate(results):
            result = list(result)
            result.insert(0, list(model_dictionary.keys())[index])
            final_results.append(result)
        columns = list(metric_dictionary.keys())
        columns.insert(0, 'method_name')
        final_result = pd.DataFrame(final_results, columns=columns)
        save_pandas_table(self.directory + '/table_result', final_result)
        return self

    # MODELS

    def add_causal_forest(self, number_of_trees=2000, max_leaf_size=1):
        return self.add_custom_model(CausalForest(number_of_trees, k=max_leaf_size))

    def add_dragonnet(self, dimensions):
        return self.add_custom_model(DragonNet(dimensions))

    # METRICS

    def add_mean_squared_error(self):
        return self.add_custom_metric('mean_squared_error',
                                      lambda truth, pred: np.sum(
                                          [(truth[i] - pred[i]) ** 2 for i in range(len(truth))]) / np.prod(truth.shape))

    # DATA GENERATORS

    def add_custom_generated_data(self, main_effect: Callable[[List[float]], float],
                                  treatment_effect: Callable[[List[float]], float],
                                  treatment_propensity: Callable[[List[float]], float],
                                  noise: Callable[[], float], dimensions: int,
                                  distributions=None):
        if distributions is None:
            distributions = [np.random.random]
        generator = data_generator.Generator(main_effect, treatment_effect, treatment_propensity, noise,
                                             dimensions=dimensions, distributions=distributions)
        return self.add_custom_generator(generator)

    def add_all_effects_generator(self, dimensions: int):
        main_effect = lambda x: 2 * x[0] - 1
        treatment_effect = lambda x: (1 + 1 / (1 + np.exp(-20 * (x[0] - 1 / 3)))) * (
                1 + 1 / (1 + np.exp(-20 * (x[1] - 1 / 3))))
        treatment_propensity = lambda x: (1 + beta.pdf(x[0], 2, 4)) / 4
        noise = lambda: 0.05 * np.random.normal(0, 1)
        return self.add_custom_generated_data(main_effect, treatment_effect, treatment_propensity, noise, dimensions)

    def add_no_treatment_effect_generator(self, dimensions: int):
        main_effect = lambda x: 2 * x[0] - 1
        treatment_effect = lambda x: 0
        treatment_propensity = lambda x: (1 + beta.pdf(x[0], 2, 4)) / 4
        noise = lambda: 0.05 * np.random.normal(0, 1)
        return self.add_custom_generated_data(main_effect, treatment_effect, treatment_propensity, noise, dimensions)

    def add_only_treatment_effect_generator(self, dimensions: int):
        main_effect = lambda x: 0
        treatment_effect = lambda x: (1 + 1 / (1 + np.exp(-20 * (x[0] - 1 / 3)))) * (
                1 + 1 / (1 + np.exp(-20 * (x[1] - 1 / 3))))
        treatment_propensity = lambda x: 0.5
        noise = lambda: 0.05 * np.random.normal(0, 1)
        return self.add_custom_generated_data(main_effect, treatment_effect, treatment_propensity, noise, dimensions)
