from compare import *
from typing import *
from scipy.stats import multivariate_normal, beta
from datetime import datetime


class Experiment:
    """
    Creates an experiment builder which can be used to create some specific experiments.
    """

    def __init__(self, seed: int = None, name: str = None):
        self.name = name
        self.reset(seed)

    def __hash__(self):
        if self.name is not None:
            return self.name
        return super().__hash__()

    def reset(self, seed: int = None):
        """
        Resets the experiment.
        :param seed: seed for random state.
        :return: self
        """
        self.results: List[pd.DataFrame] = []
        self.generators: List[(Generator, int)] = []
        self.models: List[CausalMethod] = []
        self.metrics: Dict[str, Callable[[List[float], List[float]], float]] = {}
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        self._set_defaults()
        self.trained: bool = False
        self.count: int = 0
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        seed = f"seeded_{seed}_{self.__hash__()}" if seed is not None else f"randomized_{self.__hash__()}"
        self.directory = f'experiments/experiment_{datetime_str}_{seed}'
        os.makedirs(self.directory, exist_ok=True)
        return self

    def clear(self):
        """
        Completely clear generated data, but keep the models.
        :return: self
        """
        self.results = []
        self.generators = []
        self._set_defaults()
        self.trained = False
        return self

    def add_custom_generator(self, generator: Generator, sample_size: int = 500):
        """
        Adds a custom generator.
        :param generator: Generator to be added
        :param sample_size: Number of samples to be generated by the generator
        :return: self
        """
        self.generators.append((generator, sample_size))
        return self

    def add_custom_model(self, model: CausalMethod):
        """
        Add a causal model into the experiment
        :param model: Model to be added
        :return: self
        """
        self.models.append(model)
        return self

    def add_custom_metric(self, name: str, scoring_function: Callable[[List[float]], float]):
        """
        Add a custom metric to the experiment.
        :param name: Name of the metric
        :param scoring_function: Lambda function that takes in an array of results and outputs a number
        :return: self
        """
        self.metrics[name] = scoring_function
        return self

    # RUN EXPERIMENT

    def run(self, save_data: bool = True, save_graphs: bool = True, show_graphs: bool = False):
        """
        Runs the experiment. First trains all the models and then evaluates them.
        :param save_data: Boolean whether the generated data should be stored
        :param save_graphs: Boolean whether graphs should be stored
        :param show_graphs: Boolean whether graphs should be shown
        :return: self
        """
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
                         dir=generator.directory).to_numpy()
            results = results + result

        results = results / len(self.generators)
        final_results = []
        for index, result in enumerate(results):
            result = list(result)
            result.insert(0, list(model_dictionary.keys())[index])
            final_results.append(result)
        columns = list(metric_dictionary.keys())
        columns.insert(0, 'method_name')
        final_result = pd.DataFrame(final_results, columns=columns)
        final_result = final_result.set_index('method_name')
        self.results.append(final_result)
        save_pandas_table(self.directory + '/final_table', final_result)
        self.trained = True
        return self

    # For each model run this specific test_set
    # Compare with given metrics
    def test_specific_set(self, test_set=pd.DataFrame, truth_set=pd.DataFrame):
        """
        Tests all the trained models on specific data set provided by the user. Make sure to first run the 'run' method.
        :param test_set: Dataframe of feature vectors to be tested
        :param truth_set: Dataframe of corresponding expected results
        :return: self
        """
        assert self.trained, "Models are not trained yet. Please make sure you run the full experiment first!"
        self.count += 1
        columns = [name for name in self.metrics]
        columns.insert(0, 'method_name')
        df = pd.DataFrame([], columns=columns)
        for model in self.models:
            predictions = model.estimate_causal_effect(test_set)
            row = [model.__str__()]
            for metric in self.metrics.values():
                score = metric(truth_set.to_numpy(), predictions)
                row.append(score)
            df.loc[len(df.index)] = row
        df = df.set_index('method_name')
        self.results.append(df)
        save_pandas_table(self.directory + f'/table_comparing_specific_value_{self.count}', df)
        return self

    # MODELS

    def add_causal_forest(self, number_of_trees=100, min_leaf_size=10, honest: bool=True):
        return self.add_custom_model(CausalForest(number_of_trees, k=min_leaf_size, honest=honest, id = len(self.models)))

    def add_dragonnet(self, dimensions):
        return self.add_custom_model(DragonNet(dimensions, id = len(self.models)))

    # METRICS

    def add_mean_squared_error(self):
        return self.add_custom_metric('mean_squared_error',
                                      lambda truth, pred: np.sum(
                                          [(truth[i] - pred[i]) ** 2 for i in range(len(truth))]) / np.prod(truth.shape))

    def add_absolute_error(self):
        return self.add_custom_metric('absolute_error',
                                      lambda truth, pred: np.sum(
                                          [abs(truth[i] - pred[i]) for i in range(len(truth))]) / np.prod(truth.shape))

    # DATA GENERATORS

    def _set_defaults(self):
        """
        Sets the default functions used throughout the project.
        """
        self.main_effect = lambda x: 2 * x[0] - 1
        self.treatment_effect = lambda x: (1 + 1 / (1 + np.exp(-20 * (x[0] - 1 / 3)))) * (
                1 + 1 / (1 + np.exp(-20 * (x[1] - 1 / 3))))
        # https://en.wikipedia.org/wiki/Beta_distribution
        self.treatment_propensity = lambda x: (1 + beta.pdf(x[0], 2, 4)) / 4
        self.noise = lambda: 0.05 * np.random.normal(0, 1)
        self.treatment_function = lambda propensity, noise: 1 if np.random.random() <= propensity else 0
        self.outcome_function = lambda main, treat, treat_eff, noise: main + (treat - 0.5) * treat_eff + noise
        # E[Y1 - Y0 | X] = 0.5 * treat_eff(x) + 0.5*treat_eff(x) = treat_eff(x)
        self.cate = lambda x: self.treatment_effect(x)

    def add_custom_generated_data(self, main_effect: Callable[[List[float]], float],
                                  treatment_effect: Callable[[List[float]], float],
                                  treatment_propensity: Callable[[List[float]], float],
                                  noise: Callable[[], float],
                                  cate: Callable[[List[float]], float], dimensions: int,
                                  treatment_function: Callable[[float, float], float],
                                  outcome_function: Callable[[float, float, float, float], float],
                                  distributions=None, sample_size: int = 500, name: str=None):
        if distributions is None:
            distributions = [np.random.random]
        generator = data_generator.Generator(main_effect=main_effect, treatment_effect=treatment_effect,
                                             treatment_propensity=treatment_propensity, noise=noise, cate=cate,
                                             treatment_function=treatment_function, outcome_function=outcome_function,
                                             dimensions=dimensions, distributions=distributions, name=name)
        return self.add_custom_generator(generator, sample_size=sample_size)

    def add_all_effects_generator(self, dimensions: int, sample_size: int = 500):
        main_effect = self.main_effect
        treatment_effect = self.treatment_effect
        treatment_propensity = self.treatment_propensity
        noise = self.noise
        treatment_function = self.treatment_function
        outcome_function = self.outcome_function
        cate = self.cate
        return self.add_custom_generated_data(main_effect, treatment_effect, treatment_propensity, noise, cate,
                                              dimensions, treatment_function, outcome_function,
                                              sample_size=sample_size, name='all_effects')

    def add_no_treatment_effect_generator(self, dimensions: int, sample_size: int = 500):
        main_effect = self.main_effect
        treatment_effect = lambda x: 0
        treatment_propensity = self.treatment_propensity
        noise = self.noise
        treatment_function = self.treatment_function
        outcome_function = self.outcome_function
        # E[Y1 - Y0 | X] = 0 as there is no dependence on treatment
        cate = lambda x: 0
        return self.add_custom_generated_data(main_effect, treatment_effect, treatment_propensity, noise, cate,
                                              dimensions, treatment_function, outcome_function,
                                              sample_size=sample_size, name='no_treatment_effect')

    def add_only_treatment_effect_generator(self, dimensions: int, sample_size: int = 500):
        main_effect = lambda x: 0
        treatment_effect = self.treatment_effect
        treatment_propensity = lambda x: 0.5
        noise = self.noise
        treatment_function = self.treatment_function
        outcome_function = self.outcome_function
        # E[Y1 - Y0|X] = E[0.5*treat_eff + 0.5*treat_eff] = treat_eff
        cate = lambda x: treatment_effect(x)
        return self.add_custom_generated_data(main_effect, treatment_effect, treatment_propensity, noise, cate,
                                              dimensions, treatment_function, outcome_function,
                                              sample_size=sample_size, name='only_treatment_effect')

    def add_biased_generator(self, dimensions: int, sample_size: int = 500):
        main_effect = lambda x: 0
        treatment_effect = lambda x: 1 if np.random.random() <= 0.05 else 0
        treatment_propensity = lambda x: 0.5
        noise = lambda : np.random.normal(0, 0.01)
        treatment_function = lambda propensity, noise: 1 if np.random.random() <= propensity else 0
        outcome_function = lambda main, treat, treat_eff, noise: 2 * treat * treat_eff + noise
        # E[Y1 - Y0 | X] = E[Y1|X] - E[Y0 | X] = 0.1 - 0 = 0.1
        cate = lambda x: 0.1
        return self.add_custom_generated_data(main_effect, treatment_effect, treatment_propensity, noise, cate,
                                              dimensions, treatment_function, outcome_function,
                                              sample_size=sample_size, name='biased_generator')

    def add_spiked_generator(self, dimensions: int, sample_size: int = 500):
        main_effect = self.main_effect
        # Spike around (0.5, 0.5) - equally spread through x and y
        # Very low std means a spike
        std = 0.01
        distr = multivariate_normal(cov=np.array([[std, 0], [0, std]]), mean=np.array([0.5, 0.5]),
                                    seed=42)
        treatment_effect = lambda x: distr.pdf([x[0], x[1]])
        # Closer to (0.5, 0.5), higher the chance of being treated
        treatment_propensity = lambda x: 1 - np.sqrt((x[0] - 0.5)**2 + (x[1] - 0.5)**2)
        noise = lambda: np.random.normal(0, 0.01)
        treatment_function = lambda propensity, noise: 1 if np.random.random() <= propensity else 0
        outcome_function = lambda main, treat, treat_eff, noise: main + treat * treat_eff + noise
        # E[Y1 - Y0 | X] = E[Y1 | X] - E[Y0 | X] = 1 * treat_eff = treat_eff(x)
        cate = lambda x: treatment_effect(x)
        return self.add_custom_generated_data(main_effect, treatment_effect, treatment_propensity, noise, cate,
                                              dimensions, treatment_function, outcome_function,
                                              sample_size=sample_size, name='spiked_generator')
