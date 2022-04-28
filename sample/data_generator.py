import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from typing import *

class Generator:

    # main_effect -> effect of features on the outcome
    # treatment_effect -> effect of the treatment on the outcome
    # treatment_propensity -> effect on the features on being treated
    def __init__(self, main_effect: Callable[[List[float]], float],
                 treatment_effect: Callable[[List[float]], float],
                 treatment_propensity: Callable[[List[float]], float],
                 noise: Callable[[], float],
                 dimensions: int, distributions: [Callable[[], float]]):
        # Either have 1 distribution applied on all features, or have a specific distribution per feature
        assert len(distributions) == 1 or len(distributions) == dimensions
        self.main_effect = main_effect
        self.treatment_effect = treatment_effect
        self.treatment_propensity = treatment_propensity
        self.noise = noise
        self.dimensions = dimensions
        self.distributions = distributions
        self.directory: str = f'data/data_dump_{self.__hash__()}'
        self.generated_files: Dict[str, List[str]] = {'data': [], 'graphs': []}

    def generate_data(self, number_of_samples: int, save_data: bool = True, show_graphs: bool = False,
                      save_graphs: bool = False):
        columns = [f'feature_{i}' for i in range(self.dimensions)]
        columns.append('treatment')
        columns.append('outcome')
        columns.append('main_effect')
        columns.append('treatment_effect')
        columns.append('propensity')
        df = pd.DataFrame([], columns=columns)
        for i in range(number_of_samples):
            features, treatment, outcome, main_effect, treatment_effect, propensity = self.generate_row()
            features.append(treatment)
            features.append(outcome)
            features.append(main_effect)
            features.append(treatment_effect)
            features.append(propensity)
            df.loc[len(df.index)] = features
        if save_data:
            self.save_data(df)
        if show_graphs or save_graphs:
            self.create_graphs(df)
            if save_graphs:
                self.save_graphs()
            if show_graphs:
                plt.show()
        return select_features(df, self.dimensions), df['treatment'], df['outcome'], df['main_effect'], df['treatment_effect'], df['propensity']

    def generate_row(self):
        features = []
        for dimension in range(self.dimensions):
            features.append(self.generate_feature(dimension))
        # W = bernoulli(e(x))
        propensity = self.treatment_propensity(features)
        treatment = 1 if np.random.random() <= propensity else 0
        treatment_effect = self.treatment_effect(features) # if treatment == 1 else 0
        # Y = m(x) + (W - 0.5) * t(x) + noise
        main_effect = self.main_effect(features)
        outcome = main_effect + treatment_effect + self.noise()
        return features, treatment, outcome, main_effect, treatment_effect, propensity

    def generate_feature(self, index):
        if len(self.distributions) == 1:
            return self.distributions[0]()
        return self.distributions[index]()

    def create_graphs(self, df):
        self.create_coverage_graph(df)

    def create_coverage_graph(self, df):
        plt.clf()
        feature_one = df['feature_0']
        feature_two = df['feature_1']
        maximal = df['treatment_effect'].max()
        minimal = df['treatment_effect'].min()
        color_function = lambda i: [1,
                                    min(1,
                                        1.1 * (df.iloc[i]['treatment_effect'] - minimal) / (maximal - minimal + 0.01)),
                                    0.95 * (df.iloc[i]['treatment_effect'] - minimal) / (maximal - minimal + 0.01)]
        plt.scatter(feature_one, feature_two, c=[color_function(i) for i in df.index])

    def save_graphs(self):
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        filename = f'/coverage_{time.ctime()}'.replace(' ', '_').replace(':', '-')
        self.generated_files['graphs'].append(filename)
        plt.savefig(self.directory + filename)

    def save_data(self, df):
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        filename = f'/generated_data{time.ctime()}.csv'.replace(' ', '_').replace(':', '-')
        self.generated_files['data'].append(filename)
        df.to_csv(self.directory + filename)


def select_features(df, dim):
    return df[[f'feature_{i}' for i in range(dim)]]
