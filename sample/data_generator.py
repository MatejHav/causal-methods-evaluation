import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from utils import select_features
from typing import *

class Generator:

    # main_effect -> effect of features on the outcome
    # treatment_effect -> effect of the treatment on the outcome
    # treatment_propensity -> effect on the features on being treated
    def __init__(self, main_effect: Callable[[List[float]], float],
                 treatment_effect: Callable[[List[float]], float],
                 treatment_propensity: Callable[[List[float]], float],
                 noise: Callable[[], float],
                 cate: Callable[[List[float]], float],
                 treatment_function: Callable[[float, float], float],
                 outcome_function: Callable[[float, float, float, float], float],
                 dimensions: int, distributions: [Callable[[], float]], name: str = None):
        # Either have 1 distribution applied on all features, or have a specific distribution per feature
        assert len(distributions) == 1 or len(distributions) == dimensions
        if name is None:
            name = self.__hash__()
        self.main_effect = main_effect
        self.treatment_effect = treatment_effect
        self.treatment_propensity = treatment_propensity
        self.treatment_function = treatment_function
        self.outcome_function = outcome_function
        self.noise = noise
        self.cate = cate
        self.dimensions = dimensions
        self.distributions = distributions
        self.directory: str = f'data/data_dump_{name}'
        self.generated_files: Dict[str, List[str]] = {'data': [], 'graphs': []}

    def generate_data(self, number_of_samples: int, save_data: bool = True, show_graphs: bool = False,
                      save_graphs: bool = False):
        columns = [f'feature_{i}' for i in range(self.dimensions)]
        columns.append('treatment')
        columns.append('outcome')
        columns.append('main_effect')
        columns.append('treatment_effect')
        columns.append('propensity')
        columns.append('y0')
        columns.append('y1')
        columns.append('noise')
        columns.append('cate')
        df = pd.DataFrame([], columns=columns)
        for i in range(number_of_samples):
            features, treatment, outcome, main_effect, treatment_effect, propensity, y0, y1, noise, cate = self.generate_row()
            features.append(treatment)
            features.append(outcome)
            features.append(main_effect)
            features.append(treatment_effect)
            features.append(propensity)
            features.append(y0)
            features.append(y1)
            features.append(noise)
            features.append(cate)
            df.loc[len(df.index)] = features
        if save_data:
            self.save_data(df)
        if show_graphs or save_graphs:
            self.create_graphs(df)
            if save_graphs:
                self.save_graphs()
            if show_graphs:
                plt.show()
        return select_features(df, self.dimensions), df['treatment'], df['outcome'], df['main_effect'], \
               df['treatment_effect'], df['propensity'], df['y0'], df['y1'], df['noise'], df['cate']

    def generate_row(self):
        features = []
        for dimension in range(self.dimensions):
            features.append(self.generate_feature(dimension))
        # propensity = p(T = 1 | X)
        propensity = self.treatment_propensity(features)
        treatment_noise = self.noise()
        treatment = self.treatment_function(propensity, treatment_noise)
        treatment_effect = self.treatment_effect(features)
        main_effect = self.main_effect(features)
        noise = self.noise()
        outcome = self.outcome_function(main_effect, treatment, treatment_effect, noise)
        y0 = self.outcome_function(main_effect, 0, treatment_effect, 0)
        y1 = self.outcome_function(main_effect, 1, treatment_effect, 0)
        # Cate : E[Y|X]
        cate = self.cate(features)
        # True treatment effect is y1 - y0 where there is no noise
        return features, treatment, outcome, main_effect, y1 - y0, propensity, y0, y1, noise, cate

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
        color_function = lambda i: [0,
                                    min(1,
                                        (df.iloc[i]['treatment_effect'] - minimal) / (maximal - minimal + 0.01)),
                                    1 - (df.iloc[i]['treatment_effect'] - minimal) / (maximal - minimal + 0.01)]
        plt.scatter(feature_one, feature_two, c=[color_function(i) for i in df.index])

    def save_graphs(self):
        os.makedirs(self.directory, exist_ok=True)
        filename = f'/coverage_{time.ctime()}'.replace(' ', '_').replace(':', '-')
        self.generated_files['graphs'].append(filename)
        plt.savefig(self.directory + filename)

    def save_data(self, df):
        os.makedirs(self.directory, exist_ok=True)
        filename = f'/generated_data{time.ctime()}.csv'.replace(' ', '_').replace(':', '-')
        self.generated_files['data'].append(filename)
        df.to_csv(self.directory + filename)

