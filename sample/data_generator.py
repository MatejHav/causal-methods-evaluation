import os
import time
import numpy as np
import pandas as pd
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

    def generate_data(self, number_of_samples: int, save_data: bool = True, show_graphs: bool = False,
                      save_graphs: bool = False):
        columns = [f'feature_{i}' for i in range(self.dimensions)]
        columns.append('treatment')
        columns.append('outcome')
        df = pd.DataFrame([], columns=columns)
        for i in range(number_of_samples):
            features, treatment, outcome = self.generate_row()
            features.append(treatment)
            features.append(outcome)
            df.loc[len(df.index)] = features
        if save_data:
            self.save_data(df)
        return df

    def generate_row(self):
        features = []
        for dimension in range(self.dimensions):
            features.append(self.generate_feature(dimension))
        # W = bernoulli(e(x))
        treatment = 1 if np.random.random() <= self.treatment_propensity(features) else 0
        # Y = m(x) + (W - 0.5) * t(x) + noise
        outcome = self.main_effect(features) + (treatment - 0.5) * self.treatment_effect(features) + self.noise()
        return features, treatment, outcome

    def generate_feature(self, index):
        if len(self.distributions) == 1:
            return self.distributions[0]()
        return self.distributions[index]()

    def save_data(self, df):
        directory = f'data/data_dump_{self.__hash__()}'
        if not os.path.exists(directory):
            os.mkdir(directory)
        df.to_csv(directory + f'/generated_data{time.ctime()}.csv'.replace(' ', '_').replace(':', '-'))


if __name__ == '__main__':
    samples = 500
    main_effect = lambda x: 2 * x[0] - 1
    treatment_effect = lambda x: 0
    treatment_propensity = lambda x: (1 + beta.pdf(x[0], 2, 4)) / 4
    noise = lambda: 0.05 * np.random.normal(0, 1)
    dimensions = 5
    distributions = [lambda: np.random.random()]
    sample_generator = Generator(main_effect, treatment_effect, treatment_propensity, noise, dimensions, distributions)
    sample_generator.generate_data(samples, save_data=True)

