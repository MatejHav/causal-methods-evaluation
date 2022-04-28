import numpy as np
import pandas as pd
import data_generator
from scipy.stats import beta
from typing import *


def load_data_from_generator(generator: data_generator.Generator, samples=500):
    return generator.generate_data(500, save_data=True)


def load_data_from_file(csv_file: str):
    df = pd.read_csv(csv_file)
    dimensions = len(df.columns) - 6
    return data_generator.select_features(df, dimensions), df['treatment'], df['outcome'], df['main_effect'], df['treatment_effect'], df['propensity']


if __name__ == '__main__':
    main_effect = lambda x: 2 * x[0] - 1
    treatment_effect = lambda x: 0
    treatment_propensity = lambda x: (1 + beta.pdf(x[0], 2, 4)) / 4
    noise = lambda: 0.05 * np.random.normal(0, 1)
    dimensions = 5
    distributions = [lambda: np.random.random()]
    sample_generator = data_generator.Generator(main_effect, treatment_effect, treatment_propensity, noise, dimensions,
                                                distributions)
    print(load_data_from_generator(sample_generator))
    print(load_data_from_file('data/data_dump_81132812397/generated_dataTue_Apr_26_11-20-17_2022.csv'))
