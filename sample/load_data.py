import numpy as np
import pandas as pd
import data_generator
from scipy.stats import beta
from typing import *


def load_data_from_generator(generator: data_generator.Generator, samples=500):
    return generator.generate_data(500, save_data=True)


def load_data_from_file(csv_file: str):
    df = pd.read_csv(csv_file)
    dimensions = sum([1 for name in df.columns if 'feature' in name])
    return data_generator.select_features(df, dimensions), df['treatment'], df['outcome'], df['main_effect'],\
           df['treatment_effect'], df['propensity'], df['y0'], df['y1'], df['noise'], df['cate']
