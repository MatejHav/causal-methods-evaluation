"""Loading of Data

This file defines functions necessary for loading data.

This file can also be imported as a module and contains the following
functions:

    * load_data_from_generator : functions that generates data based on the inputs
    * load_data_from_file : reads a file and loads data from there
"""

import pandas as pd
import data_generator

def load_data_from_generator(generator: data_generator.Generator, samples=500):
    return generator.generate_data(samples, save_data=True)


def load_data_from_file(csv_file: str):
    df = pd.read_csv(csv_file)
    dimensions = sum([1 for name in df.columns if 'feature' in name])
    return data_generator.select_features(df, dimensions), df['treatment'], df['outcome'], df['main_effect'],\
           df['treatment_effect'], df['propensity'], df['y0'], df['y1'], df['noise'], df['cate']
