"""Loading of Data

This file defines functions necessary for loading data.

This file can also be imported as a module and contains the following
functions:

    * load_data_from_generator : functions that generates data based on the inputs
    * load_data_from_file : reads a file and loads data from there
"""

import numpy as np
import pandas as pd
import data_generator

def load_data_from_generator(generator: data_generator.Generator, samples=500):
    return generator.generate_data(samples, save_data=True)


def load_data_from_file(csv_file: str):
    if 'ihdp' in csv_file:
        return load_ihdp(csv_file)
    df = pd.read_csv(csv_file)
    dimensions = sum([1 for name in df.columns if 'feature' in name])
    return data_generator.select_features(df, dimensions), df['treatment'], df['outcome'], df['main_effect'],\
           df['treatment_effect'], df['propensity'], df['y0'], df['y1'], df['noise'], df['cate']

# Taken from https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP
def load_ihdp(file: str):
    array = np.genfromtxt(file, delimiter=',')
    n = array.shape[0]
    # columns: treatment, y_factual, y_cfactual, mu0, mu1, x1, …, x25
    treatment = array[:, 0]
    y0 = np.array([array[i, 1 if treatment[i] == 0 else 2] for i in range(n)])
    y1 = np.array([array[i, 1 if treatment[i] == 1 else 2] for i in range(n)])
    cate = array[:, 4] - array[:, 3]
    features = pd.DataFrame(array[:, 5:], columns=[f'feature_{i}' for i in range(array.shape[1] - 5)])
    treatment = pd.DataFrame(treatment, columns=['treatment'])
    outcome = pd.DataFrame(array[:, 1], columns=['outcome'])
    main_effect = pd.DataFrame(np.zeros((n, 1)), columns=['main_effect'])
    treatment_effect = pd.DataFrame(np.zeros((n, 1)), columns=['treatment_effect'])
    propensity = pd.DataFrame(np.zeros((n, 1)), columns=['propensity'])
    y0 = pd.DataFrame(y0, columns=['y0'])
    y1 = pd.DataFrame(y1, columns=['y1'])
    noise = pd.DataFrame(np.zeros((n, 1)), columns=['noise'])
    cate = pd.DataFrame(cate, columns=['cate'])
    return features, treatment, outcome, main_effect, treatment_effect, propensity, y0, y1, noise, cate


