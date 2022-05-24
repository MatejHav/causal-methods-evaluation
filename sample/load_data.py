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
        return load_ihdp()
    df = pd.read_csv(csv_file)
    dimensions = sum([1 for name in df.columns if 'feature' in name])
    return data_generator.select_features(df, dimensions), df['treatment'], df['outcome'], df['main_effect'],\
           df['treatment_effect'], df['propensity'], df['y0'], df['y1'], df['noise'], df['cate']

# Taken from https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP
def load_ihdp():
    index = 42 #np.random.randint(0, 999)
    train = np.load('datasets/ihdp/ihdp_npci_1-1000.train.npz')
    test = np.load('datasets/ihdp/ihdp_npci_1-1000.test.npz')
    n = len(train['x']) + len(test['x'])
    # columns: ['ate', 'mu1', 'mu0', 'yadd', 'yf', 'ycf', 't', 'x', 'ymul']
    treatment = np.concatenate((train['t'][:, index], test['t'][:, index]))
    mu1 = np.concatenate((train['mu1'][:, index], test['mu1'][:, index]))
    mu0 = np.concatenate((train['mu0'][:, index], test['mu0'][:, index]))
    yf = np.concatenate((train['yf'][:, index], test['yf'][:, index]))
    ycf = np.concatenate((train['ycf'][:, index], test['ycf'][:, index]))
    x = np.concatenate((train['x'][:, :, index], test['x'][:, :, index]))
    train.close()
    test.close()
    y0 = np.array([yf[i] if treatment[i] == 0 else ycf[i] for i in range(n)])
    y1 = np.array([yf[i] if treatment[i] == 1 else ycf[i] for i in range(n)])
    cate = mu1 - mu0
    features = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(25)])
    treatment = pd.DataFrame(treatment, columns=['treatment'])
    outcome = pd.DataFrame(yf, columns=['outcome'])
    main_effect = pd.DataFrame(np.zeros((n, 1)), columns=['main_effect'])
    treatment_effect = pd.DataFrame(np.zeros((n, 1)), columns=['treatment_effect'])
    propensity = pd.DataFrame(np.zeros((n, 1)), columns=['propensity'])
    y0 = pd.DataFrame(y0, columns=['y0'])
    y1 = pd.DataFrame(y1, columns=['y1'])
    noise = pd.DataFrame(np.zeros((n, 1)), columns=['noise'])
    cate = pd.DataFrame(cate, columns=['cate'])
    return features, treatment, outcome, main_effect, treatment_effect, propensity, y0, y1, noise, cate


