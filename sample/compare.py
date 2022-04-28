import numpy as np
import pandas as pd
from typing import *
from causal_effect_methods import *
from data_generator import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table


def run(methods: Dict[str, CausalMethod],
        score_functions: Dict[str, Callable[[List[float], List[float]], float]],
        data_generator: Generator = None, data_file: str = None, samples: int = 500, save_table: bool=False, dir: str=''):
    assert data_generator is not None or data_file is not None, "Data must be either generated or read from a file."
    scoring_list = [score_functions[key] for key in score_functions]
    columns = [key for key in score_functions.keys()]
    columns.insert(0, 'method_name')
    X, y, W, main_effect, true_effect, propensity = None, None, None, None, None, None
    if data_generator is not None:
        X, y, W, main_effect, true_effect, propensity = load_data_from_generator(data_generator, samples)
    elif data_file is not None:
        X, y, W, main_effect, true_effect, propensity = load_data_from_file(data_file)
    X = X.join(W)
    X = X.join(y)
    df = pd.DataFrame([], columns=columns)
    for method in methods:
        model = methods[method]
        results = run_model(model, scoring_list, X, y, main_effect, true_effect, propensity)
        results.insert(0, method)
        df.loc[len(df.index)] = results
    if save_table:
        save_pandas_table(dir, df)
    return df

def save_pandas_table(dir, df):
    plt.clf()
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table(ax, df)
    plt.savefig(dir)
    df.to_csv(dir + '.csv')

def run_model(model: CausalMethod, score_functions: List[Callable[[List[float], List[float]], float]],
              feature_data: List[List[float]], outcome: List[float], main_effect: List[float],
              treatment_effect: List[float], treatment_propensity: List[float]):
    X_train, X_test, y_train, y_test = train_test_split(feature_data,
                                                        model.create_base_truth(outcome, main_effect,
                                                                                treatment_effect, treatment_propensity),
                                                        test_size=0.25, random_state=42)
    dimensions = sum([1 for name in X_train.columns if 'feature' in name])
    # For training I only want to see the overall outcome
    # So I use overall outcome for training rather than the true effect
    model.train(select_features(X_train, dimensions), y_train, X_train['treatment'])

    # I want to estimate the effect of treatment on the outcome, so I have to test against the real effect
    # (so no outside influences other than treatment)
    results = model.estimate_causal_effect(select_features(X_test, dimensions))
    return [score_function(y_test.to_numpy(), results) for score_function in score_functions]