import numpy as np
import pandas as pd
from typing import *
from causal_effect_methods import *
from data_generator import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def run(methods: Dict[str, CausalMethod],
        score_functions: Dict[str, Callable[[List[float], List[float]], float]],
        data_generator: Generator=None, data_file: str=None, samples: int=500):
    assert data_generator is not None or data_file is not None, "Data must be either generated or read from a file."
    scoring_list = [score_functions[key] for key in score_functions]
    columns = [key for key in score_functions.keys()]
    columns.insert(0, 'method_name')
    X, y, W, true_effect = None, None, None, None
    if data_generator is not None:
        X, y, W, true_effect = load_data_from_generator(data_generator, samples)
    elif data_file is not None:
        X, y, W, true_effect = load_data_from_file(data_file)
    X = X.join(W)
    X = X.join(y)
    df = pd.DataFrame([], columns=columns)
    # df['method_name'] = df['method_name'].astype(str)
    for method in methods:
        model = methods[method]
        results = run_model(model, scoring_list, X, true_effect)
        results.insert(0, method)
        df.loc[len(df.index)] = results
    return df


def run_model(model: CausalMethod, score_functions: List[Callable[[List[float], List[float]], float]],
              feature_data: List[List[float]], true_effect: List[float]):
    X_train, X_test, y_train, y_test = train_test_split(feature_data, true_effect, test_size=0.25, random_state=42)
    dimensions = len(X_train.columns) - 3
    # For training I only want to see the overall outcome
    # So I use overall outcome for training rather than the true effect
    model.train(select_features(X_train, dimensions), X_train['outcome'], X_train['treatment'])

    # I want to estimate the effect of treatment on the outcome, so I have to test against the real effect
    # (so no outside influences other than treatment)
    results = model.estimate_causal_effect(select_features(X_test, dimensions))
    return [score_function(y_test, results) for score_function in score_functions]


if __name__ == '__main__':
    forest = CausalForest(2000, k=1)
    main_effect = lambda x: 0
    # main_effect = lambda x: 2 * x[0] - 1
    # treatment_effect = lambda x: 0
    treatment_effect = lambda x: (1 + 1/(1 + np.exp(-20 * (x[0] - 1/3)))) * (1 + 1/(1 + np.exp(-20 * (x[1] - 1/3))))
    # treatment_effect = lambda x: np.random.random()
    treatment_propensity = lambda x: 0.5
    # treatment_propensity = lambda x: (1 + beta.pdf(x[0], 2, 4)) / 4
    noise = lambda: 0.0 * np.random.normal(0, 1)
    dimensions = 2
    distributions = [lambda: np.random.random()]
    sample_generator = data_generator.Generator(main_effect, treatment_effect, treatment_propensity, noise, dimensions,
                                                distributions)
    print(run({'causal_forest': forest}, {'mse': mean_squared_error}, data_generator=sample_generator, samples=2500))