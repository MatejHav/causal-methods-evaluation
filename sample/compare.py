from causal_effect_methods import *
from data_generator import *
from sklearn.model_selection import train_test_split
from utils import save_pandas_table, generate_coverage_of_model_graph


def run(methods: Dict[str, CausalMethod],
        score_functions: Dict[str, Callable[[List[float], List[float]], float]],
        data_generator: Generator = None, data_file: str = None, samples: int = 500, save_table: bool = False,
        dir: str = ''):
    assert data_generator is not None or data_file is not None, "Data must be either generated or read from a file."
    scoring_list = [score_functions[key] for key in score_functions]
    columns = [key for key in score_functions.keys()]
    columns.insert(0, 'method_name')
    X, W, y, main_effect, true_effect, propensity, y0, y1, noise, cate = None, None, None, None, None,\
                                                                         None, None, None, None, None
    if data_generator is not None:
        X, W, y, main_effect, true_effect, propensity, y0, y1, noise, cate = load_data_from_generator(data_generator,
                                                                                                      samples)
    elif data_file is not None:
        X, W, y, main_effect, true_effect, propensity, y0, y1, noise, cate = load_data_from_file(data_file)
    df = pd.DataFrame([], columns=columns)
    all_data = X.join([W, y, main_effect, true_effect, propensity, y0, y1, noise, cate])
    for method in methods:
        model = methods[method]
        results = run_model(model, scoring_list, X, W, y, main_effect, true_effect, propensity, y0, y1, noise, cate,
                            save_table=save_table, dir=dir)
        results.insert(0, method)
        df.loc[len(df.index)] = results
        if save_table:
            generate_coverage_of_model_graph(model, all_data, dir + f'/model_coverage_{model}.png')
    df = df.set_index('method_name')
    if save_table:
        save_pandas_table(dir + '/inter_table', df)
    return df

def run_model(model: CausalMethod, score_functions: List[Callable[[List[float], List[float]], float]],
              feature_data: pd.DataFrame, treatment: pd.DataFrame, outcome: pd.DataFrame, main_effect: pd.DataFrame,
              treatment_effect: pd.DataFrame, treatment_propensity: pd.DataFrame,
              y0: pd.DataFrame, y1: pd.DataFrame, noise: pd.DataFrame, cate: pd.DataFrame,
              save_table=False, dir=''):
    all_data = feature_data.join(treatment)
    all_data = all_data.join(outcome)
    all_data = all_data.join(main_effect)
    all_data = all_data.join(treatment_effect)
    all_data = all_data.join(treatment_propensity)
    all_data = all_data.join(y0)
    all_data = all_data.join(y1)
    all_data = all_data.join(noise)
    all_data = all_data.join(cate)
    # Ensure that X_train and X_test hold all values needed
    X_train, X_test, y_train, y_test = train_test_split(all_data,
                                                        model.create_training_truth(outcome, main_effect,
                                                                                    treatment_effect,
                                                                                    treatment_propensity,
                                                                                    y0, y1, noise, cate),
                                                        test_size=0.25, random_state=42)
    # Select only features for training
    model.train(select_features(X_train), y_train, X_train['treatment'])

    # Overwrite y_test based on the model prediction expectation
    y_test = model.create_testing_truth(X_test['outcome'], X_test['main_effect'], X_test['treatment_effect'],
                                        X_test['propensity'], X_test['y0'], X_test['y1'], X_test['noise'], X_test['cate'])

    # Select only features for testing
    results = model.estimate_causal_effect(select_features(X_test))
    if save_table:
        select_features(X_test).to_csv(dir + f'/testing_set_{model}.csv')
        y_test.to_csv(dir + f'/base_truth_for_testing_set_{model}.csv')
        save_pandas_table(dir + f'/table_predictions_{model}', pd.DataFrame(results, columns=[f'prediction_{i}' for i in
                                                                                              range(results.shape[
                                                                                                        1])] if len(
            results.shape) > 1 else ['prediction']))
    return [score_function(y_test.to_numpy(), results) for score_function in score_functions]
