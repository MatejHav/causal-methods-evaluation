import os
import pandas as pd
import numpy as np
import time

# Disable TesnorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from experiment import Experiment
from session import Session
from parameterizer import Parameterizer


def main():
    t = time.time_ns()
    print('STARTING...')
    basic_experiment()
    print(f'FINISHED IN {(time.time_ns() - t) * 1e-9} SECONDS.')


def parameterize_sample_size_biased():
    dimensions = 5
    sample_sizes = [{'sample_size': 50},
                    {'sample_size': 100},
                    {'sample_size': 250},
                    {'sample_size': 500},
                    {'sample_size': 750},
                    {'sample_size': 1000},
                    {'sample_size': 1250},
                    {'sample_size': 1500},
                    {'sample_size': 1750},
                    {'sample_size': 2000}
                    ]
    param_function = lambda d: lambda: Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=1, number_of_trees=500) \
        .add_causal_forest(min_leaf_size=1, number_of_trees=500) \
        .add_mean_squared_error() \
        .add_biased_generator(dimensions=dimensions, sample_size=d['sample_size'])
    Parameterizer(param_function, sample_sizes, name='sample_size_biased_general').run(save_graphs=True)
    Parameterizer(param_function, sample_sizes, name='sample_size_biased_specific').run_specific(
        pd.DataFrame(np.zeros((10, 5)), columns=[f'feature_{i}' for i in range(dimensions)]),
        pd.DataFrame(np.zeros((10, 1)) + 0.1, columns=['outcome']))


def parameterize_sample_size_general():
    dimensions = 5
    sample_sizes = [{'sample_size': 50},
                    {'sample_size': 100},
                    {'sample_size': 250},
                    {'sample_size': 500},
                    {'sample_size': 750},
                    {'sample_size': 1000},
                    {'sample_size': 1250},
                    {'sample_size': 1500},
                    {'sample_size': 1750},
                    {'sample_size': 2000}
                    ]
    param_function = lambda d: lambda: Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=1, number_of_trees=500) \
        .add_causal_forest(min_leaf_size=1, number_of_trees=500) \
        .add_mean_squared_error() \
        .add_all_effects_generator(dimensions=dimensions, sample_size=d['sample_size'])
    Parameterizer(param_function, sample_sizes, name='sample_size_normal_general').run(save_graphs=True)


def parameterize_number_of_trees():
    dimensions = 5
    tree_numbers = [{'number_of_trees': 2 ** 4},
                    {'number_of_trees': 2 ** 6},
                    {'number_of_trees': 2 ** 8},
                    {'number_of_trees': 2 ** 9},
                    {'number_of_trees': 2 ** 10},
                    {'number_of_trees': 2 ** 11}
                    ]
    param_function = lambda d: lambda: Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=1, number_of_trees=d['number_of_trees']) \
        .add_causal_forest(min_leaf_size=1, number_of_trees=d['number_of_trees']) \
        .add_mean_squared_error() \
        .add_biased_generator(dimensions=dimensions, sample_size=500)
    Parameterizer(param_function, tree_numbers, name='number_of_trees_biased_general').run()
    Parameterizer(param_function, tree_numbers, name='number_of_trees_biased_specific').run_specific(
        pd.DataFrame(np.zeros((40, 5)), columns=[f'feature_{i}' for i in range(dimensions)]),
        pd.DataFrame(np.zeros((40, 1)) + 0.1, columns=['outcome']))


def parameterize_leaf_size():
    dimensions = 5
    leaf_size = [{'min_leaf_size': 1},
                 {'min_leaf_size': 10},
                 {'min_leaf_size': 20},
                 {'min_leaf_size': 50},
                 {'min_leaf_size': 75},
                 {'min_leaf_size': 100}
                 ]
    param_function = lambda d: lambda: Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=d['min_leaf_size'], number_of_trees=500) \
        .add_causal_forest(min_leaf_size=d['min_leaf_size'], number_of_trees=500) \
        .add_mean_squared_error() \
        .add_biased_generator(dimensions=dimensions, sample_size=500)
    Parameterizer(param_function, leaf_size, name='leaf_size_biased_general').run(save_graphs=True)
    Parameterizer(param_function, leaf_size, name='leaf_size_biased_specific').run_specific(
        pd.DataFrame(np.zeros((40, 5)), columns=[f'feature_{i}' for i in range(dimensions)]),
        pd.DataFrame(np.zeros((40, 1)) + 0.1, columns=['outcome']), save_graphs=True)


def parameterize_specific_spiked_sample_size():
    dimensions = 5
    sample_sizes = [{'sample_size': 1000},
                    {'sample_size': 1250},
                    {'sample_size': 1500},
                    {'sample_size': 1750},
                    {'sample_size': 2000},
                    {'sample_size': 2250},
                    {'sample_size': 2500},
                    {'sample_size': 2750},
                    {'sample_size': 3000}
                    ]
    param_function = lambda d: lambda: Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=1, number_of_trees=500) \
        .add_causal_forest(min_leaf_size=1, number_of_trees=500) \
        .add_mean_squared_error() \
        .add_spiked_generator(dimensions=dimensions, sample_size=d['sample_size'])
    Parameterizer(param_function, sample_sizes, name='sample_size_spiked_general').run(save_graphs=True)
    Parameterizer(param_function, sample_sizes, name='sample_size_spiked_specific').run_specific(
        pd.DataFrame(np.ones((40, 5)) / 2, columns=[f'feature_{i}' for i in range(dimensions)]),
        pd.DataFrame(np.zeros((40, 1)) + 15.915494309189528, columns=['outcome']), save_graphs=True)


def basic_session():
    dimensions = 5
    get_experiment = lambda: Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=1, number_of_trees=500) \
        .add_causal_forest(min_leaf_size=1, number_of_trees=500) \
        .add_mean_squared_error() \
        .add_biased_generator(dimensions)
    Session(get_experiment, 'basic_session').run_specific(
        pd.DataFrame(np.zeros((40, 5)), columns=[f'feature_{i}' for i in range(dimensions)]),
        pd.DataFrame(np.zeros((40, 1)) + 0.1, columns=['outcome']))


def basic_experiment():
    dimensions = 5
    sample_size = 50
    Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=1, number_of_trees=500) \
        .add_causal_forest(min_leaf_size=1, number_of_trees=500) \
        .add_mean_squared_error() \
        .add_biased_generator(dimensions=dimensions, sample_size=sample_size) \
        .run(save_data=True, save_graphs=True, show_graphs=False)


def spiked_experiment():
    dimensions = 5
    sample_size = 5000
    Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=10, number_of_trees=500) \
        .add_causal_forest(min_leaf_size=10, number_of_trees=500) \
        .add_mean_squared_error() \
        .add_spiked_generator(dimensions, sample_size) \
        .run(save_data=True, save_graphs=True, show_graphs=False)


if __name__ == '__main__':
    main()
