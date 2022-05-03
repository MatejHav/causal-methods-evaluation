import pandas as pd
import numpy as np

from builders import Experiment


def main():
    dimensions = 5
    sample_size = 3000
    Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=1, number_of_trees=500) \
        .add_causal_forest(min_leaf_size=1, number_of_trees=500) \
        .add_mean_squared_error() \
        .add_absolute_error() \
        .add_biased_generator(dimensions, sample_size=sample_size) \
        .run(save_data=True, save_graphs=True, show_graphs=False) \
        .test_specific_set(
        pd.DataFrame(np.zeros((40, 5)), columns=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']),
        pd.DataFrame(np.zeros((40, 1)) + 0.1, columns=['outcome']))


if __name__ == '__main__':
    main()
