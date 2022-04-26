from econml.dml import CausalForestDML as EconCausalForest
from abc import abstractmethod, ABC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from load_data import *
from data_generator import select_features

class CausalMethod(ABC):

    @abstractmethod
    def estimate_causal_effect(self, x):
        pass

    @abstractmethod
    def train(self, x, y, w):
        pass


class CausalForest(CausalMethod):

    def __init__(self, number_of_trees, method_effect='auto', method_predict='auto', k=1):
        self.forest = EconCausalForest(model_t=method_effect, model_y=method_predict, n_estimators=number_of_trees,
                                       min_samples_leaf=k, criterion='mse', random_state=42)

    def train(self, x, y, w):
        self.forest.fit(Y=y,
                        T=w,
                        X=x,
                        cache_values=True)

    def estimate_causal_effect(self, x):
        return self.forest.effect(x)

    def compute_cate(self, features):
        return self.forest.cate_feature_names(features)

if __name__ == '__main__':
    forest = CausalForest(20, k=10)
    main_effect = lambda x: 2 * x[0] - 1
    treatment_effect = lambda x: x[0]
    treatment_propensity = lambda x: (1 + beta.pdf(x[0], 2, 4)) / 4
    noise = lambda: 0.05 * np.random.normal(0, 1)
    dimensions = 5
    distributions = [lambda: np.random.random()]
    sample_generator = data_generator.Generator(main_effect, treatment_effect, treatment_propensity, noise, dimensions,
                                                distributions)
    X, W, y, true_effect = load_data_from_file('data/data_dump_130783695794/generated_dataTue_Apr_26_14-38-31_2022.csv')
    dimensions = len(X.columns) - 1
    X = X.join(W)
    X = X.join(y)
    X_train, X_test, y_train, y_test = train_test_split(X, true_effect, test_size=0.25, random_state=42)
    # For training I only want to see the overall outcome
    # So I use overall outcome for training rather than the true effect
    forest.train(select_features(X_train, dimensions), X_train['outcome'], X_train['treatment'])

    # I want to estimate the effect of treatment on the outcome, so I have to test against the real effect
    # (so no outside influences other than treatment)
    results = forest.estimate_causal_effect(select_features(X_test, dimensions))
    score = mean_squared_error(y_test, results)
    print(score)

