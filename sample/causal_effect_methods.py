import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K
from econml.dml import CausalForestDML as EconCausalForest
from abc import abstractmethod, ABC

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
from keras.optimizer_v1 import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from load_data import *
from data_generator import select_features
from sample.other_methods.dragonnet.experiment.ihdp_main import make_dragonnet
from keras.losses import *
from keras.metrics import *

from sample.other_methods.dragonnet.experiment.models import regression_loss, binary_classification_loss, \
    treatment_accuracy, track_epsilon, dragonnet_loss_binarycross


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

class DragonNet(CausalMethod):

    # Not sure what reg_l2 is but I base it on DragonNet implementation
    def __init__(self, dimensions, reg_l2=0.01):
        self.dragonnet = make_dragonnet(dimensions, reg_l2)

    def train(self, x, y, w):
        metrics = [mean_squared_error]

        self.dragonnet.compile(
            optimizer=Adam(lr=1e-3),
            loss=mean_squared_error, metrics=metrics)

        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0, mode='auto',
                              min_delta=1e-8, cooldown=0, min_lr=0)

        ]

        self.dragonnet.fit(x=x, y=y, callbacks=adam_callbacks,
                      validation_split=0.2,
                      epochs=100,
                      batch_size=64, verbose=0)

    def estimate_causal_effect(self, x):
        return self.dragonnet.predict(x)


if __name__ == '__main__':
    main_effect = lambda x: 2 * x[0] - 1
    treatment_effect = lambda x: (1 + 1 / (1 + np.exp(-20 * (x[0] - 1 / 3)))) * (
            1 + 1 / (1 + np.exp(-20 * (x[1] - 1 / 3))))
    treatment_propensity = lambda x: (1 + beta.pdf(x[0], 2, 4)) / 4
    noise = lambda: 0.05 * np.random.normal(0, 1)
    dimensions = 5
    distributions = [lambda: np.random.random()]
    sample_generator = data_generator.Generator(main_effect, treatment_effect, treatment_propensity, noise, dimensions,
                                                distributions)
    # X, W, y, true_effect = load_data_from_file('data/full_gen/generated_dataTue_Apr_26_14-38-31_2022.csv')
    X, W, y, main_effect, true_effect, propensity = load_data_from_generator(sample_generator)
    method = DragonNet(dimensions)
    dimensions = len(X.columns)
    X = X.join(W)
    X = X.join(y)
    base_truth = pd.DataFrame(y).join(main_effect)
    base_truth = base_truth.join(true_effect)
    base_truth = base_truth.join(propensity)
    X_train, X_test, y_train, y_test = train_test_split(X, base_truth, test_size=0.25, random_state=42)
    # For training I only want to see the overall outcome
    # So I use overall outcome for training rather than the true effect
    method.train(select_features(X_train, dimensions), y_train, X_train['treatment'])

    # I want to estimate the effect of treatment on the outcome, so I have to test against the real effect
    # (so no outside influences other than treatment)
    results = method.estimate_causal_effect(select_features(X_test, dimensions))
    score = 0
    convert = {0: 'outcome', 1: 'main_effect', 2:'treatment_effect', 3:'propensity'}
    for i in range(len(results)):
        for j in range(4):
            score += (y_test.iloc[i][convert[j]] - results[i][j]) ** 2
    score = score / (4 * len(results))
    print(f'MSE: {score}')

