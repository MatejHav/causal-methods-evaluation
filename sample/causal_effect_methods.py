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

    @abstractmethod
    def create_training_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise):
        pass

    @abstractmethod
    def create_testing_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise):
        pass

    @abstractmethod
    def __str__(self):
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

    def create_training_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise):
        return outcome

    def create_testing_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise):
        return treatment_effect

    def __str__(self):
        return 'causal_forest'

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
        results = self.dragonnet.predict(x)
        return results

    def create_training_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise):
        base_truth = pd.DataFrame(y0).join(y1)
        base_truth = base_truth.join(treatment_propensity)
        base_truth = base_truth.join(noise)
        return base_truth

    def create_testing_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise):
        return self.create_training_truth(outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise)

    def __str__(self):
        return 'dragonnet'


