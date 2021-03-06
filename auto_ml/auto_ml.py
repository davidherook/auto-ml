import os
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from auto_ml.plots import plot_pred_vs_actual, plot_residual, plot_nn_history, plot_nn_accuracy
from auto_ml.util import generate_hash, load_pickle, save_pickle, save_yaml, load_yaml, save_model_h5, load_model_h5

MODEL_DIR = 'model'
OUTPUT_DIR = 'output'

def get_model_type(config: dict) -> str:
    valid_models = ['linear_regression', 'random_forest_regression',
    'logistic_regression', 'random_forest_classifier', 'neural_net']
    for i in valid_models:
        if i in config.keys():
            return i

def build_model(nn_layers, nn_optimizer, input_dim=None):
    nn = Sequential()
    for layer in nn_layers:
        nn.add(
            Dense(layer['nodes'],
                activation=layer['activation']))
    opt = Adam(lr=nn_optimizer['learning_rate'])
    nn.compile(
        loss=nn_optimizer['loss'],
        optimizer=opt,
        metrics=nn_optimizer['metrics'])
    return nn


class AutoML(object):

    def __init__(self, model=None, config=None):

        if model is not None:
            config_path = os.path.join(MODEL_DIR, model, 'config.yaml')
            self.config = load_yaml(config_path)
            self.features = self.config['features']
            self.target = self.config['target']

            self.active_model = get_model_type(self.config)
            if self.active_model == 'neural_net':
                model_path = os.path.join(MODEL_DIR, model, '{}.h5'.format(model))
                self.model = load_model_h5(model_path) 
            else:
                model_path = os.path.join(MODEL_DIR, model, '{}.pk'.format(model))
                self.model = load_pickle(model_path) 

        if config is not None:
            self.model_hash = generate_hash()
            self.config = config
            self.features = config['features']
            self.target = config['target']

            self.active_model = get_model_type(self.config)
            if self.active_model == 'neural_net':
                self.nn_layers = config['neural_net']['architecture']
                self.nn_optimizer = config['neural_net']['optimizer']
                self.training = config['neural_net']['training']
            else:
                self.model_params = config[self.active_model]

    def train(self, data, save=True):
        X, y = data[self.features], data[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=7)
        print('Training on {}, Testing on {}'.format(self.X_train.shape[0], self.X_test.shape[0]))

        if self.active_model == 'neural_net':
            self.train_nn()
        else: 
            self.train_scikit()

        if save:
            self.save_model_artifacts(self.model, self.config)

    def train_nn(self):
        self.model = build_model(self.nn_layers, self.nn_optimizer)
        self.history = self.model.fit(self.X_train, 
            self.y_train,
            epochs=self.training['epochs'],
            batch_size=self.training['batch_size'],
            validation_split=self.training['validation_split'],
            verbose=True)

    def train_scikit(self):
        if self.active_model == 'random_forest_regression':
            self.model = RandomForestRegressor(**self.model_params)
        elif self.active_model == 'linear_regression':
            self.model = LinearRegression(**self.model_params)
        elif self.active_model == 'random_forest_classifier':
            self.model = RandomForestClassifier(**self.model_params)
        elif self.active_model == 'logistic_regression':
            self.model = LogisticRegression(**self.model_params)

        print('Fitting {}...'.format(self.model))
        self.model.fit(self.X_train, self.y_train)

    def save_model_artifacts(self, model, config):
        model_dir = os.path.join(MODEL_DIR, self.model_hash)
        os.mkdir(model_dir)

        config_path = os.path.join(model_dir, 'config.yaml')

        if self.active_model == 'neural_net':
            model_path = os.path.join(model_dir, '{}.h5'.format(self.model_hash))
            save_model_h5(model, model_path)
        else:
            model_path = os.path.join(model_dir, '{}.pk'.format(self.model_hash))
            save_pickle(model, model_path)

        save_yaml(config, config_path)
        print('Saved model artifacts')


class AutoMLRegressor(AutoML):

    def __init__(self, model=None, config=None):
        super().__init__(model, config)

    def predict(self, X_test):
        print('Predicting {} instances...'.format(X_test.shape[0]))
        X_test = X_test[self.features]
        if self.active_model == 'neural_net':
            return self.model.predict(X_test)[:,0]
        else:
            return self.model.predict(X_test)
    
    def save_model_output(self, df_test):
        output_dir = os.path.join(OUTPUT_DIR, self.model_hash)
        os.mkdir(output_dir)

        df_test_path = os.path.join(output_dir, 'test_data.csv')
        metrics_path = os.path.join(output_dir, 'metrics.json')
        scatter_path = os.path.join(output_dir, 'pred_vs_actual.png')
        residual_path = os.path.join(output_dir, 'residual.png')

        df_test.to_csv(df_test_path)
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        if self.active_model == 'neural_net':
            history_path = os.path.join(output_dir, 'history.png')
            accuracy_path = os.path.join(output_dir, 'accuracy.png')
            plot_nn_history(self.history, show=True, save_to=history_path)
            plot_nn_accuracy(self.history, show=True, save_to=accuracy_path)

        plot_pred_vs_actual(df_test[self.target], df_test[self.target + '_pred'], save_to=scatter_path)
        plot_residual(df_test[self.target], df_test[self.target + '_pred'], save_to=residual_path)
        print('Saved model output')

    def evaluate(self, save=True):
        def mean_abs_pct_error(y_test, y_pred):
            return np.mean( np.abs( (y_test - y_pred) / y_test ) ) * 100

        y_pred = self.predict(self.X_test)

        target_mean, target_std = np.mean(self.y_test), np.std(self.y_test)
        tst_R = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        maep = mean_abs_pct_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        err_std = (self.y_test - y_pred).std()

        self.metrics = {
            "test_size": self.y_test.shape[0],
            "target_mean": target_mean,
            "target_std": target_std,
            "validation": {
                "Test R-sq":tst_R,
                "MAE": mae,
                "MAEP": maep,
                "RMSE": rmse,
                "Error Std": err_std
            }
        }

        print(json.dumps(self.metrics, indent=4))
        
        X_test_copy = self.X_test.copy()
        X_test_copy[self.target] = self.y_test
        X_test_copy[self.target + '_pred'] = y_pred

        if save:
            self.save_model_output(X_test_copy)


class AutoMLClassifier(AutoML):

    def __init__(self, model=None, config=None):
        super().__init__(model, config)

    def predict(self, X_test):
        print('Predicting {} instances...'.format(X_test.shape[0]))
        X_test = X_test[self.features]
        if self.active_model == 'neural_net':
            return self.model.predict_classes(X_test)[:,0]
        else:
            return self.model.predict(X_test)
            # predict proba

    def pretty_confusion_matrix(self, y_true, y_pred):
        cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
        cm.columns = ['Predicted {}'.format(c) for c in cm.columns]
        cm.index = ['Actual {}'.format(c) for c in cm.index]
        return cm

    def evaluate(self, save=True):
        y_pred = self.predict(self.X_test)
        print(y_pred)
        target_mean, target_std = np.mean(self.y_test), np.std(self.y_test)

        print(self.pretty_confusion_matrix(self.y_test, y_pred))

        self.metrics = {
            "test_size": self.y_test.shape[0],
            "target_mean": target_mean,
            "target_std": target_std,
            "validation": {
                "Accuracy": accuracy_score(self.y_test, y_pred),
                "Precision": precision_score(self.y_test, y_pred),
                "Recall": recall_score(self.y_test, y_pred),
                "F1": f1_score(self.y_test, y_pred)
            }
        }

        print(json.dumps(self.metrics, indent=4))
        X_test_copy = self.X_test.copy()
        X_test_copy[self.target] = self.y_test
        X_test_copy[self.target + '_pred'] = y_pred
        
        if save:
            self.save_model_output(X_test_copy)

    def save_model_output(self, df_test):
        output_dir = os.path.join(OUTPUT_DIR, self.model_hash)
        os.mkdir(output_dir)

        df_test_path = os.path.join(output_dir, 'test_data.csv')
        metrics_path = os.path.join(output_dir, 'metrics.json')

        df_test.to_csv(df_test_path)
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        if self.active_model == 'neural_net':
            history_path = os.path.join(output_dir, 'history.png')
            accuracy_path = os.path.join(output_dir, 'accuracy.png')
            plot_nn_history(self.history, show=True, save_to=history_path)
            plot_nn_accuracy(self.history, show=True, save_to=accuracy_path)

        print('Saved model output')