import os
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from auto_ml.plots import plot_pred_vs_actual, plot_residual, plot_nn_history, plot_nn_accuracy, plot_nn_histories
from auto_ml.util import *

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

            self.train_ratio = config['train_ratio']
            self.validation_ratio = config['validation_ratio']
            self.test_ratio = config['test_ratio']

            self.active_model = get_model_type(self.config)
            if self.active_model == 'neural_net':
                self.nn_layers = config['neural_net']['architecture']
                self.nn_optimizer = config['neural_net']['optimizer']
                self.training = config['neural_net']['training']
                self.histories = None
            else:
                self.model_params = config[self.active_model]

    def cross_val_train(self, data, k=4):
        """Returns a list of tensorflow.python.keras.callbacks.History objects"""
        scores = []
        histories = []

        X, y = data[self.features], data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

        # save test set for evaluation later
        self.X_test, self.y_test = X_test, y_test

        # Fold the data over the training set
        # This gives us training and validation set
        # The self.X_test is saved for evaluation after
        # Original dataset = 200,000
        # Train = 160,000
        # train, val = 120,000, 40,000
        for train, val in KFold(n_splits=k, random_state=7).split(X_train, y_train):

            print('Training on {}, validating on {}...'.format(len(train), len(val)))
            self.X_train, self.y_train = X_train.iloc[train], y_train.iloc[train]
            self.X_val, self.y_val = X_train.iloc[val], y_train.iloc[val]

            if self.active_model == 'neural_net':
                history = self.train_nn()
                histories.append(history)
            else: 
                self.train_scikit()

        return histories
            
    def train(self, data, save=True, cross_val=True):

        if cross_val: 
            self.histories = self.cross_val_train(data, k=4)
        
        else:
            X, y = data[self.features], data[self.target]

            splits = train_val_test_split(X, y,
                self.train_ratio, 
                self.validation_ratio, 
                self.test_ratio)

            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = splits
            print('Training on {}\nValidating on {}\nTesting on {}...'.format(self.X_train.shape[0], self.X_val.shape[0], self.X_test.shape[0]))

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
            validation_data=(self.X_val, self.y_val),
            verbose=False)
        return self.history

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
            history_path = os.path.join(output_dir, 'model_loss.png')
            accuracy_path = os.path.join(output_dir, 'model_accuracy.png')
            plot_nn_history(self.history, show=True, save_to=history_path)

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

    def evaluate(self, save=True):
        y_pred = self.predict(self.X_test)
        target_mean, target_std = np.mean(self.y_test), np.std(self.y_test)

        print(pretty_confusion_matrix(self.y_test, y_pred))

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
            history_path = os.path.join(output_dir, 'model_loss.png')
            accuracy_path = os.path.join(output_dir, 'model_accuracy.png')
            cross_val_path = os.path.join(output_dir, 'cross_val.png')
            # TODO reconcile with plot_nn_histories
            # right now it will show the last fold
            plot_nn_history(self.history, show=True, save_to=history_path)
            plot_nn_accuracy(self.history, show=True, save_to=accuracy_path)
            if self.histories is not None:  
                plot_nn_histories(self.histories, save_to=cross_val_path)

        print('Saved model output')