import os
import json
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from auto_ml.plots import plot_pred_vs_actual, plot_nn_history
from auto_ml.util import generate_hash, load_pickle, save_pickle, \
    save_yaml, load_yaml, save_model_h5, load_model_h5

MODEL_DIR = 'model'
OUTPUT_DIR = 'output'

def build_model(nn_layers, nn_optimizer, input_dim=None):
    nn = Sequential()
    for layer in nn_layers:
        nn.add(
            Dense(layer['nodes'],
                activation=layer['activation']))
    opt = Adam(lr=0.01)
    nn.compile(
        loss=nn_optimizer['loss'],
        optimizer=opt,
        metrics=["mae", "mse"])
    return nn

# TODO
# def build_model_scikit():
#     return RandomForestRegressor()

class AutoML(object):

    def __init__(self, model=None, config=None):

        if model is not None:
            # if scikit:
            # model_path = os.path.join(MODEL_DIR, model, '{}.pk'.format(model))
            # self.model = load_pickle(model_path) 
            model_path = os.path.join(MODEL_DIR, model, '{}.h5'.format(model))
            self.model = load_model_h5(model_path) 
            config_path = os.path.join(MODEL_DIR, model, 'config.yaml')
            self.config = load_yaml(config_path)
            self.features = self.config['features']
            self.target = self.config['target']

        if config is not None:
            self.model_hash = generate_hash()
            self.config = config
            self.features = config['features']
            self.target = config['target']
            self.nn_layers = config['architecture']
            self.nn_optimizer = config['optimizer']
            self.training = config['training']

    def predict(self, X_test):
        print('Predicting {} instances...'.format(X_test.shape[0]))
        X_test = X_test[self.features]
        # if scikit:
        # return self.model.predict(X_test)
        return self.model.predict(X_test)[:,0]

    def save_model_artifacts(self, model, config):
        model_dir = os.path.join(MODEL_DIR, self.model_hash)
        os.mkdir(model_dir)

        model_path = os.path.join(model_dir, '{}.h5'.format(self.model_hash))
        # if scikit:
        # model_path = os.path.join(model_dir, '{}.pk'.format(self.model_hash))

        config_path = os.path.join(model_dir, 'config.yaml')

        # if scikit:
        # save_pickle(model, model_path)
        save_model_h5(model, model_path)
        save_yaml(config, config_path)
        print('Saved model artifacts')
    
    def save_model_output(self, df_test):
        output_dir = os.path.join(OUTPUT_DIR, self.model_hash)
        os.mkdir(output_dir)

        df_test_path = os.path.join(output_dir, 'test_data.csv')
        metrics_path = os.path.join(output_dir, 'metrics.json')
        history_path = os.path.join(output_dir, 'history.png')
        scatter_path = os.path.join(output_dir, 'pred_vs_actual.png')

        df_test.to_csv(df_test_path)
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        plot_nn_history(self.history, show=True, save_to=history_path)
        plot_pred_vs_actual(df_test['y'], df_test['y_pred'], save_to=scatter_path)
        print('Saved model output')

    def train(self, data):
        X, y = data[self.features], data[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=7)
        print('\n\nTraining on {}, Testing on {}'.format(self.X_train.shape[0], self.X_test.shape[0]))

        self.model = build_model(self.nn_layers, self.nn_optimizer)
        self.history = self.model.fit(self.X_train, 
            self.y_train,
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            validation_split=0.2,
            verbose=True)
        # if scikit:
        # self.model.fit(self.X_train, self.y_train)

        self.save_model_artifacts(self.model, self.config)

    def evaluate(self):
        def mean_abs_pct_error(y_test, y_pred):
            return np.mean( np.abs( (y_test - y_pred) / y_test ) ) * 100

        y_pred = self.model.predict(self.X_test)[:,0]
        # if scikit: 
        # y_pred = self.model.predict(self.X_test)

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
        X_test_copy['y'] = self.y_test
        X_test_copy['y_pred'] = y_pred
        self.save_model_output(X_test_copy)