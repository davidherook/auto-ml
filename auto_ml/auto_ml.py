import os
import json
import yaml
import numpy as np
from auto_ml.util import generate_hash, load_pickle, save_pickle, save_yaml, load_yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

MODEL_DIR = 'model'
OUTPUT_DIR = 'output'

class AutoML(object):

    def __init__(self, model=None, config=None):

        if model is not None:
            model_path = os.path.join(MODEL_DIR, model, '{}.pk'.format(model))
            self.model = load_pickle(model_path) 

            config_path = os.path.join(MODEL_DIR, model, 'config.yaml')
            self.config = load_yaml(config_path)
            self.features = self.config['features']
            self.target = self.config['target']

        if config is not None:
            self.model_hash = generate_hash()
            self.config = config
            self.features = self.config['features']
            self.target = self.config['target']

    def predict(self, X_test):
        print('Predicting {} instances...'.format(X_test.shape[0]))
        X_test = X_test[self.features]
        return self.model.predict(X_test)

    def build_model(self):
        return RandomForestRegressor()

    def save_model_artifacts(self, model, config):
        model_dir = os.path.join(MODEL_DIR, self.model_hash)
        os.mkdir(model_dir)

        model_path = os.path.join(model_dir, '{}.pk'.format(self.model_hash))
        config_path = os.path.join(model_dir, 'config.yaml')
        save_pickle(model, model_path)
        save_yaml(config, config_path)
        print('Saved model artifacts')
    
    def save_model_output(self, df_test):
        output_dir = os.path.join(OUTPUT_DIR, self.model_hash)
        os.mkdir(output_dir)

        df_test_path = os.path.join(output_dir, 'test_data.csv')
        metrics_path = os.path.join(output_dir, 'metrics.json')

        df_test.to_csv(df_test_path)
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print('Saved model output')

    def train(self, data):
        X, y = data[self.features], data[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=7)
        print('\n\nTraining on {}, Testing on {}'.format(self.X_train.shape[0], self.X_test.shape[0]))

        self.model = self.build_model()
        self.model.fit(self.X_train, self.y_train)

        self.save_model_artifacts(self.model, self.config)

    def evaluate(self):
        def mean_abs_pct_error(y_test, y_pred):
            return np.mean( np.abs( (y_test - y_pred) / y_test ) ) * 100

        y_pred = self.model.predict(self.X_test)

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