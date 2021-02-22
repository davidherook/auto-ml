import os
import yaml
from auto_ml.util import generate_hash, load_pickle, save_pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from shutil import copyfile

MODEL_DIR = 'model'
OUTPUT_DIR = 'output'

class AutoML(object):

    def __init__(self, model=None, config=None):

        self.model_hash = generate_hash()

        if model is not None:
            model_path = os.path.join(MODEL_DIR, '{}.pk'.format(model))
            self.model = load_pickle(model_path) 

            config_path = os.path.join(MODEL_DIR, 'config.yaml')
            with open(config_path) as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
            self.features = self.config['features']
            self.target = self.config['target']

        if config is not None:
            self.config = config
            self.features = self.config['features']
            self.target = self.config['target']

    def predict(self, X_test):
        print('Predicting {} instances...'.format(X_test.shape[0]))
        X_test = X_test[self.features]
        return self.model.predict(X_test)

    def model(self):
        return RandomForestRegressor()

    def save(self, model):
        filename = os.path.join(MODEL_DIR, '{}.pk'.format(self.model_hash))
        save_pickle(model, filename)
        print(f'Saved model to {filename}')

    def save_yaml(self, conf):
        filename = os.path.join(MODEL_DIR, 'config.yaml')
        with open(filename, 'w') as f:
            documents = yaml.dump(conf, f)

    def train(self, data):
        X, y = data[self.features], data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)
        print('\n\nTraining on {}, Testing on {}'.format(X_train.shape[0], X_test.shape[0]))

        model = self.model()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        X_test['y'] = y_test
        X_test['y_pred'] = y_pred

        # Persist model and config in model dir 
        self.save(model)
        self.save_yaml(self.config)

        # Write results to output dir
        output_dir = os.path.join(OUTPUT_DIR, self.model_hash)
        output_test_set_path = os.path.join(output_dir, 'test_data.csv')
        os.mkdir(output_dir)
        X_test.to_csv(output_test_set_path)