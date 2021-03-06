########################################################################################
# Example, regression:
# python train.py -c config.yaml -d data/boston_housing.csv
#
# Example, classification:
# python train.py -c config.yaml -d data/boston_housing_clf.csv
########################################################################################

import argparse
import pandas as pd
from auto_ml.auto_ml import AutoMLRegressor, AutoMLClassifier
from auto_ml.util import load_yaml

if __name__ == '__main__':

    print('\n\n'+'*'*40)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Training configuration')
    parser.add_argument('-d', '--data', type=str, help='Training data')
    parser.add_argument('-n', '--no-save', action='store_false', help='Whether to save')
    args = vars(parser.parse_args())
    config_path = args['config']
    data = args['data']
    save = args['no_save']

    config = load_yaml(config_path)
    df = pd.read_csv(data)

    if config['problem_type'] == 'regression':
        automl = AutoMLRegressor(config=config)
    elif config['problem_type'] == 'classification':
        automl = AutoMLClassifier(config=config)

    automl.train(data=df, save=save)
    automl.evaluate(save=save)
    print('*'*40+'\n\n')
