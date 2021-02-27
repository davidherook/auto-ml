########################################################################################
# python train.py -c config.yaml -d data/boston_housing.csv
########################################################################################

import argparse
import pandas as pd
from auto_ml.auto_ml import AutoML
from auto_ml.util import load_yaml

if __name__ == '__main__':

    print('\n\n'+'*'*40)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Training configuration')
    parser.add_argument('-d', '--data', type=str, help='Training data')
    args = vars(parser.parse_args())
    config_path = args['config']
    data = args['data']

    config = load_yaml(config_path)
    df = pd.read_csv(data)

    automl = AutoML(config=config)
    automl.train(data=df)
    automl.evaluate()
    print('*'*40+'\n\n')
