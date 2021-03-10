########################################################################
# Load AutoMLRegressor with its hash and use it to predict
# Data set must be in csv format 
# Writes the data set with predictions the same directory as --data
#
# python predict.py --model 1614976220 --data data/test_data.csv
########################################################################

import os
import argparse
import pandas as pd
from auto_ml.auto_ml import AutoMLRegressor, AutoMLClassifier
from auto_ml.util import load_yaml
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_DIR = 'model'
  
if __name__ == '__main__':
    
    print('\n\n'+'*'*40)
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='Model hash to use to predict')
    parser.add_argument('-d', '--data', type=str, help='Path to the file with test data to predict')
    args = vars(parser.parse_args())
    data = args['data']
    model = args['model']
    config_path = os.path.join(MODEL_DIR, model, 'config.yaml')
    config = load_yaml(config_path)

    write_to = data.replace('.csv', '_predictions.csv')

    X_test = pd.read_csv(data)

    if config['problem_type'] == 'regression':
        automl = AutoMLRegressor(model=model)
    elif config['problem_type'] == 'classification':
        automl = AutoMLClassifier(model=model)

    X_test[automl.target + '_pred'] = automl.predict(X_test)
    X_test.to_csv(write_to, index=False)
    print(f'Predictions written to {write_to}')
    print('*'*40+'\n\n')
