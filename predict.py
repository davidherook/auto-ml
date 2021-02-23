########################################################################
# Load AutoML with its hash and use it to predict
# Data set must be in csv format 
# Writes the data set with predictions the same directory as --data
#
# python predict.py --model 1613958519 --data data/test_data.csv
########################################################################

import yaml
import argparse
import pandas as pd
from auto_ml.auto_ml import AutoML
    
  
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='Model hash to use to predict')
    parser.add_argument('-d', '--data', type=str, help='Path to the file with test data to predict')
    args = vars(parser.parse_args())
    data = args['data']
    model = args['model']

    write_to = data.replace('.csv', '_predictions.csv')

    X_test = pd.read_csv(data)

    automl = AutoML(model=model)
    X_test['y_pred'] = automl.predict(X_test)

    X_test.to_csv(write_to)
    print(f'Predictions written to {write_to}')

