########################################################################################
#   Predict new instances
#       python predict.py -c config.yaml -d data/boston_housing.csv -m model/rf_regression.pk
########################################################################################

import yaml
import argparse
import pandas as pd

import sys 
sys.path.append('../')

from util.util import check_features_exist, load_model
    
       
if __name__ == '__main__':

    output_predictions_path = f'data/predictions.csv'
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Training configuration')
    parser.add_argument('-d', '--data', type=str, help='Data to predict')
    parser.add_argument('-m', '--model', type=str, help='Model to use to predict')
    args = vars(parser.parse_args())
    config_path = args['config']
    data = args['data']
    model_path = args['model']
    
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        features = config['features']

    # Read data
    df = pd.read_csv(data)
    check_features_exist(df, features)
    print('\n\nPredicting {} instances of new data...'.format(df.shape[0]))

    # Load model
    model = load_model(model_path)

    # Predict 
    X_test = df[features]
    X_test_copy = X_test.copy()
    y_pred = model.predict(X_test)
    X_test_copy.loc[:, 'y_pred'] = y_pred

    # Write output 
    df_results = pd.merge(df, X_test_copy.drop(features, axis=1), how='left', left_index=True, right_index=True)
    df_results.to_csv(output_predictions_path)
    print(f'Results written to {output_predictions_path}')

