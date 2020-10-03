########################################################################################
#   Try several models for a regression problem
#
#
#
#
#
#   From the regression dir:
#       python train.py -c config.yaml
########################################################################################

import yaml
import argparse
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor

models = [
    ('linear_regression', LinearRegression()),
    ('sgd_regression', SGDRegressor(random_state=7)),
    ('rf_regression', RandomForestRegressor(n_estimators=50, random_state=7))
]

if __name__ == '__main__':

    run_time = datetime.now().strftime('%Y_%m_%d_%HH_%MM_%SS')

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Training configuration')
    args = vars(parser.parse_args())
    config_path = args['config']

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        data = config['training_data']
        features = config['features']
        target = config['target']

    # Read data
    df = pd.read_csv(data)

    # Train / Test split
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)
    print('Training on {}, Testing on {}'.format(X_train.shape[0], X_test.shape[0]))

    print(X_train.index)
    print(X_test.index)

    # Predict with all models 
    X_test_copy = X_test.copy()
    for name, model in models:
        print(f'\nRunning {name}...')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # all_pred.append((name, y_pred))
        X_test_copy.loc[:, f'y_pred_{name}'] = model.predict(X_test)
        # X_test.loc[:, 'model'] = name
    
    # y_pred_cols = [f'y_pred_{name}' for name, model in models]
    df_results = pd.merge(df, X_test_copy.drop(features, axis=1), how='left', left_index=True, right_index=True)

    print(X_test_copy)
    print(df_results)

    



        
    
