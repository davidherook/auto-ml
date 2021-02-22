########################################################################################
# python train.py -c config.yaml -d data/boston_housing.csv
########################################################################################

import yaml
import argparse
import pandas as pd

from auto_ml.auto_ml import AutoML

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Training configuration')
    parser.add_argument('-d', '--data', type=str, help='Training data')
    args = vars(parser.parse_args())
    config_path = args['config']
    data = args['data']

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    df = pd.read_csv(data)

    automl = AutoML(config=config)
    automl.train(data=df)

    # Train / Test split
    # X, y = df[features], df[target]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)
    # print('\n\nTraining on {}, Testing on {}'.format(X_train.shape[0], X_test.shape[0]))

    # Predict with all models 
    # X_test_copy = X_test.copy()
    # for name, model in models:
    #     print(f'Running {name}...')
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     X_test_copy.loc[:, f'y_pred_{name}'] = y_pred

    #     # Print validation metrics
    #     metrics = validation_summary(y_test, y_pred, print_results=True, description='Full Test Set')
    #     rsq = metrics['validation']['Test R-sq']

    #     if save_output:
    #         plot_pred_vs_actual(y_test, y_pred, title=f'{name}, Rsq={rsq}', save_to=f'{output_dir}{name}_actual_vs_pred.png')

    #     if save_models:
    #         # Persist model(s) for deployment
    #         save_model(model, f'{model_dir}{name}.pk')
    #         print(f'Model has been saved as {model_dir}{name}.pk')
    
    # Write results to output
    # if save_output:
    #     df_results = pd.merge(df, X_test_copy.drop(features, axis=1), how='left', left_index=True, right_index=True)
    #     df_results.to_csv(output_predictions_path)
    #     print(f'Results written to {output_predictions_path}')
    #     with open(output_config_path, 'w') as f:
    #         f.write(json.dumps(config))

    # # Alias columns and write results to tableau folder
    # df_results_aliased, alias_map = alias_features(df_results, features=features, target=target[0])
    # df_results_aliased.to_csv(tableau_predictions_path, index=False)
    # print(f'Results written to {tableau_predictions_path}')

    # alias_json = json.dumps(alias_map)
    # with open(tableau_alias_map_path, "w") as f:
    #     f.write(alias_json)

    

 

    

    



        
    
