import pandas as pd
import pickle
import matplotlib.pyplot as plt

def alias_features(df, features, target):
    """Standardize the column names to be used in tableau for different datasets
    
    Args:
        df (pd.DataFrame): The dataframe for which column names are aliased
        features (list): Features used in the model 
        target (string): Target variable for prediction 

    Returns:
        df (pd.DataFrame): The dataframe with aliased columns
        aliases(dict): The mapping of variable name to alias
    """
    aliases = {}
    for i, j in enumerate(features):
        aliases.update({j:f'x{i}'})
    aliases.update({target:'y'})
    df.rename(columns=aliases, inplace=True)
    return df, aliases

def check_features_exist(df, features):
    """Confirm that list of features is in df.columns
    
    Args:
        df (pd.DataFrame): The dataframe for which column names are aliased
        features (list): Features used in the model 

    Returns:
        features_ok(bool): Whether all features are present
    """
    features_ok = set(features).issubset(df.columns)
    if not features_ok:
        raise ValueError('Features provided in the config were not found in the dataset. The dataset features are:\n {}'.format(list(df.columns)))
    return True

def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    return pickle.load(open(filename, 'rb'))

def plot_pred_vs_actual(y_true, y_pred, title="Actual vs. Predicted", save_to=None):
    plt.scatter(y_true, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    if save_to is not None:
        plt.savefig(save_to)
    plt.clf()



