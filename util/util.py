import pandas as pd

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


