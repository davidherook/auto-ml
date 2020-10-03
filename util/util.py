


def alias_features(features, target):
    '''Standardize the features names by mapping the features in config.yaml with the names
    x1, x2, x3 ... xn and the target as y
    
    arguments:
        features: list
        target: string'''
    feats = {}
    for i, j in enumerate(features):
        feats.update({j:f'x{i}'})
    targ = {target: 'y'}
    return feats, targ