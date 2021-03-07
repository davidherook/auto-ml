import time
import yaml
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model as load_keras_model

def generate_hash():
    t = int(time.time())
    return f"{t}"

def save_pickle(model, filename):
    pickle.dump(model, open(filename, 'wb'))

def load_pickle(filename):
    return pickle.load(open(filename, 'rb'))

def save_yaml(yamlfile, filename):
    with open(filename, 'w') as f:
        yaml_data = yaml.dump(yamlfile, f, sort_keys=False)

def load_yaml(filename):
    with open(filename) as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_data

def save_model_h5(model, filename):
    model.save(filename,
        overwrite=True,
        include_optimizer=True,
        save_format="h5")

def load_model_h5(filename):
    return load_keras_model(filename)

def train_val_test_split(X, y, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20):
    test_size = 1 - train_ratio
    val_size = test_ratio / (test_ratio + val_ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=val_size, random_state=7)
    return X_train, X_val, X_test, y_train, y_val, y_test

def pretty_confusion_matrix(y_true, y_pred):
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
    cm.columns = ['Predicted {}'.format(c) for c in cm.columns]
    cm.index = ['Actual {}'.format(c) for c in cm.index]
    return cm
