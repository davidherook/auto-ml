import time
import yaml
import pickle
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
