import time
import yaml
import pickle

def generate_hash():
    t = int(time.time())
    return f"{t}"

def save_pickle(model, filename):
    pickle.dump(model, open(filename, 'wb'))

def load_pickle(filename):
    return pickle.load(open(filename, 'rb'))

def save_yaml(yamlfile, filename):
    with open(filename, 'w') as f:
        documents = yaml.dump(yamlfile, f)

def load_yaml(filename):
    with open(filename) as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_data
