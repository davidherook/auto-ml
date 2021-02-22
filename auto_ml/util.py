import time
import pickle

def generate_hash():
    t = int(time.time())
    return f"{t}"

def save_pickle(model, filename):
    pickle.dump(model, open(filename, 'wb'))

def load_pickle(filename):
    return pickle.load(open(filename, 'rb'))
