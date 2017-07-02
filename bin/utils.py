import pickle
import os

def save_pkl(obj, path, name):
    with open(os.path.join(path, name) + '.pkl','wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(name, path):
    with open(os.path.join(path,name) + '.pkl','rb') as f:
        return pickle.load(f)
