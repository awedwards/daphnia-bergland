import pickle
import os
import numpy as np
import re

def save_pkl(obj, path, name):
    with open(os.path.join(path, name) + '.pkl','wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(name, path):
    with open(os.path.join(path,name) + '.pkl','rb') as f:
        return pickle.load(f)

def merge_channels(im,channel1,channel2):
    copy = im[:,:,channel1].copy()
    copy[np.where(im[:,:,channel2])] = 1
    return copy

def parse(s):
    
    # this method parses clone ids and finds pond name and id
    #
    # if only the pond is given (e.g. Cyrus), the id defaults to pond name

    pattern = '^A?(?P<pond>[DW]>\s?\d{1,2}|Cyril|DBunk|Male|C14|Chard)[\s?|\.?|_?]?A?(?P<cloneid>\d{1,3})?A?'
    m = re.search(pattern, s)
    pond, cloneid = m.groups()

    if cloneid is None:
        cloneid = pond

    if 'D' in pond and '8' in pond:
        pond = 'D8'
    elif 'D' in pond and '10' in pond:
        pond = 'D10'

    return pond, cloneid
