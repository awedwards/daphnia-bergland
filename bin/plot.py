from clone import Clone
import utils
import os
import pandas as pd
import numpy as np
from collections import defaultdict

def dict_to_df(clones):

    data = defaultdict(list)
    
    for bc in clones.iterkeys():
        for dtkey in clones[bc].iterkeys():
            for clone in clones[bc][dtkey].itervalues():
                for k,v in clone.__dict__.iteritems():
                    data[k].append(v)

    return pd.DataFrame(data = data)

def df_to_csv(df, path):   
    df.to_csv(path, sep=',')

def csv_to_df(path):
    return pd.read_csv(path, sep=',')
