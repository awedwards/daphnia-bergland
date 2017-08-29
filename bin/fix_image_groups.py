import os
import utils
from clone import Clone

DATADIR = "/mnt/spicy_4/daphnia/data/"
ANALYSISDIR = "/mnt/spicy_4/daphnia/analysis"

clone_data = utils.load_pkl("clonedata",ANALYSISDIR)

for k in clone_data.keys():
    for clone in clone_data[k]:
        micro = ''
        full = ''
        close = ''

        try:
            micro = clone.micro_filepath
        except AttributeError:
            pass

        try:
            full = clone.full_filepath
        except AttributeError:
            pass

        try:
            close = clone.close_filepath
        except AttributeError:
            pass

        if micro == '' or full == '':
            print full,micro,close

            
