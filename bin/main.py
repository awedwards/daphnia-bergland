from clone import Clone
import pickle
import os
import numpy as np
from collections import defaultdict

DATADIR = "/mnt/spicy_4/daphnia/data"
SEGDATADIR = "/mt/spicy_4/daphnia/daphnia_with_appendages/"

def save_pkl(obj,name):
    with open('/mnt/spicy_4/daphnia/analysis/' + name + '.pkl','wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(name):
    with open('/mnt/spicy_4/daphnia/analysis/' + name + '.pkl','rb') as f:
        return pickle.load(f)

files = os.listdir(DATADIR)
clone_dict = defaultdict(list)

print "Loading clone data\n"
try:
    clone_dict = load_pkl("clonedata")
except IOError:
    print "Clone data could not be located. Building from scratch:\n"
    
    for f in files:
      if f.endswith(".bmp"):
          delim = "_"
          fileparts = f.split(delim)
          imagetype = fileparts[0]
          clone_id = fileparts[1]
          treatment = fileparts[2]
          replicate = fileparts[3]
          rig = fileparts[4]
          datetime = fileparts[5]
          
          filebase = delim.join((clone_id,treatment,replicate,rig,datetime))
      
          for val, i in enumerate(clone_dict[clone_id]):
              if i.filebase == filebase:
                  continue
      
          print "Adding " + filebase + " to clone list and calculating scale"
          
          try:
              clone_dict[clone_id].append(Clone(clone_id,treatment,replicate,rig,datetime,DATADIR,SEGDATADIR))
          except AttributeError:
              if f.startswith("._"): print "Skipping " + f + ". Probably should delete that."
    save_pkl(clone_dict,"clonedata")

count = 0
for k in clone_dict.keys():
    for clone in clone_dict[k]:
        try:
            if clone.pixel_to_mm is None or np.isnan(clone.pixel_to_mm):
                count += 1
        except AttributeError:
            count += 1

print count
