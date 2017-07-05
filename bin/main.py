from clone import Clone
import utils
import plot
import os
import numpy as np
from collections import defaultdict
import cv2

DATADIR = "/mnt/spicy_4/daphnia/data"
SEGDATADIR = "/mnt/spicy_4/daphnia/daphnia_with_appendages/"
ANALYSISDIR = "/mnt/spicy_4/daphnia/analysis/"
doAreaCalc = False

files = os.listdir(DATADIR)
clone_dict = defaultdict(list)

print "Loading clone data\n"
try:
    clone_dict = utils.load_pkl("clonedata", ANALYSISDIR)
except IOError:
    print "Clone data could not be located. Building from scratch:\n"
    
    for f in files:
      skip = False
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
                  skip = True
                  break
      
          print "Adding " + filebase + " to clone list and calculating scale"
          if not skip:     
              try:
                  clone_dict[clone_id].append(Clone(clone_id,treatment,replicate,rig,datetime,DATADIR,SEGDATADIR))
              except AttributeError:
                  if f.startswith("._"): print "Skipping " + f + ". Probably should delete that."
    utils.save_pkl(clone_dict, ANALYSISDIR, "clonedata")

if doAreaCalc:
    for keys in clone_dict.keys():
        for clone in clone_dict[keys]:
            print "Calculating area for " + clone.filebase + "\n"
            try:
                split = clone.split_channels(cv2.imread(clone.full_seg_filepath))
                clone.calculate_area(split)
                print clone.animal_area

            except AttributeError:
                pass

    utils.save_pkl(clone_dict, ANALYSISDIR, "clonedata")


