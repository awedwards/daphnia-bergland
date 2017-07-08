from clone import Clone
import utils
import plot
import os
import numpy as np
from collections import defaultdict
import cv2

DATADIR = "/mnt/spicy_4/daphnia/data"
SEGDATADIR = "/mnt/spicy_4/daphnia/analysis/full_segmentation_output_20170708"
ANALYSISDIR = "/mnt/spicy_4/daphnia/analysis/"
doAreaCalc = True
doAnimalEllipseFit = True
doEyeEllipseFit = True

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

analysis = True
with open("/home/austin/Documents/daphnia_analysis_log.txt", "wb") as f:
    if analysis:
        for keys in clone_dict.keys():
            for clone in clone_dict[keys]:
                print "Analyzing " + clone.filebase
                try:
                    split = clone.split_channels(cv2.imread(clone.full_seg_filepath))
                    if doAreaCalc:
                        print "Calculating area for " + clone.filebase + "\n"
                        clone.calculate_area(split)
                    
                    if doAnimalEllipseFit:
                        print "Fitting ellipse for " + clone.filebase + "\n"
                        clone.fit_ellipse(split,"animal")
                        
                    if doEyeEllipseFit:
                        print "Fitting ellipse for eye of " + clone.filebase + "\n"
                        clone.fit_ellipse(split,"eye")
                except AttributeError as e:
                    f.write("Error analyzing " + clone.filebase + " because: " + str(e) + "\n")
                    

        utils.save_pkl(clone_dict, ANALYSISDIR, "clonedata")
