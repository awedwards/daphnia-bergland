from __future__ import division
from clone import Clone
import utils
import plot
import os
import numpy as np
from collections import defaultdict
import cv2

DATADIR = "/mnt/spicy_4/daphnia/data"
SEGDATADIR = "/mnt/spicy_4/daphnia/analysis/full_segmentation_output_20170711"
ANALYSISDIR = "/mnt/spicy_4/daphnia/analysis/"

doAreaCalc = True
doAnimalEllipseFit = True
doEyeEllipseFit = True
doBodyLandmarks = True
doLength = True
doPedestalScore = True

files = os.listdir(DATADIR)
clone_dict = defaultdict(list)

print "Loading clone data\n"
try:
    clone_dict = utils.load_pkl("clonedata", ANALYSISDIR)
except IOError:
    print "Clone data could not be located. Building from scratch:\n"
    
    for f in files:
      skip = False
      if f.startswith("._"):
          continue
      elif f.endswith(".bmp"):
          delim = "_"
          fileparts = f.split(delim)
          
          imagetype = fileparts[0]

          if len(fileparts) == 7:
              clone_id = delim.join([fileparts[1], fileparts[2]])
              fileparts.pop(1)
          elif len(fileparts) == 6:
              clone_id = fileparts[1]
          else:
              raise(IndexError)

          treatment = fileparts[2]
          replicate = fileparts[3]
          rig = fileparts[4]
          datetime = fileparts[5][:-4]
          filebase = delim.join((clone_id,treatment,replicate,rig,datetime))
          if filebase.startswith("clone") or filebase.startswith("full"):
              print fileparts
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

total = 0
for keys in clone_dict.keys():
    total += len(clone_dict[keys])

so_far = 0
with open("/home/austin/Documents/daphnia_analysis_log.txt", "wb") as f:
    if analysis:
        for keys in clone_dict.keys():
            so_far += len(clone_dict[keys])
            for clone in clone_dict[keys]:
                print "Analyzing " + str(clone.filebase)
                try:
                    split = clone.split_channels(cv2.imread(clone.full_seg_filepath))
                    if doAreaCalc:
                        clone.calculate_area(split)
                    
                    if doAnimalEllipseFit:
                        clone.fit_animal_ellipse(split)
                        
                    if doEyeEllipseFit:
                        clone.fit_ellipse(split, "eye", 4.6)

                    if doBodyLandmarks:
                        clone.find_body_landmarks(split)

                    if doLength:
                        clone.calculate_length()

                    if doPedestalScore:
                        clone.calculate_pedestal_score(split)
                except AttributeError as e:
                    f.write("Error analyzing " + clone.filebase + " because: " + str(e) + "\n")        
            print "Analyzed " + str(so_far) + " out of " + str(total) + "(" + str((so_far/total)*100) + "%)"
            print "Saving pickle"
            utils.save_pkl(clone_dict, ANALYSISDIR, "clonedata")
