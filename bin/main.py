from __future__ import division
from clone import Clone
import utils
import plot
import os
import numpy as np
from collections import defaultdict
import cv2

DATADIR = "/mnt/spicy_4/daphnia/data"
SEGDATADIR = "/mnt/spicy_4/daphnia/analysis/simplesegmentation"
ANALYSISDIR = "/mnt/spicy_4/daphnia/analysis/"
ext = '.bmp'

doAreaCalc = True
doAnimalEllipseFit = True
doEyeEllipseFit = True
doBodyLandmarks = True
doLength = True
doPedestalScore = False

files = os.listdir(DATADIR)
clone_dict = defaultdict(list)

print "Loading clone data\n"
build_clonedata = True

try:
    if build_clonedata: raise(IOError)
    clone_dict = utils.load_pkl("clonedata", ANALYSISDIR)
except IOError:
    print "Clone data could not be located. Building from scratch:\n"
    
    # load manual_scales
    manual_scales = dict()
    with open(os.path.join(ANALYSISDIR, "manual_scales.txt"),'rb') as f:
        line = f.readline()
        while line:
            filename,conversion = line.strip().split(',')
            manual_scales[filename] = conversion
            line = f.readline()

    for f in files:
        
        if f.startswith("._"):
            continue
        elif f.endswith(ext) and f.startswith("full_"):

            imagetype,barcode,clone_id,treatment,replicate,rig,datetime = utils.parse(f)
            
            if barcode is not None:
                filebase = "_".join((barcode,clone_id,treatment,replicate,rig,datetime)) + ext 
          
                print "Adding " + filebase + " to clone list and calculating scale"
                
                clone_dict[clone_id].append(Clone(barcode,clone_id,treatment,replicate,rig,datetime,DATADIR,SEGDATADIR))
                
                try:
                    clone_dict[clone_id][-1].pixel_to_mm = manual_scales[clone_dict[clone_id][-1].micro_filepath]
                except (KeyError, AttributeError):
                    pass

    utils.save_pkl(clone_dict, ANALYSISDIR, "clonedata")

analysis = True

total = 0
for keys in clone_dict.keys():
    total += len(clone_dict[keys])

so_far = 0
with open("/home/austin/Documents/daphnia_analysis_log.txt", "wb") as f:
    if analysis:
        for keys in clone_dict.keys()[4:]:
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
                        im = cv2.imread(clone.full_filepath)
                        clone.get_pedestal_height(im,split)
                        clone.calculate_pedestal_score()
                except AttributeError as e:
                    print str(e)
                    f.write("Error analyzing " + clone.filebase + " because: " + str(e) + "\n")        
            print "Analyzed " + str(so_far) + " out of " + str(total) + "(" + str((so_far/total)*100) + "%)"
            print "Saving pickle"
            utils.save_pkl(clone_dict, ANALYSISDIR, "clonedata")
