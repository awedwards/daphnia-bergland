from __future__ import division
import utils
from clone import Clone
import plot
import os
import pandas as pd
import numpy as np
import cv2
from collections import defaultdict
from datetime import datetime as dt

DATADIR = "/mnt/spicy_4/daphnia/dummy"
SEGDATADIR = "/mnt/spicy_4/daphnia/analysis/simplesegmentation"
CLOSESEGDATADIR = "/mnt/spicy_4/daphnia/analysis/simplesegmentation_close"
ANALYSISDIR = "/mnt/spicy_4/daphnia/analysis/"
INDUCTIONMETADATADIR = "/mnt/spicy_4/daphnia/analysis/MetadataFiles/induction"
ext = '.bmp'
datafile = os.path.join(ANALYSISDIR, "daphnia_analysis_results.txt")

analysis = False

if analysis == True:
  doAreaCalc = True
  doAnimalEllipseFit = True
  doEyeEllipseFit = True
  doBodyLandmarks = True
  doLength = True
  doOrientation = True
  doPedestalScore = True
else:
  doAreaCalc = False
  doAnimalEllipseFit = False
  doEyeEllipseFit = False
  doBodyLandmarks = False
  doLength = False
  doOrientation = False
  doPedestalScore = False

files = os.listdir(DATADIR)

print "Loading clone data\n"

build_clonedata = True 

try:
    if build_clonedata: raise(IOError)

    df = utils.csv_to_df(datafile)
    clones = utils.df_to_clonelist(df, datadir=DATADIR, segdir=SEGDATADIR)

except IOError:
    
    clones = utils.build_clonelist(DATADIR, SEGDATADIR, ANALYSISDIR, INDUCTIONMETADATADIR)
"""    
with open(os.path.join(ANALYSISDIR, "daphnia_analysis_results.txt"), "wb") as f:

    cols = ["filebase",
    "barcode",
    "cloneid",
    "pond",
    "id",
    "season",
    "treatment",
    "replicate",
    "rig",
    "datetime",
    "inductiondate",
    "animal_area",
    "animal_length",
    "pixel_to_mm",
    "animal_x_center",
    "animal_y_center",
    "animal_major",
    "animal_minor",
    "animal_theta",
    "eye_x_center",
    "eye_y_center",
    "eye_major",
    "eye_minor",
    "eye_theta",
    "anterior",
    "posterior",
    "dorsal",
    "ventral",
    "head",
    "tail"]
    
    f.write( "\t".join(cols) + "\n")

    for barcode in clones.iterkeys():
        for dt in clones[barcode].iterkeys():
	    clone = clones[barcode][dt]["full"]
	    print "Analyzing " + str(clone.filepath)
	    success = True
	    try:
		split = clone.split_channels(cv2.imread(clone.seg_filepath))
		if doAreaCalc:
		    print "Calculating area."
		    clone.calculate_area(split)
		if doAnimalEllipseFit:
		    print "Fitting ellipse to body."
		    clone.fit_animal_ellipse(split)
		if doEyeEllipseFit:
		    print "Fitting ellipse to eye."
		    clone.fit_eye_ellipse(split)
		if doOrientation:
		    print "Finding animal orientation."
		    clone.get_anatomical_directions()
		if doBodyLandmarks:
		    print "Finding body landmarks."
		    if clone.tail is None:
			im = cv2.imread(clone.filepath)
			clone.find_body_landmarks(im, split)
		if doLength:
		    print "Calculating length."
		    clone.calculate_length()
		#if doPedestalScore:
		#    clone.initialize_snake()
	    except AttributeError as e:
		print str(e)
		#f.write("Error analyzing " + clone.filebase + " because: " + str(e) + "\n")
		success = False
            
	    if success:
	        tmpdata = list()
	        
                for c in cols:
                    val = str(getattr(clone, c))
                    
                    if val is not None:
		        tmpdata.append( val )
                    else:
                        tmpdata.append("")
	    
	        f.write( "\t".join(tmpdata) + "\n")
"""
