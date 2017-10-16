from __future__ import division
from clone import Clone
import utils
import plot
import os
import pandas as pd
import numpy as np
import cv2
from collections import default_dict
from openpyxl import load_workbook 

DATADIR = "/mnt/spicy_4/daphnia/data"
SEGDATADIR = "/mnt/spicy_4/daphnia/analysis/simplesegmentation"
CLOSESEGDATADIR = "/mnt/spicy_4/daphnia/analysis/simplesegmentation_close"
ANALYSISDIR = "/mnt/spicy_4/daphnia/analysis/"
INDUCTIONMETADATADIR = "/mnt/spicy_4/daphnia/analysis/MetadataFiles/induction"
ext = '.bmp'

analysis = True

if analysis == True:
  doAreaCalc = True
  doAnimalEllipseFit = True
  doEyeEllipseFit = True
  doBodyLandmarks = True
  doLength = True
  doOrientation = True
  doPedestalScore = True

files = os.listdir(DATADIR)

print "Loading clone data\n"

build_clonedata = False 

try:
    if build_clonedata: raise(IOError)
    clones = utils.load_pkl("clonedata", ANALYSISDIR)
except IOError:
    
    print "Clone data could not be located. Building from scratch:\n"`:w

    clones = utils.recursivedict()
    
    print "Loading induction data\n"
    inductiondates = dict()
    inductionfiles = os.listdir(INDUCTIONMETADATADIR)

    for i in inductionfiles:
        if not i.startswith("._") and (i.endswith(".xlsx") or i.endswith(".xls")):
            print "Loading " + i
            wb = load_workbook(os.path.join(INDUCTIONMETADATADIR,i),data_only=True)
            data = wb["Inductions"].values
            cols = next(data)[0:]
            data = list(data)
	    df = pd.DataFrame(data, columns=cols)
            df = df[df.ID_Number.notnull()] 
            for j,row in df.iterrows():
                if not row['ID_Number'] == "NaT":
                    inductiondates[row['ID_Number']] = row['Induction_Date']
 
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
        elif f.endswith(ext) and (f.startswith("full_") or f.startswith("close_")):
            
            print "Adding " + f + " to clone list and calculating scale"
            imagetype,barcode,clone_id,treatment,replicate,rig,datetime = utils.parse(f)
            
            if barcode is not None:
          
                if barcode in inductiondates.iterkeys():
                    induced = inductiondates[barcode]
                else:
                    induced = None
                
                if imagetype == "full":
                    segdir = SEGDATADIR
                elif imagetype == "close":
                    segdir = CLOSESEGDATADIR

                clones[barcode][datetime][imagetype] = Clone(imagetype,barcode,clone_id,treatment,replicate,rig,datetime,induced,DATADIR,segdir)
                if imagetype == "close":
                    clones[barcode][datetime][imagetype].pixel_to_mm = 1105.33
                try:
                    clones[barcode][datetime][imagetype].pixel_to_mm = manual_scales[clones[barcode].micro_filepath]
                except (KeyError, AttributeError):
                    pass

    utils.save_pkl(clones, ANALYSISDIR, "clonedata")

so_far = 0

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
    "animal_length"
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
    
    f.write( ",".join(cols) + "/n")

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
		if doBodyLandmarks and imtype == "full":
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
		f.write("Error analyzing " + clone.filebase + " because: " + str(e) + "\n")
		success = False
            
	    if success:
	        tmpdata = list()
	        for c in cols:
		    tmpdata.append( getattr(clone, c) )
	    
	        f.write( ",".join(tmpdata) + "/n")
	    
	    if so_far%100==0:
		print "Saving " + str(so_far) + " out of many"
