from __future__ import division
from clone import Clone
import utils
import plot
import os
import pandas
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

doAreaCalc = True
doAnimalEllipseFit = True
doEyeEllipseFit = True
doBodyLandmarks = True
doLength = False
doOrientation = True
doPedestalScore = False

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
            df = pandas.DataFrame(data, columns=cols)
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

analysis = True
so_far = 0

def analyze(clone):
    try:
        im = cv2.imread(clone.full_filepath)
        split = clone.split_channels(cv2.imread(clone.full_seg_filepath))

        clone.calculate_area(split)
        clone.fit_animal_ellipse(split)
        clone.fit_ellipse(split,"eye", 4.6)
        clone.find_body_landmarks(im, split)
        clone.calculate_length()
        print "Analyzed " + clone.full_filepath
        return clone
    except Exception as e:
        print Exception

with open("/home/austin/Documents/daphnia_analysis_log.txt", "wb") as f:
    if analysis:
        for barcode in clones.iterkeys():
            for dt in clones[barcode].iterkeys():
                for imtype in clones[barcode][dt].iterkeys():
                    clone = clones[barcode][dt][imtype]
                    print "Analyzing " + str(clone.filepath)
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
                        if doPedestalScore:
                            im = cv2.imread(clone.filepath)
                            clone.get_pedestal_height(im,split)
                            clone.calculate_pedestal_score()
                    except AttributeError as e:
                        print str(e)
                        f.write("Error analyzing " + clone.filebase + " because: " + str(e) + "\n")
                    so_far+=1

                    if so_far%100==0:
                        print "Saving " + str(so_far) + " out of many"
                        utils.save_pkl(clones, ANALYSISDIR, "clonedata")
        utils.save_pkl(clones,ANALYSISDIR, "clonedata")
