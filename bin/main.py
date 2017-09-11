from __future__ import division
from clone import Clone
import utils
import plot
import os
import pandas
import numpy as np
from collections import defaultdict
import cv2
from openpyxl import load_workbook 

DATADIR = "/mnt/spicy_4/daphnia/data"
SEGDATADIR = "/mnt/spicy_4/daphnia/analysis/simplesegmentation"
CLOSESEGDATADIR = "/mnt/spicy_4/daphnia/analysis/simplesegmentation_close"
ANALYSISDIR = "/mnt/spicy_4/daphnia/analysis/"
INDUCTIONMEDATADIR = "/mnt/spicy_4/daphnia/analysis/MetadataFiles/induction"
ext = '.bmp'

doAreaCalc = True
doAnimalEllipseFit = True
doEyeEllipseFit = True
doBodyLandmarks = True
doLength = True
doPedestalScore = False

files = os.listdir(DATADIR)
clones = dict()

print "Loading induction data\n"
inductiondates = dict()
inductionfiles = os.listdir(INDUCTIONMEDATADIR)

for i in inductionfiles:
    if not i.startswith("._") and (i.endswith(".xlsx") or i.endswith(".xls")):
        print "Loading " + i
        wb = load_workbook(os.path.join(INDUCTIONMEDATADIR,i),data_only=True)
        data = wb["Inductions"].values
        cols = next(data)[0:]
        data = list(data)
        df = pandas.DataFrame(data, columns=cols)
        df = df[df.ID_Number.notnull()] 
        for j,row in df.iterrows():
            if not row['ID_Number'] == "NaT":
                inductiondates[row['ID_Number']] = row['Induction_Date']

print "Loading clone data\n"
build_clonedata = True 
clones = defaultdict(list)

try:
    if build_clonedata: raise(IOError)
    clones = utils.load_pkl("clonedata", ANALYSISDIR)
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
                if barcode in inductiondates.keys():
                    induced = inductiondates[barcode]
                else:
                    induced = None

                clones[barcode].append(Clone(barcode,clone_id,treatment,replicate,rig,datetime,induced,DATADIR,SEGDATADIR,CLOSESEGDATADIR))
                
                try:
                    clones[barcode].pixel_to_mm = manual_scales[clones[barcode].micro_filepath]
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
        for barcode in clones.keys():
            for clone in clones[barcode]:
                clonelist = clones[barcode]
                print "Analyzing " + str(clone.filebase)
                try:
                    split = clone.split_channels(cv2.imread(clone.full_seg_filepath))
                    if doAreaCalc:
                        clone.calculate_area(split)
                    if doAnimalEllipseFit:
                        clone.fit_animal_ellipse(split)
                    if doEyeEllipseFit:
                        clone.fit_eye_ellipse(split)

                    if doBodyLandmarks:
                        if clone.tail is None:
                            im = cv2.imread(clone.full_filepath)
                            clone.find_body_landmarks(im, split)

                    if doLength:
                        clone.calculate_length()
                    if doPedestalScore:
                        im = cv2.imread(clone.full_filepath)
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
