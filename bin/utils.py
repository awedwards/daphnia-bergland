from __future__ import division
from clone import Clone
import pickle
import os
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from openpyxl import load_workbook 
from ast import literal_eval
import cv2

def save_pkl(obj, path, name):
    with open(os.path.join(path, name) + '.pkl','wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(name, path):
    with open(os.path.join(path,name) + '.pkl','rb') as f:
        return pickle.load(f)

def merge_channels(im,channel1,channel2):
    copy = im[:,:,channel1].copy()
    copy[np.where(im[:,:,channel2])] = 1
    return copy

def parsePond(s):
    
    # this method parses clone ids and finds pond name and id
    #
    # if only the pond is given (e.g. Cyrus), the id defaults to pond name

    pattern = '^(A?[DW]?\s?\d{1,2}|Cyril|DBunk|Male|C14|Chard)[\s?|\.?|_?]?(A?\d{1,3}A?)?'
    m = re.search(pattern, s)
    pond, cloneid = m.groups()

    if cloneid is None:
        cloneid = pond

    return pond, cloneid

def parse(s):

    pattern = '^([A-Za-z]{4,10})_(\d{6})?_?(DBunk_?\s?\d{1,3}|Male_\d|A?[DW]?\s?\d{1,2}[_|\s|.]?A?\d{1,3}A?|Cyril|C14|Chard)_(juju\d?|ctrl|NA|(?:\d*\.)?\d+)_(\d[A-Z]|NA)_(Rig[AB])_(\d{8}T\d{6}).bmp$'
    
    m = re.search(pattern,s)
    filetype,barcode,cloneid,treatment,replicate,rigId,datetime = m.groups()

    return filetype,barcode,cloneid,treatment,replicate,rigId,datetime

def norm(x):

    # normalizes an array such that values lie between 0 and 1

    return (x - np.min(x)) / (np.max(x) - np.min(x))

def recursivedict():
    
    # initializes a default dictionary with an arbitrary number of dimensions

    return defaultdict(recursivedict)

def load_induction_data(filepath):
    
    print "Loading induction data\n"
    inductiondates = dict()
    inductionfiles = os.listdir(filepath)

    for i in inductionfiles:
        if not i.startswith("._") and (i.endswith(".xlsx") or i.endswith(".xls")):
            print "Loading " + i
            wb = load_workbook(os.path.join(filepath,i), data_only=True)
            data = wb["Inductions"].values
            cols = next(data)[0:]
            data = list(data)
	    df = pd.DataFrame(data, columns=cols)
            df = df[df.ID_Number.notnull()]

            for j,row in df.iterrows():
                if not str(row['ID_Number']) == "NaT":
                    time = pd.Timestamp(row['Induction_Date'])
                    inductiondates[str(int(row['ID_Number']))] = time.strftime("%Y%m%dT%H%M%S")

    return inductiondates

def load_manual_scales(filepath):

     # load manual_scales
    manual_scales = {}
    with open(os.path.join(filepath, "manual_scales.txt"),'rb') as f:
        line = f.readline()
        while line:
            filename,conversion = line.strip().split(',')
            manual_scales[filename] = conversion
            line = f.readline()
    
    return manual_scales

def write_pedestal_data(data, filepath):

    with open(filepath, "w") as f:
        
        f.write('\t'.join(["filebase","pedestal_data"]) + "\n")

        for k, v in data.iteritems():
            f.write('\t'.join([k, v]) + "\n")

def append_pedestal_line(clone_filebase, data, filepath):

    with open(filepath, "a") as f:
        f.write('\t'.join([clone_filebase, str(data)]) + "\n")

def load_pedestal_data(filepath):
    
    data = {}
    
    with open(filepath, "r") as f:
        line = f.readline()
        while line:
            d = line.strip().split("\t")
            data[d[0]] = np.array(literal_eval(d[1]))
            line = f.readline()

    return data

def build_clonelist(datadir, analysisdir, inductiondatadir, ext=".bmp"):
    
    # input: paths to data, segmented data and metadata files

    clones = recursivedict()
   
    inductiondates = load_induction_data(inductiondatadir)
    manual_scales = load_manual_scales(analysisdir)

    files = os.listdir(datadir)
    
    for f in files:
        
        if f.startswith("._"):
            continue
        
        elif f.endswith(ext) and f.startswith("full_"):
            
            filebase = f[5:]

            print "Adding " + f + " to clone list"
            imagetype,barcode,clone_id,treatment,replicate,rig,datetime = parse(f)
            
            if barcode is not None:
          
                if str(barcode) in inductiondates.iterkeys():
                    induction = inductiondates[str(barcode)]
                else:
                    induction = None
                
                clones[barcode][datetime][imagetype] = Clone( filebase,
                        imagetype,
                        barcode,
                        clone_id,
                        treatment,
                        replicate,
                        rig,
                        datetime,
                        induction,
                        datadir)
        
                if imagetype == "close":
                    clones[barcode][datetime][imagetype].pixel_to_mm = 1105.33
                try:
                    clones[barcode][datetime][imagetype].pixel_to_mm = manual_scales[clones[barcode].micro_filepath]
                except (KeyError, AttributeError):
                    pass
    
    return clones

def csv_to_df(csvfile):
    
    try:
        return pd.read_csv(csvfile, sep="\t")
    except Exception as e:
        print "Could not load csv because: " + str(e)
        return

def df_to_clonelist(df, datadir = None, segdir = None):

    clones = recursivedict()

    for index, row in df.iterrows():
        clone = Clone( row['filebase'],
		'full',
                row['barcode'],
                row['cloneid'],
                row['treatment'],
                row['replicate'],
                row['rig'],
                row['datetime'],
                row['inductiondate'],
                datadir)

        for k in row.keys():
            try:
                setattr(clone, k, literal_eval(row[k]))
            except (ValueError, SyntaxError):
                setattr(clone, k, row[k])

        clones[str(row['barcode'])][str(row['datetime'])]['full'] = clone
    
    return clones

def dfrow_to_clonelist(df, irow, datadir = None, segdir = None):

    row = df.iloc[irow]

    return Clone( row['filebase'],
		'full',
                row['barcode'],
                row['cloneid'],
                row['treatment'],
                row['replicate'],
                row['rig'],
                row['datetime'],
                row['inductiondate'],
                datadir)

def update_clone_list(clones, loadedclones):

     for barcode in loadedclones.iterkeys():
        for dt in loadedclones[barcode].iterkeys():
            clones[barcode][dt]['full'] = loadedclones[barcode][dt]['full']
            clones[barcode][dt]['full'].analyzed = True
     return clones  

def save_clonelist(clones, path, outfile, cols):
   
    with open(os.path.join(path, outfile), "wb+"):
        f.write( "\t".join(cols) + "\n")
        
    for barcode in clones.iterkeys():
        for dt in clones[barcode].iterkeys():
            clone = clones[barcode][dt]["full"]
            write_clone(clone, cols, path, outfile)

def update_attr(src, dest, attr):

    setattr(dest, attr, getattr(src, attr))

def write_clone(clone, cols, path, outfile):

    try:        
        with open(os.path.join(path, outfile), "ab+") as f:

            tmpdata = []
                    
            for c in cols:
            
                val = str(getattr(clone, c))
                
                if val is not None:
                    tmpdata.append( val )
                else:
                    tmpdata.append("")
        
            f.write( "\t".join(tmpdata) + "\n")

    except (IOError, AttributeError):
        print "Can't write clone data to file"

def analyze_clone(clone, flags, pedestal_data=None):

    #try:

    im = cv2.imread(clone.filepath, cv2.IMREAD_GRAYSCALE)

    if "getPxtomm" in flags:
        print "Extracting pixel-to-mm conversion factor"
        try:
            micro = cv2.imread(clone.micro_filepath)
            clone.pixel_to_mm = clone.calc_pixel_to_mm(micro)
        except Exception as e:
            print "Could not extract because: " + str(e)
    if clone.pixel_to_mm is not None:
        if "doEyeAreaCalc" in flags:
            print "Calculating area for eye."
            clone.find_eye(im)
            clone.get_eye_area()


        if "doAntennaMasking" in flags:
            print "Masking antenna and fitting ellipse to animal."
            clone.mask_antenna(im)

        if "doAnimalAreaCalc" in flags:
            print "Calculating area for animal."
            clone.count_animal_pixels(im)
            clone.get_animal_area()

        if "getOrientationVectors" in flags:
            print "Calculating orientation vectors."
            clone.get_orientation_vectors()

        if "doLength" in flags:
            print "Calculating length"
            clone.calculate_length()

        if "doPedestalScore" in flags:
            print "Calculating pedestal area"
            try:
                coords = pedestal_data[clone.filebase]
                clone.get_pedestal_area(coords)
                clone.get_pedestal_max_height(coords)
                clone.get_pedestal_theta(coords)

            except KeyError:
                print "No pedestal data for clone " + clone.filebase
    #except AttributeError:
    #    print "Error during analysis of " + clone.filepath
