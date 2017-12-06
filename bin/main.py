from __future__ import division
import utils
import pandas as pd
from clone import Clone
import os
import cv2

DATADIR = "/mnt/spicy_4/daphnia/data"
SEGDATADIR = "/mnt/spicy_4/daphnia/02_simplesegmentation"
CLOSESEGDATADIR = "/mnt/spicy_4/daphnia/analysis/simplesegmentation_close"
ANALYSISDIR = "/mnt/spicy_4/daphnia/analysis/"
INDUCTIONMETADATADIR = "/mnt/spicy_4/daphnia/analysis/MetadataFiles/induction"
ext = '.bmp'

outfile = "analysis_results.txt"

analysis = True
build_clonedata = False

flgs = []

if analysis == True:
    flgs.append("getPxtomm")
    flgs.append("doAreaCalc")
    flgs.append("doAnimalEllipseFit")
    flgs.append("doEyeEllipseFit")
    flgs.append("doBodyLandmarks")
    flgs.append("doLength")
    flgs.append("doOrientation")
    flgs.append("doPedestalScore")

print "Loading clone data\n"


try:
    df = utils.csv_to_df(os.path.join(ANALYSISDIR, outfile))
    loaded = utils.df_to_clonelist(df, datadir=DATADIR, segdir=SEGDATADIR)
    clones = utils.build_clonelist(DATADIR, SEGDATADIR, ANALYSISDIR, INDUCTIONMETADATADIR)
    clones = utils.update_clone_list(clones, loaded)
    print "Successfully updated clone list"

except (AttributeError, IOError):
    
    clones = utils.build_clonelist(DATADIR, SEGDATADIR, ANALYSISDIR, INDUCTIONMETADATADIR)

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
        "total_animal_pixels",
        "animal_area",
        "total_eye_pixels",
        "eye_area",
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
	"ant_vec",
	"pos_vec",
	"dor_vec",
	"ven_vec",
        "head",
        "tail"]

try:
    if os.stat(os.path.join(ANALYSISDIR, outfile)).st_size == 0:
        raise IOError
except IOError:
    print "Starting new output file"
    with open(os.path.join(ANALYSISDIR, outfile), "wb+") as f:
        f.write( "\t".join(cols) + "\n")

pedestal_data = pd.DataFrame( columns = ['filepath', 'pedestal'] )

row = 0

if analysis:
    for barcode in clones.keys():
        for dt in clones[barcode].keys():
            clone = clones[barcode][dt]["full"]
            
            if not clone.analyzed:
                print "Analyzing " + clone.filebase
                utils.analyze_clone(clone, flgs)
                utils.write_clone(clone, cols, ANALYSISDIR, outfile)
                clone.analyzed = True
                
            if "doPedestalScore" in flgs:
                 clone.initialize_snake()
                 im = cv2.imread(os.path.join(DATADIR, clone.filepath), cv2.IMREAD_GRAYSCALE)
                 print "Fitting pedestal for " + clone.filebase
                 pedestal = clone.fit_pedestal(im)
                 pedestal_data.loc[row] = [clone.filebase, pedestal]
                 row+=1
    pedestal_data.to_csv(os.path.join(ANALYSISDIR, "pedestal_fits.txt", sep="\t"))
        
#utils.save_clonelist(clones, ANALYSISDIR, "analysis_results_test.txt", cols)
