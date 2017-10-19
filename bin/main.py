from __future__ import division
import utils
from clone import Clone
import os

DATADIR = "/mnt/spicy_4/daphnia/data"
SEGDATADIR = "/mnt/spicy_4/daphnia/analysis/simplesegmentation"
CLOSESEGDATADIR = "/mnt/spicy_4/daphnia/analysis/simplesegmentation_close"
ANALYSISDIR = "/mnt/spicy_4/daphnia/analysis/"
INDUCTIONMETADATADIR = "/mnt/spicy_4/daphnia/analysis/MetadataFiles/induction"
ext = '.bmp'

outfile = "analysis_results.txt"

analysis = True

flgs = []

if analysis == True:
    flgs.append("getPxtomm")
    flgs.append("doAreaCalc")
    flgs.append("doAnimalEllipseFit")
    flgs.append("doEyeEllipseFit")
    flgs.append("doBodyLandmarks")
    flgs.append("doLength")
    flgs.append("doOrientation")
    #flgs.append("doPedestalScore")

files = os.listdir(DATADIR)

print "Loading clone data\n"

build_clonedata = False

try:
    df = utils.csv_to_df(os.path.join(ANALYSISDIR, outfile))
    loaded = utils.df_to_clonelist(df, datadir=DATADIR, segdir=SEGDATADIR)
    clones = utils.build_clonelist(DATADIR, SEGDATADIR, ANALYSISDIR, INDUCTIONMETADATADIR)
    clones = utils.update_clone_list(clones, loaded)

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

try:
    if os.stat(os.path.join(ANALYSISDIR, outfile)) == 0:
        raise IOError
except IOError:
    with open(os.path.join(ANALYSISDIR, outfile), "wb+") as f:
        f.write( "\t".join(cols) + "\n")
     
if analysis:
    for barcode in clones.keys():
        for dt in clones[barcode].keys():
            clone = clones[barcode][dt]["full"]
            if not clone.analyzed:
                print "Analyzing " + clone.filebase
                utils.analyze_clone(clone, flgs)
                utils.write_clone(clone, cols, ANALYSISDIR, outfile)
                clone.analyzed = True

#utils.save_clonelist(clones, ANALYSISDIR, "analysis_results_test.txt", cols)
