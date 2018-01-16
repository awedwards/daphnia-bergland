from __future__ import division
import utils
import pandas as pd
from clone import Clone
import os
import cv2
import cPickle

DATADIR = "/mnt/spicy_4/daphnia/data"
ANALYSISDIR = "/mnt/spicy_4/daphnia/analysis/"
INDUCTIONMETADATADIR = "/mnt/spicy_4/daphnia/analysis/MetadataFiles/induction"
ext = '.bmp'

outfile = "analysis_results.txt"

analysis = True
build_clonedata = False

flgs = []

if analysis == True:
    flgs.append("getPxtomm")
    flgs.append("doEyeAreaCalc")
    flgs.append("doAntennaMasking")
    flgs.append("doAnimalAreaCalc")
    flgs.append("doBodyLandmarks")
    flgs.append("getOrientationVectors")

print "Loading clone data\n"


try:
    df = utils.csv_to_df(os.path.join(ANALYSISDIR, outfile))
    loaded = utils.df_to_clonelist(df, datadir=DATADIR)
    clones = utils.build_clonelist(DATADIR, ANALYSISDIR, INDUCTIONMETADATADIR)
    clones = utils.update_clone_list(clones, loaded)
    print "Successfully updated clone list"

except (AttributeError, IOError):
    
    clones = utils.build_clonelist(DATADIR, ANALYSISDIR, INDUCTIONMETADATADIR)

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
        "anterior",
        "posterior",
        "dorsal",
        "ventral",
	"ant_vec",
	"pos_vec",
	"dor_vec",
	"ven_vec",
        "eye_dorsal",
        "head",
        "tail",
        "tail_tip",
        "ventral_mask_endpoints",
        "dorsal_mask_endpoints",
        "anterior_mask_endpoints",
        "posterior_mask_endpoints"]

try:
    if os.stat(os.path.join(ANALYSISDIR, outfile)).st_size == 0:
        raise IOError
except (IOError, OSError):
    print "Starting new output file"
    with open(os.path.join(ANALYSISDIR, outfile), "wb+") as f:
        f.write( "\t".join(cols) + "\n")

try:
    pedestal_data = utils.load_pedestal_data( os.path.join(ANALYSISDIR, "pedestal.txt") )
except IOError:
    pedestal_data = {}

if analysis:
    for barcode in clones.keys():
        for dt in clones[barcode].keys():
            clone = clones[barcode][dt]["full"]
            
            if clone.filebase in pedestal_data.keys(): clone.pedestal_analyzed = True
            else: clone.pedestal_analyzed = False

            if not clone.analyzed:
                print "Analyzing " + clone.filebase
                utils.analyze_clone(clone, flgs)
                utils.write_clone(clone, cols, ANALYSISDIR, outfile)
                clone.analyzed = True
                
            if "doPedestalScore" in flgs:
                if not clone.pedestal_analyzed:
                    try:
                        clone.initialize_snake()
                        im = cv2.imread(os.path.join(DATADIR, clone.filepath), cv2.IMREAD_GRAYSCALE)
                        print "Fitting pedestal for " + clone.filebase
                        pedestal_data[clone.filebase] = clone.fit_pedestal(im)

                        utils.append_pedestal_line(clone.filebase, pedestal_data[clone.filebase], os.path.join(ANALYSISDIR, "pedestal.txt"))
                    except Exception as e:
                        print "Failed to fit pedestal for " + clone.filebase + " because of " + str(e)

        
#utils.save_clonelist(clones, ANALYSISDIR, "analysis_results_test.txt", cols)
