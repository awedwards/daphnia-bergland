from __future__ import division
import utils
import pandas as pd
from clone import Clone
import os
import cv2

DATADIR = "/mnt/spicy_4/daphnia/data"
ANALYSISDIR = "/mnt/spicy_4/daphnia/analysis/"
INDUCTIONMETADATADIR = "/mnt/spicy_4/daphnia/analysis/MetadataFiles/induction"
ext = '.bmp'

current = "analysis_results.txt"
out = "analysis_results_current.txt"

analysis = True
build_clonedata = False

flags = []

if analysis == True:
    #flags.append("getPxtomm")
    #flags.append("doEyeAreaCalc")
    #flags.append("doAntennaMasking")
    #flags.append("doAnimalAreaCalc")
    #flags.append("getOrientationVectors")
    #flags.append("doLength")
    #flags.append("fitPedestal")
    flags.append("doPedestalScore")

print "Loading clone data\n"

try:
    df = utils.csv_to_df(os.path.join(ANALYSISDIR, current))
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
        "animal_length_pixels",
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
        "posterior_mask_endpoints",
        "pedestal_max_height_pixels",
        "pedestal_area_pixels",
	"pedestal_max_height",
	"pedestal_area",
        "pedestal_window_max_height_pixels",
        "pedestal_window_area_pixels",
        "pedestal_window_max_height",
        "pedestal_window_area"]

try:
    if os.stat(os.path.join(ANALYSISDIR, out)).st_size == 0:
        raise IOError
except (IOError, OSError):
    print "Starting new output file"
    with open(os.path.join(ANALYSISDIR, out), "wb+") as f:
        f.write( "\t".join(cols) + "\n")

try:
    "Loading pedestal data"
    pedestal_data = utils.load_pedestal_data( os.path.join(ANALYSISDIR, "pedestal_final.txt") )
except IOError:
    pedestal_data = {}

if analysis:
    for barcode in clones.keys():
        for dt in clones[barcode].keys():
            
            clone = clones[barcode][dt]["full"]
            if clone.filebase not in ["110558_D8_256_ctrl_2B_RigB_20170921T142058.bmp"]:
                
                if clone.filebase in pedestal_data.keys(): clone.pedestal_analyzed = True
                else: clone.pedestal_analyzed = False

                print "Analyzing " + clone.filebase
                utils.analyze_clone(clone, flags, pedestal_data=pedestal_data)
                utils.write_clone(clone, cols, ANALYSISDIR, out)
                    
                if "fitPedestal" in flags:
                    if not clone.pedestal_analyzed:
                        try:
                            im = cv2.imread(os.path.join(DATADIR, clone.filepath), cv2.IMREAD_GRAYSCALE)
                            clone.initialize_snake(im)
                            print "Fitting pedestal for " + clone.filebase
                            clone.fit_pedestal(im)
                            pedestal_data[clone.filebase] = [clone.pedestal, clone.iPedestal]

                            utils.append_pedestal_line(clone.filebase, pedestal_data[clone.filebase], os.path.join(ANALYSISDIR, "pedestal_final.txt"))

                        except Exception as e:
                            print "Failed to fit pedestal for " + clone.filebase + " because of " + str(e)

            
    #utils.save_clonelist(clones, ANALYSISDIR, "analysis_results_test.txt", cols)
