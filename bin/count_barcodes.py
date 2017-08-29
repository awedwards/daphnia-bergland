import os
import utils

DATADIR = "/mnt/spicy_4/daphnia/data/"

files = os.listdir(DATADIR)
barcodes = dict()
for f in files:
    if not f.startswith("._") and f.endswith(".bmp"):

        filetype, barcode,cloneid,treatment,replicate,rigId,datetime = utils.parse(f)
        
        try:
            if cloneid not in barcodes[barcode]:
                barcodes[barcode].append(cloneid)
                barcodes[barcode].append(f)
        except KeyError: barcodes[barcode] = [cloneid,f]
for k in barcodes.keys():
    if len(barcodes[k])>2:
        print k


