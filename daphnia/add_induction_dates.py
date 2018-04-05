import os
import utils
from clone import Clone
from openpyxl import load_workbook
import datetime

print "Laading induction data\n"
inductiondates = dict()
inductionfiles = os.listdir(INDUCTIONMEDATADIR)

for i in inductionfiles:
    if not i.startswith("._") and (i.endswith(".xlsx") or i.endswith(".xls")):
        print i
        wb = load_workbook(os.path.join(INDUCTIONMEDATADIR,i),data_only=True)
        data = wb["Inductions"].values
        cols = next(data)[0:]
        data = list(data)
        df = pandas.DataFrame(data, columns=cols)
        df = df[df.ID_Number.notnull()]
        for j,row in df.iterrows():
            if not row['ID_Number'] == "NaT":
                induction = row['Induction_Date'].to_datetime()
                inductiondates[str(int(row['ID_Number']))] = induction.strftime('%Y%m%dT%H%M%S')

clones = utils.load_pkl(ANALYSIS_DIR
