#!/usr/bin/env python3

import numpy as np
import glob
from joblib import dump, load
from Plot import PlotQuick
import ThreeD_Tracking as td


Directory = "/Users/keegan/Desktop/Research/visualisation/Gilad_Builds_and_data/real_det_data/second_set/"

BackFiles, SigFiles = [], []
SkyCountList, RealCountList, CountList = [], [], []

    
#BackFiles.extend([filename for filename in glob.iglob(Directory+'*Signal.txt', recursive=True)])
#SigFiles.extend([filename for filename in glob.iglob(Directory+'*Background.txt', recursive=True)])

BackFiles = [Directory+"MTS0{}Background.txt".format(i) for i in range(1,5)]
SigFiles = [Directory+"MTS0{}Signal.txt".format(i) for i in range(1,5)]

B_Data = dict((BackFiles[i], np.loadtxt(BackFiles[i])) for i in range(len(BackFiles)))
S_Data = dict((SigFiles[i], np.loadtxt(SigFiles[i])) for i in range(len(SigFiles)))


for i in range(len(B_Data)):
    
    RealCountList.append(S_Data[SigFiles[i]])
    SkyCountList.append(B_Data[BackFiles[i]])
    
#    PlotQuick(B_Data[BackFiles[i]],Save=True,Title="Background {}".format(i))
#    PlotQuick(S_Data[SigFiles[i]],Save=True,Title="Signal {}".format(i))
    
    Diff =  S_Data[SigFiles[i]] - B_Data[BackFiles[i]]
    Diff[Diff < 0] = 0
    CountList.append(Diff)
    
#    PlotQuick(Diff,Save=True,Title="Difference (S-B) {}".format(i))
    
#    Diff =  B_Data[BackFiles[i]] - S_Data[SigFiles[i]]
#    Diff[Diff < 0] = 0
#    CountList.append(Diff)
#
#    PlotQuick(Diff,Save=True,Title="Difference (B-S) {}".format(i+1))

RowSky = [td.ReadRowDataFileFastest('/Users/keegan/Desktop/Research/visualisation/home_versions/simulation_1.1.0/RowData/Sat_Jun_19_19-59-51_2021/RDS_[0,0]cm_20m_25cm_0a_0b_0c_[0,0,20]m.out',[500*i,0]) for i in [1.5, 0.5, -0.5, -1.5]] # use the same sky data for all of them
RowSkyList = RowSky ##for i in range(len(B_Data))]
Seperations = np.ones(4) * 25 # [cm]

ReadDict = {'Row Sky List':RowSkyList, \
            'Sky Count List':SkyCountList, \
            'Real Count List':RealCountList, \
            'Subtracted Count List':CountList, \
            'Seperations':Seperations}

#dump(ReadDict,"ReadDict3.joblib")
dump(ReadDict,"ReadDict3.joblib")



