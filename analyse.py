#!/usr/bin/env python3

import numpy as np
#import glob
from joblib import dump, load
from Plot import PlotQuick
import ThreeD_Tracking as td


analyse = True
skip = False

if not skip:
    if analyse:
    #    ReadDict = load("ReadDict2.joblib")
        ReadDict = load("ReadDict3.joblib")
        
        AnalyseDict = td.AnalyseData(ReadDict)
        
    #    dump(AnalyseDict,"AnalyseDict2.joblib")
        dump(AnalyseDict,"AnalyseDict3.joblib")
    
    else:
    #    ReadDict = load("ReadDict2.joblib")
    #    AnalyseDict = load("AnalyseDict2.joblib")
        AnalyseDict = load("AnalyseDict3.joblib")
    
    
    
    shape = np.shape(ReadDict['Subtracted Count List'][0])
    #print("shape is ",shape)
    
    
    ClusteredList = []
    HittingData = []
    ObjectCounts = []
    ObjectMax = []
    BinPointList = []
    ObjectOnes = []
    
    for i in range(len(AnalyseDict['All Indices'])):
        ClusteredArray = np.zeros(shape)
    
    #    for j in range(len(AnalyseDict['All Indices'][i])):
        
        for ind_tup in AnalyseDict['All Indices'][i]:
            ClusteredArray[ind_tup[0],ind_tup[1]] += \
                ReadDict['Subtracted Count List'][i][ind_tup[0],ind_tup[1]]
    
        PlotQuick(ReadDict['Subtracted Count List'][i],Save=True,Title='Original {}'.format(i))
        PlotQuick(ClusteredArray,Save=True,Title='Clustered {}'.format(i))
    
        ClusteredList.append(ClusteredArray)
        
        PixelHits = td.ClusteredHittingPoints(ReadDict['Row Sky List'][i], AnalyseDict['All Indices'][i], td.ClusterLayer, ReadDict['Seperations'][i])
        
        BinPoints = td.AlterHittingPoints(PixelHits, True, 1000, td.Which, ReadDict['Row Sky List'][i]['DetectorPos'])
                    
        TempCounts = td.ObjectView(PixelHits, td.ProjectionPixel, td.ObjectZ, td.ImageVolume, ReadDict['Seperations'][i])
    
    #    PlotQuick(TempCounts)
    
        HittingData.append(PixelHits)
        BinPointList.append(BinPoints)
        ObjectCounts.append(TempCounts)
        
        TempOnes = np.copy(TempCounts)
        TempOnes[TempOnes > 0] = 1

        ObjectOnes.append(TempOnes)
        ObjectMax.append(np.max(TempCounts))
    
    ObjectCounts = np.array(ObjectCounts)
    ObjectCounts = td.ScaleLayers(ObjectCounts, True)
    ObjectGroups, OverlapLists = td.GroupOverlaps(ObjectCounts)
    DetectorCounts, AddedMaxima, Targets = td.ScaleGroups(ObjectCounts, ObjectGroups, ObjectMax, td.OverlapCutoff) # Scales the array from each beam identically and  
                                                                                                             # creates Target without need for the classifier          
    
    AnalyseDict['Cluster Images'] = ClusteredList
    AnalyseDict['Hitting Data'] = HittingData
    AnalyseDict['Object Counts'] = ObjectCounts 
    AnalyseDict['Object Ones'] = ObjectOnes
    AnalyseDict['Detector Counts'] = DetectorCounts
    AnalyseDict['Targets'] = Targets
    AnalyseDict['Object Groups'] = ObjectGroups

#PlotQuick(np.sum(ObjectOnes,axis=0))

cut = 3

ObjectCuts = np.copy(np.sum(ObjectOnes,axis=0))
ObjectCuts[ObjectCuts < cut] = 0

AnalyseDict['Object Cuts'] = ObjectCuts

#PlotQuick(ObjectCuts)

td.ScatterDistance(ObjectCuts, td.Cutoff, td.ObjectZ, td.ImageVolume, ReadDict['Seperations'][0])
# td.ScatterDistance(np.sum(ObjectCounts,axis=0), td.Cutoff, td.ObjectZ, td.ImageVolume, ReadDict['Seperations'][0])
td.ScatterDistance(np.sum(ObjectCounts,axis=0) * ObjectCuts, td.Cutoff, td.ObjectZ, td.ImageVolume, ReadDict['Seperations'][0])

for i in range(len(AnalyseDict['Object Groups'])):   
        TempHitting = []
        
        for j in AnalyseDict['Object Groups'][i]:
            for k in range(len(AnalyseDict['Hitting Data'][j])):
                TempHitting.append(AnalyseDict['Hitting Data'][j][k])
            
        td.AlterHittingPoints(TempHitting, True, 1000, td.Which, [0,0])
    

dump(AnalyseDict,'AnalyseDict3_full.joblib')


# for k in range(td.ProjectionPixel[2]):
#     if np.sum(ObjectCuts[:,:,k]) != 0:
        
        
#VisDict = td.VisualiseObjects(AnalyseDict, ReadDict)

#dump(VisDict,"VisDict.joblib")
#VisDict = load("VisDict.joblib")

#PlotQuick(VisDict['Isolated Objects'])