#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 13:29:30 2020

@author: keegan
"""


# Standard
# import ROOT
import numpy as np
import datetime
import math
import os
import glob

# Plotting
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Machine Learning
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation

from joblib import dump, load
from scipy.optimize import curve_fit
import imageio



print("Hello Viewer!")

begin_time = datetime.datetime.now()



#---------------------------------- Special Commands --------------------------------------



#   %matplotlib qt5         <== run in shell to create interactive plot
#   %matplotlib inline      <== and this one to create an inline plot

#   imageio.mimsave('path/to/movie.gif', Data)      <== to create a gif out of Data (3D) (must be np.transpose of my convention)



#---------------------------------- Global Variables -----------------------------------------


#---------------------
# Detector Information
#---------------------
TriggerSize = 41.6                                              #[cm]
BarWidth = 3.2                                                  #[cm]
BarHight = 1.7                                                  #[cm]
NumOfBars = 27
PlanesSeperation = 25                                           #[cm]
TriggerWidth = 1                                                #[cm]

#---------------------
# Volume Specs
#---------------------

TopDepth = 2000                                                 #[cm] Distance from detector to surface
ImageLayerSize = [2000, 2000]                                   #[cm] Dimensions of the image layers 
ObjectZ = 0 #200                                                   #[cm] Distance from top of detector to first image layer
ImageVolume = [3000, 3000, TopDepth - ObjectZ]                  #[cm] Dimensions of image volume

#TopDepth = 4000
#ImageLayerSize = [2000, 2000]
#ImageVolume = [2000, 2000, TopDepth - ObjectZ]  

#---------------------
# Imaging 
#---------------------
ProjectionPixel = [143, 143, 30]                                #Resolution of images, ProjectionPixel[2] is number of image layers
Cutoff = [0.3, 0.5, 0.7, 0.8]                                   #Thresholds as fraction of maximum in ScatterDistance()

#ProjectionPixel = [143, 143, 30]   
 
#---------------------
# Clustering 
#---------------------
Divide = [16, 16]                                               #[X divisions, Y divisions] of image layers in LocalMaxIndices() and LayerCluster()
ClusterLayer = ProjectionPixel[2]-1                                                #Which layer the 2D clustering is done on
LocalCutoff = 0.3                                               #Cutoff as a fraction of maxima for LocalMaxIndices()
PercentCutoff = 0.05                                             #Cutoff as a fraction of maxima for ClusterAlgorithm()
OverlapCutoff = 0.4                                             #Cutoff as fraction of number of hitting points for determining overlap with ScaleGroups() 
LocalCutoff3D = 0.6                                            #Index cutoff for LayerCluster()
PercentCutoff3D = 0.35                                          #Clustering cutoff for LayerCluster()
 
                                                                #Take PercentCutoff3D as largest st. final output is non zero

#---------------------
# Run Options
#---------------------
Train = False                                                   #True ==> the data will be used to train the classifier (else: nothing)
PredictLayers = False                                           #True ==> classifier will classify the data (else: you will be asked)
Which = None                                                   #None ==> real analysis will be run. [i,j,k,l] all are 1 or 0 for manufactured beams             
Iterate = False                                                 #
dense = True
    

# --------------------------------- Translation of Gilad's Code -----------------------------------------



def ReadRowDataFileFastest(FileName,DetectorPos): #must use 'path/Filename' (including quoatations) in function call 
    
    #--------------------------------------------------------------------
    # Reads RowData.out files and produces a Dictionary with the 
    # information contained in the file
    #--------------------------------------------------------------------
    
    begin_time = datetime.datetime.now()
    
    G = np.loadtxt(FileName, dtype = str, delimiter = '~') #Delimiter '~' is chosen in order to avoid default whitespace delimiter
    Gsize = len(G)
    
    EventWordInd = []
    n = 0
    
    for i in range(Gsize):    
        if G[i] == '*Event*':
            n += 1
            EventWordInd.append(i)
    
    RowData = {'FileName':FileName,
               'NumberOfEvents':n,
               'DateAndTime':[],
               'DetectorPos':DetectorPos, #Coordinates (in x,y plane) of the Detector
               'BarsReadout': [[],[]],
               'UpperTrigPos':[[],[]],
               'LowerTrigPos':[[],[]]} #Trig Positions are left empty, can be filled if needed
    
    for i in range(n):
        RowData['DateAndTime'].append(G[EventWordInd[i]+2])   
       
    for i in range(n-1): # -1 to avoid error from length of last event 
        Bar = []
        Length = []
        
        for l in range(12):
            if 12 <= len(G[EventWordInd[i]+l]) <= 13: #Bounds depend delicately on precision of path lengths
                Bar.append(G[EventWordInd[i]+l][0:3])
                Length.append(G[EventWordInd[i]+l][4:])
    
        BarFloat = np.float_(Bar).tolist() #Converts string elements to float
        LengthFloat = np.float_(Length).tolist()
        
        RowData['BarsReadout'][0].append(BarFloat)
        RowData['BarsReadout'][1].append(LengthFloat)
        
    print('There were ', n,' events in ', FileName,', the simulation ran between ', \
          RowData['DateAndTime'][0],' - ',RowData['DateAndTime'][n-1],'.')
    
    tictoc = datetime.datetime.now() - begin_time
    print('It took ', tictoc,' to read the file.') #',FileName)

    return RowData



def CalcLocalPos(Bar,Length): #Arguments are BarsReadout elements from one event
    # LocalPos == [LocalX or LocalY, LocalZ]
    
    #--------------------------------------------------------------------
    # Determines position of the muon through each layer of scintillators
    # as well as outputs list of which bars produced the signal for each
    # event
    #--------------------------------------------------------------------
    
    LengthLocal = [[],[],[],[]]
    BarLocal = [[],[],[],[]]
    LocalPos = [[],[],[],[]]
    
    for n in range(len(Bar)): #Sorts Bar and Length data into nested lists corresponding to each layer on the detector
         LengthLocal[math.floor(Bar[n]/100)-1].append(Length[n])
         BarLocal[math.floor(Bar[n]/100)-1].append(Bar[n])
         
    a = np.sqrt(BarHight**2+(BarWidth/2)**2)
    Alpha = np.arctan(2*BarHight/BarWidth)
    
    for i in range(len(LengthLocal)):
        NumOfBarsLocal = len(LengthLocal[i])
        
        if NumOfBarsLocal == 0: # No signal, return error
            X,Z = -9999,-9999
        
        elif NumOfBarsLocal == 1: 
            #X,Z = BarWidth/2, BarHight/2
            
            if BarLocal[i][0]%2 == 0: #The first bar's vertex is facing down
                X,Z = 0, 0
            
            else:
                X,Z = 0, BarHight
            
            #Takes tip instead of middle of bar
            
            
        elif NumOfBarsLocal >= 2:
            Readout = []
            mxind = LengthLocal[i].index(max(LengthLocal[i]))
            
            if mxind == 0: #The first bar has the max readout so we take the first and second bars
                Readout = [LengthLocal[i][mxind],LengthLocal[i][mxind+1]]
            
            elif mxind == len(LengthLocal[i])-1: #The last bar has the max readout so we take the last bar and the one before
                Readout = [LengthLocal[i][mxind],LengthLocal[i][mxind-1]]
            
            else: #The max readout is somewhere at the middle, so we take this bar and it's highest neighbor
                Readout = [LengthLocal[i][mxind], np.amax([LengthLocal[i][mxind+1], LengthLocal[i][mxind-1]])]
            '''
            else: #The max readout is somewhere at the middle, so we take this bar and it's highest neighbor
                Readout = np.amax([[LengthLocal[i][mxind],LengthLocal[i][mxind+1]],\
                                   [LengthLocal[i][mxind],LengthLocal[i][mxind-1]]],axis = 0) 
            '''
            if BarLocal[i][0]%2 == 0: #The first bar's vertex is facing down
                X = BarWidth/2 - (a*Readout[0]/(Readout[0]+Readout[1]))*math.cos(Alpha)
                Z = BarHight/2 - (a*Readout[0]/(Readout[0]+Readout[1]))*math.sin(Alpha)
            
            else: #The first bar's vertex is facing up
                X = -BarWidth/2 + (a*Readout[1]/(Readout[0]+Readout[1]))*math.cos(Alpha)
                Z = BarHight/2 - (a*Readout[1]/(Readout[0]+Readout[1]))*math.sin(Alpha)
            
        LocalPos[i] = [X,Z] #[X/2 + np.random.random()*X, Z/2 + np.random.random()*Z] #randomize 50% (increases computing time)
            
    return LocalPos, BarLocal 



def CalcAbsPos(LocalPos,BarLocal, Seperation):
    #AbsPos[i] == [AbsX or AbsY, AbsZ] 
    
    #--------------------------------------------------------------------
    # Deterimines position relative to centre of the detector layer, 
    # and height measured from the bottom of the detector
    #--------------------------------------------------------------------
    
    AbsPos = [[0,0],[0,0],[0,0],[0,0]]
    
    if [-9999,-9999] in LocalPos:
        return -9999
        
    else:    
        for l in range(len(LocalPos)):  
            FirstBarIndex = BarLocal[l][0]
            i = math.floor(FirstBarIndex/100)
            
            if i == 1: #XUp
                AbsPos[l][1] = LocalPos[l][1] + TriggerWidth + Seperation + 3.5 * BarHight
                FirstBarIndex = FirstBarIndex - 100
           
            if i == 2: #YUp
                AbsPos[l][1] = LocalPos[l][1] + TriggerWidth + Seperation + 2.5 * BarHight
                FirstBarIndex = FirstBarIndex - 200
            
            if i == 3: #XDown
                AbsPos[l][1] = LocalPos[l][1] + TriggerWidth + 1.5 * BarHight
                FirstBarIndex = FirstBarIndex - 300
            
            if i == 4: #YDown
                AbsPos[l][1] = LocalPos[l][1] + TriggerWidth + 0.5 * BarHight
                FirstBarIndex = FirstBarIndex - 400
            
            
            if FirstBarIndex%2 == 0 : #The first bar's vertex is facing down
                AbsPos[l][0] = LocalPos[l][0] - (NumOfBars / 4 - 0.25) * BarWidth + BarWidth / 2 * FirstBarIndex
                
            else: #The first bar's vertex is facing up
                AbsPos[l][0] = LocalPos[l][0] - (NumOfBars / 4 - 0.25) * BarWidth + BarWidth / 2 * (FirstBarIndex + 1) 
                
        return AbsPos 



def CalcEventHittingPoints(Bar, Length, ZImage, DetectorPos, Seperation): #( <RowData>['BarsReadout'][0][i], <RowData>['BarsReadout'][1][i], ...)
    
    #--------------------------------------------------------------------
    # Determines where on each plane (z = const.) the muon passed through 
    #--------------------------------------------------------------------
    
    HittingPoints = np.zeros(9).reshape((3,3))
    
    [LocalPos,BarLocal] = CalcLocalPos(Bar,Length)
    
    AbsPos = CalcAbsPos(LocalPos,BarLocal, Seperation)
    
    if AbsPos == -9999:
        HittingPoints = [[-9999]*3]*3
    
    else:    
        AbsXUp = AbsPos[0]
        AbsYUp = AbsPos[1]
        AbsXDown = AbsPos[2]
        AbsYDown = AbsPos[3]
        
        ZUp = 2 * TriggerWidth + 4 * BarHight + Seperation
        ZSurf = TopDepth
        
        dZx = AbsXUp[1] - AbsXDown[1]
        dX = AbsXUp[0] - AbsXDown[0]
        
        dZy = AbsYUp[1] - AbsYDown[1]
        dY = AbsYUp[0] - AbsYDown[0]
        
        if dX != 0 and dY != 0:
            ax = dZx / dX  
            ay = dZy / dY
        
            bx = AbsXUp[1] - ax * AbsXUp[0]
            by = AbsYUp[1] - ay * AbsYUp[0]
            
            HittingPoints[0][0] = (ZImage - bx) / ax + DetectorPos[0] #XImage
            HittingPoints[0][1] = (ZImage - by) / ay + DetectorPos[1] #YImage
            HittingPoints[0][2] = ZImage #ZImage
            
            HittingPoints[1][0] = (ZUp - bx) / ax + DetectorPos[0] #XUp
            HittingPoints[1][1] = (ZUp - by) / ay + DetectorPos[1] #YUp
            HittingPoints[1][2] = ZUp #ZUp
            
            HittingPoints[2][0] = (ZSurf - bx) / ax + DetectorPos[0] #XSurf
            HittingPoints[2][1] = (ZSurf - by) / ay + DetectorPos[1] #YSurf
            HittingPoints[2][2] = ZSurf #ZSurf
            
        else:
            HittingPoints = [[-9999]*3]*3
    
    return HittingPoints 



def PazAnalysis(RowData, Seperation, Iterate):
    
    #--------------------------------------------------------------------
    # Creates a 3D array counting the number of trajectories passsing 
    # through each pixel in the image layers specified by the global
    # variables 
    #--------------------------------------------------------------------
    
    begin_time = datetime.datetime.now()
    
    N = RowData['NumberOfEvents']
    
    if Iterate == True:
        DetectorCounts = np.zeros((ProjectionPixel[0], ProjectionPixel[1], ProjectionPixel[2])) 
        
        for k in range(N-1):
            for i in range(ProjectionPixel[2]):
                ZImage = 2 * TriggerWidth + 4 * BarHight + Seperation + i*TopDepth/ProjectionPixel[2] #Z coordinate of Image Layer
                HittingPoints = CalcEventHittingPoints(RowData['BarsReadout'][0][k], RowData['BarsReadout'][1][k], ZImage, RowData['DetectorPos'], Seperation)
                
                if np.all(HittingPoints == -9999):
                    DetectorCounts = -9999
                    break
                
                else:  
                    Iind = int(round((HittingPoints[0][0] + ImageLayerSize[0]/2)*(ProjectionPixel[0]-1)/ImageLayerSize[0])) 
                    Jind = int(round((HittingPoints[0][1] + ImageLayerSize[1]/2)*(ProjectionPixel[1]-1)/ImageLayerSize[1]))
                    
                    if Iind>0 and Jind>0 and Iind<len(DetectorCounts[0]) and Jind<len(DetectorCounts[1]): 
                            DetectorCounts[Iind,Jind,i] += 1 
        
    else:
        DetectorCounts = np.zeros((ProjectionPixel[0], ProjectionPixel[1])) 
        
        for k in range(N-1):
            ZImage = 2 * TriggerWidth + 4 * BarHight + Seperation + ClusterLayer*TopDepth/ProjectionPixel[2] #Z coordinate of Image Layer
            HittingPoints = CalcEventHittingPoints(RowData['BarsReadout'][0][k], RowData['BarsReadout'][1][k], ZImage, RowData['DetectorPos'], Seperation)
            
            if np.all(HittingPoints == -9999):
                DetectorCounts = -9999
                break
            
            else:  
                Iind = int(round((HittingPoints[0][0] + ImageLayerSize[0]/2)*(ProjectionPixel[0]-1)/ImageLayerSize[0])) 
                Jind = int(round((HittingPoints[0][1] + ImageLayerSize[1]/2)*(ProjectionPixel[1]-1)/ImageLayerSize[1]))
                
                if Iind>0 and Jind>0 and Iind<len(DetectorCounts[0]) and Jind<len(DetectorCounts[1]): 
                        DetectorCounts[Iind,Jind] += 1 
    
    tictoc = datetime.datetime.now() - begin_time
    print('It took ', tictoc,' to analyse the RowData from ', RowData['FileName'])
    
    return DetectorCounts 



# -------------------------- Plotting Functions -----------------------------------------


    
def ScatterDistance(Data, Cutoff, ObjectZ, ImageVolume, Seperation):
    #Expects 3D array
    #Computing time grows very quickly with number of data points
    
    #--------------------------------------------------------------------
    # Plots Data by mapping each element to a point in space and color
    # coordinates the values using Cutoff variable. Not for use with 
    # the unprocessed output of PazAnalysis (too many points to compute)
    #--------------------------------------------------------------------
    
    Max = np.max(Data)
    
    fig =  plt.figure(figsize=(15,15))
    ax = fig.gca(projection='3d')
    ax.set(xlim=(-ImageVolume[0]/2, ImageVolume[0]/2), ylim=(-ImageVolume[1]/2, ImageVolume[1]/2)) 
    ax.set_zlim(0, TopDepth)
    ax.view_init(elev=20, azim=0)
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('depth [cm]')

    
    cols = ['b','c', 'g', 'y', 'w']
    # cols.reverse()
    
    Shape = np.shape(Data)
    
#    d = Shape[0] * Shape[1] * Shape[2]
    
#    Nones = np.array([None]*d,dtype='str') #[None for i in range(d)] # list of labels for data points
#    print(Nones)
    Nones = np.array(np.ones(len(cols)),dtype='str')
    
    Nones[0] = ' blue > {}'.format(Cutoff[3])
    Nones[1] = ' {} > cyan > {}'.format( Cutoff[3], Cutoff[2])
    Nones[2] = ' {} > green > {}'.format( Cutoff[2], Cutoff[1])
    Nones[3] = ' {} > yellow > {}'.format( Cutoff[1], Cutoff[0])
    Nones[4] = ' {} > white'.format(Cutoff[0])
    
#    print(Nones)
    
#    ax.scatter(x, y, z, cmap=cm.coolwarm, c=col, marker='s', s=80, linewidth=0, alpha=opac, label=lbl)
    
    for j in range(Shape[2]):
        z = 2 * TriggerWidth + 4 * BarHight + Seperation + ObjectZ + j*ImageVolume[2]/Shape[2]
        
        for k in range(Shape[0]):
            for l in range(Shape[1]):
                
                if Data[k,l,j] != 0:
                    x = k*ImageVolume[0]/Shape[0] - ImageVolume[0]/2 
                    y = l*ImageVolume[1]/Shape[1] - ImageVolume[1]/2
                    
                    if Data[k][l][j] > Max*Cutoff[3]:
                        col = cols[0]
                        opac = 0.5
                    
                    elif Data[k][l][j] > Max*Cutoff[2]:
                        col = cols[1]
                        opac = 0.5
                       
                    elif Data[k][l][j] > Max*Cutoff[1]:
                        col = cols[2]
                        # opac = 0.5
                        opac = 0.4
                    
                    elif Data[k][l][j] > Max*Cutoff[0]:
                        col = cols[3]
                        # opac = 0.5
                        opac = 0.3
                        
                    else:
                        col = cols[3]
                        # opac = 0.5
                        opac = 0.2
                    
#                    lbl = Nones[l + k * Shape[0] + j * Shape[0] * Shape[1]]
                    
#                    if j == 0 and k == 0 and l == 0:
#                        for i in range(4):
#                            print(Nones[i])
#                            ax.scatter(x, y, z, cmap=cm.coolwarm, c=col, marker='s', s=80, linewidth=0, alpha=opac, label=Nones[i])
                        
#                    else:
                    ax.scatter(x, y, z, cmap=cm.coolwarm, c=col, marker='s', s=80, linewidth=0, alpha=opac)
                        
                        
#    legend1 = ax.legend(*ax.legend_elements(),
#                    loc="lower left", title="Classes")
#    ax.add_artist(legend1)
    ax.text2D(0.95, 0.95, "Percent of Global Max\n"+("{}\n"*len(cols)).format(*Nones), transform=ax.transAxes)
   
    # plot detector positions
    [ax.plot([500*i],[0],[0],marker='x',color='r') for i in [1.5, 0.5, -0.5, -1.5]]
        
#    ax.legend()
    plt.ion()
    plt.show()
    
#%%    
    
from joblib import load

def PlotQuick(Data, ThreeD):   
    #Expects Boolean value for ThreeD
    
    #--------------------------------------------------------------------
    # Plots each image layer in Data as a list of colormapped images
    #--------------------------------------------------------------------    
    
    def PlotQuick2D(Data):
        
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(1,1,1)
        
        #ax.set_title('PazVsReal')
        plt.imshow(Data, alpha = 0.5)
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.patch.set_alpha(0)
        ax.set_frame_on(False)
        plt.colorbar(orientation='vertical')
        
        plt.show()
   
    
    if ThreeD == True:
        Shape = np.shape(Data)
        
        for i in range(Shape[2]):
            PlotQuick2D(Data[:,:,i])
    
    else:
        PlotQuick2D(Data)
    
#%%

def AlterHittingPoints(PixelHits, Plot, DetectHits, Which, DetectorPosition): 
    #Set DetectHits = -1 for analysis or to view all trajectories
    
    #-----------------------------------------------------------------
    # Outputs Hitting points at top and bottom of decay volume whose 
    # trajectories go through an element of ClusterIndices, also 
    # includes manipulated beams generated from those trajectories 
    # (if Which != None)
    #-----------------------------------------------------------------
    
    def BinPoints(PixelHits, BinData, ax, Tilt, Plot, col, opac): # Changes hitting points from Pixel hits to manipulate trajectory beams 
        for i in range(len(PixelHits)):                                                              # and plots the beam if Plot == True
            # x = np.array([PixelHits[i][0][1][0] + Tilt[0][0] + DetectorPosition[0], PixelHits[i][0][2][0] + Tilt[1][0] + DetectorPosition[0]])
            # y = np.array([PixelHits[i][0][1][1] + Tilt[0][1] + DetectorPosition[1], PixelHits[i][0][2][1] + Tilt[1][1] + DetectorPosition[1]])  
            x = np.array([PixelHits[i][0][1][0] + Tilt[0][0], PixelHits[i][0][2][0] + Tilt[1][0]])
            y = np.array([PixelHits[i][0][1][1] + Tilt[0][1], PixelHits[i][0][2][1] + Tilt[1][1]])  
            z = np.array([PixelHits[i][0][1][2], PixelHits[i][0][2][2]])
            BinData.append([x,y,z])
            
            if Plot == True:
                ax.plot(x,y,z,col,alpha=opac) 
                
    np.random.shuffle(PixelHits)
    
    PixelHits = PixelHits[:DetectHits] #Reduces number of data points to handle interactive plotting and reduce computing time
    
    BinData = []
    
    if Plot == True:
        fig =  plt.figure(figsize=(15,15))
        ax = fig.gca(projection='3d')
        ax.set(xlim=(-ImageVolume[0]/2, ImageVolume[0]/2), ylim=(-ImageVolume[1]/2, ImageVolume[1]/2)) 
        ax.set_zlim(0, TopDepth)
        ax.view_init(elev=20, azim=0)
        ax.set_xlabel('x [cm]')
        ax.set_ylabel('y [cm]')
        ax.set_zlabel('depth [cm]')

        
    else:
        ax = None
    
    if Which != None: #Manufactures beams with altered positions from 1 detector. Creates artificial interference data (2 detectors 4 objects)
        if Which[0] == 1:
            BinPoints(PixelHits, BinData, ax, [[500,0],[-1100,500]], Plot, 'y', 0.2)    
            BinPoints(PixelHits, BinData, ax, [[-500,0],[-700,500]], Plot, 'b', 0.2) 
        
        if Which[1] == 1:
            BinPoints(PixelHits, BinData, ax, [[500,0],[-1100,-500]], Plot, 'y', 0.2)    
            BinPoints(PixelHits, BinData, ax, [[-500,0],[-700,-500]], Plot, 'b', 0.2)
        
        if Which[2] == 1:
            BinPoints(PixelHits, BinData, ax, [[500,0],[0,0]], Plot, 'y', 0.2)    
            BinPoints(PixelHits, BinData, ax, [[-500,0],[900,0]], Plot, 'b', 0.2) 
        
        if Which[3] == 1:
            BinPoints(PixelHits, BinData, ax, [[500,0],[0,-900]], Plot, 'y', 0.2)    
            BinPoints(PixelHits, BinData, ax, [[-500,0],[900,-900]], Plot, 'b', 0.2)
    
    else:
        #for i in range(len(PixelHits)):
        # BinPoints(PixelHits, BinData, ax, [[DetectorPosition[0],DetectorPosition[1]],[DetectorPosition[0],DetectorPosition[1]]], Plot, 'b', 0.5)
        BinPoints(PixelHits, BinData, ax, [[0,0],[0,0]], Plot, 'b', 0.5)

    if Plot == True:
        
        # Plot detector positions
        [ax.plot([500*i],[0],[0],marker='x',color='r') for i in [1.5, 0.5, -0.5, -1.5]]
        
        plt.ion()
        plt.show()
        
        
    return BinData 

    

# -------------------------- Clustering Algorithms -----------------------------------------



def ClusterAlgorithm(Data, Threshold, StartIndex):
    #Expects 2D array
    
    #--------------------------------------------------------------------
    # Isolates regions in Data above the threshold given the StartIndex
    #--------------------------------------------------------------------
    
    def Check(ActiveIndices, Shift, Zeros, Temp): # Checks to see if the pixel Shift-ed relative to ActiveIndices[j][k] is above Threshold
        
        if 0 <= ActiveIndices[j][k][0] + Shift[0] < Shape[0] and 0 <= ActiveIndices[j][k][1] + Shift[1] < Shape[1]:
            if Data[ActiveIndices[j][k][0] + Shift[0], ActiveIndices[j][k][1] + Shift[1]] < Threshold:   
                if Data[ActiveIndices[j][k][0] + Shift[0], ActiveIndices[j][k][1] + Shift[1]] == 0:
                    Zeros += 1
                
            elif not [ActiveIndices[j][k][0] + Shift[0], ActiveIndices[j][k][1] + Shift[1]] in ActiveIndices[j-1]:
                if not [ActiveIndices[j][k][0] + Shift[0], ActiveIndices[j][k][1] + Shift[1]] in Temp:
                    Temp.append([ActiveIndices[j][k][0] + Shift[0], ActiveIndices[j][k][1] + Shift[1]])
            
        return Zeros, Temp
    
    ActiveIndices = [[StartIndex]]  
    Shape = np.shape(Data)
    ClusteredData = np.zeros((Shape[0], Shape[1]))
    
    i = 0
    j = 0
    while i == 0:
        Temp = []
        LayerZeros = 0
        
        for k in range(len(ActiveIndices[j])):   
                                             
            Zeros = 0
            
            Zeros, Temp = Check(ActiveIndices, [1,0], Zeros, Temp)
            Zeros, Temp = Check(ActiveIndices, [0,1], Zeros, Temp)
            Zeros, Temp = Check(ActiveIndices, [-1,0], Zeros, Temp)
            Zeros, Temp = Check(ActiveIndices, [0,-1], Zeros, Temp)
            
            if Zeros >= 2:
                LayerZeros += 1
        
            if len(Temp) > 1000 or k > 1000:
                i += 1
                break
        
        if len(Temp) == 0:
            i += 1
            
        elif LayerZeros <= len(ActiveIndices[j]):
            ActiveIndices.append(Temp)
            j += 1
        
        else: 
            i += 1
        
    for layer in ActiveIndices:
        for pixel in layer:
            ClusteredData[pixel[0], pixel[1]] = Data[pixel[0], pixel[1]]
    
    ClusterDict = {'Active Indices':ActiveIndices, \
                   'Clustered Array':ClusteredData, \
                   'Start Index':StartIndex, \
                   'Threshold Value':Threshold}
        
    return ClusterDict



def LocalMaxIndices(Data, LocalCutoff, Divide):
    #Expects 2D arrays for Data
        
    #--------------------------------------------------------------------
    # Finds extrema above the LocalCutoff in the regions or Data 
    # specified by Divide and outputs the corresponding Index and Value
    #--------------------------------------------------------------------
    
    Shape = np.shape(Data)
    
    DivX = Divide[0]
    DivY = Divide[1]
    
    IndListX = np.linspace(0, Shape[0], num=DivX).astype(int)
    IndListY = np.linspace(0, Shape[1], num=DivY).astype(int)
    
    Value = []
    Index = []
    LayerMaxima = np.max(Data)
    
    for j in range(DivX -1):
        for k in range(DivY - 1):
            Max = np.max(Data[IndListX[j]:IndListX[j+1],IndListY[k]:IndListY[k+1]])  
            
            if  LayerMaxima * LocalCutoff: #Max > LocalCutoff: #
#                Temp = []
#                for l in range(len(np.where(Data[IndListX[j]:IndListX[j+1],IndListY[k]:IndListY[k+1]] == Max)[0])): 
#                    TempInd = [IndListX[j] + np.where(Data[IndListX[j]:IndListX[j+1],IndListY[k]:IndListY[k+1]] == Max)[0][l], \
#                               IndListY[k] + np.where(Data[IndListX[j]:IndListX[j+1],IndListY[k]:IndListY[k+1]] == Max)[1][l]] 
#                                                                                                    
#                    if not TempInd in Temp:
#                        Temp.append(TempInd) 
        
                #only take first maxima in the region
                
                Temp = [IndListX[j] + np.where(Data[IndListX[j]:IndListX[j+1],IndListY[k]:IndListY[k+1]] == Max)[0][0], \
                        IndListY[k] + np.where(Data[IndListX[j]:IndListX[j+1],IndListY[k]:IndListY[k+1]] == Max)[1][0]]
        
                Value.append(Max)
                Index.append(Temp)
    
    return Value, Index



def ClusterMaxima(Data, Value, Index, PercentCutoff):
    #Expects 2D array
    
    #--------------------------------------------------------------------
    # Runs the ClusterAlgorithm at each element of Index and provided the 
    # results have more than five nonzero pixels, then combines them to  
    # form one image
    #--------------------------------------------------------------------
    
    AllClusters = []
    Area = []
    ActiveIndices = []
    
    LayerMaxima = np.max(Data)
    
    for i in range(len(Index)):
        
        for l in range(len(Index[i])):
            Dict = ClusterAlgorithm(Data, LayerMaxima*PercentCutoff, [Index[i][l][0],Index[i][l][1]]) 
            
            if np.count_nonzero(Dict['Clustered Array']) > 5: # Requires two full layers of clustered Pixels 
                AllClusters.append(Dict['Clustered Array']) # ( 1 start pixel + 4 adjacent to it) to be considered significant
                n = np.count_nonzero(AllClusters)
                
                for layer in Dict['Active Indices']:
                    for ind in layer:
                        if not ind in ActiveIndices and len(Dict['Active Indices']) > 1:
                            ActiveIndices.append(ind)
                
                Area.append(n)
        
    AllClusters = np.transpose(AllClusters)
    
    Shape = np.shape(Data)
    
    Maximized = np.zeros((Shape[1],Shape[0]))
    
    if len(AllClusters) != 0:
        for k in range(Shape[0]):
            for l in range(Shape[1]):  
                Maximized[k][l] = np.max(AllClusters[k][l]) 
                    
    Maximized = np.transpose(Maximized) 
    
    LayerDict = {'Active Indices':ActiveIndices, \
                 'Clustered Array':Maximized, \
                 'Start Value':Value, \
                 'Start Index':Index, \
                 'Area':Area}
    
    return LayerDict



# ---------------------------------- Tracking -----------------------------------------



def TrackPixel(RowData, Indices, Color, opac, Seperation): 
    #Choose indices from a 3D DetectorCount array for Indices
    
    #-----------------------------------------------------------------
    # Plots trajectories through one pixel in DetectorCounts
    # Could be useful in the future for displaced vertex reconstruction
    #-----------------------------------------------------------------
    
    begin_time = datetime.datetime.now()
    
    PixelHits = []
    N = RowData['NumberOfEvents']
    
    for k in range(N-1): 
        ZImage = 2 * TriggerWidth + 4 * BarHight + Seperation + Indices[2]*TopDepth/ProjectionPixel[2]
        HittingPoints = CalcEventHittingPoints(RowData['BarsReadout'][0][k], RowData['BarsReadout'][1][k], ZImage, RowData['DetectorPos'])
            
        if np.all(HittingPoints == -9999):
            continue
        
        else:  
            Iind = int(round((HittingPoints[0][0] + ImageLayerSize[0]/2)*(ProjectionPixel[0]-1)/ImageLayerSize[0]))
            Jind = int(round((HittingPoints[0][1] + ImageLayerSize[0]/2)*(ProjectionPixel[1]-1)/ImageLayerSize[1]))
        
        #if Iind > 0 and Jind > 0: #Plots all trajectories
        if Iind == Indices[0] and Jind == Indices[1] and Iind > 0 and Jind > 0:
            PixelHits.append(HittingPoints)
    
    fig =  plt.figure(figsize=(15,15))
    ax = fig.gca(projection='3d')
    ax.set(xlim=(-ImageLayerSize[0]/2, ImageLayerSize[0]/2), ylim=(-ImageLayerSize[1]/2, ImageLayerSize[1]/2)) 
    ax.set_zlim(0, TopDepth)
        
    for i in range(len(PixelHits)):
        x = np.array([PixelHits[i][0][0],PixelHits[i][1][0],PixelHits[i][2][0]]) 
        y = np.array([PixelHits[i][0][1],PixelHits[i][1][1],PixelHits[i][2][1]]) 
        z = np.array([PixelHits[i][0][2],PixelHits[i][1][2],PixelHits[i][2][2]]) 
        
        ax.plot(x,y,z,Color,alpha=opac)
    
    plt.ion()
    plt.show()
    
    tictoc = datetime.datetime.now() - begin_time
    print('It took ', tictoc,' to track the data')



def ClusteredHittingPoints(RowData, ClusterIndices, Layer, Seperation):  
    #Expects ClusterDict['Active Indices'] for Indices
    #Layer corresponds to the image layer used to create ClusterIndices

    #-----------------------------------------------------------------
    # Determines hitting points of trajectories that go through the 
    # clustered image
    #-----------------------------------------------------------------

    begin_time = datetime.datetime.now()
    
    N = RowData['NumberOfEvents']
    
    Pos = RowData['DetectorPos'] #[cm]
    
#    IndexList = []
#    for i in range(len(ClusterIndices)): 
#        if Which == False:
#            IndexList.append(ClusterIndices[i])
#    
#        else:
#            for j in range(len(ClusterIndices[i])):
#                IndexList.append(ClusterIndices[i][j])
        
    PixelHits = []
    PixelIndex = []
    
    for k in range(N-1):  
        ZImage = 2 * TriggerWidth + 4 * BarHight + Seperation + Layer * TopDepth / ProjectionPixel[2]
        
        HittingPoints = CalcEventHittingPoints(RowData['BarsReadout'][0][k], RowData['BarsReadout'][1][k], ZImage, RowData['DetectorPos'], Seperation)
#        HittingPoints = CalcEventHittingPoints(RowData['BarsReadout'][0][k], RowData['BarsReadout'][1][k], ZImage, [0,0], Seperation)
        
#        print("Hitting points are: ",HittingPoints)
        
        if np.all(HittingPoints == -9999):
            continue

        else:
#            Iind = int(round((HittingPoints[0][0] + ImageVolume[0]/2 )*(ProjectionPixel[0]-1)/ImageLayerSize[0])) 
#            Jind = int(round((HittingPoints[0][1] + ImageVolume[0]/2 )*(ProjectionPixel[1]-1)/ImageLayerSize[1])) 
            Iind = int(round((HittingPoints[0][0] - Pos[0] + ImageLayerSize[0]/2)*(ProjectionPixel[0]-1)/ImageLayerSize[0])) 
            Jind = int(round((HittingPoints[0][1] - Pos[1] + ImageLayerSize[0]/2)*(ProjectionPixel[1]-1)/ImageLayerSize[1])) 


#        if [Iind,Jind] in IndexList and Iind > 0 and Jind > 0:
        if (Iind,Jind) in ClusterIndices and Iind > 0 and Jind > 0:
            PixelIndex.append([Iind,Jind]) 
            PixelHits.append([HittingPoints]) # + [RowData['DetectorPos'][0],RowData['DetectorPos'][1],0]])
            
            ##### use hitting points rotate and return thos instead
            '''
            y = lowest hitting point
            y' = highest hitting point
            x = y' - y # so traj is line segment through the origin
            x' = R x # rotate the line segment according to detector orientation
            x'' = x' + y # translate it back to original position
            
            x'' and y together give the track for the muon
            
            '''
            
            
    
    tictoc = datetime.datetime.now() - begin_time
    print('It took ', tictoc,' to track the data')

    return PixelHits 

                   

# -------------------------- Iterative Beam Imaging -----------------------------------------



def ObjectView(Data, Resolution, ObjectZ, ImageVolume, Seperation):
    
    #-----------------------------------------------------------------
    # Takes Data hitting points and counts hits in the ImageVolume at 
    # ObjectZ above the detector with resolution Resolution
    #-----------------------------------------------------------------
    
    DetectorCounts = np.zeros((Resolution[0],Resolution[1],Resolution[2]))
    
    for i in range(len(Data)):
        AbsXUp = Data[i][0][2][0]
        AbsYUp = Data[i][0][2][1]  
        AbsXDown = Data[i][0][1][0]
        AbsYDown = Data[i][0][1][1]
        
        ZUp = Data[i][0][2][2]             
        ZDown = Data[i][0][1][2]
        dZ = ZUp - ZDown
        
        if dZ == 0:
            continue                    
        
        dX = AbsXUp - AbsXDown     
        ax = dZ / dX                           
        bx = ZUp - ax * AbsXUp         
        
        dY = AbsYUp - AbsYDown
        ay = dZ / dY
        by = ZUp - ay * AbsYUp
        
        for j in range(Resolution[2]):
            ZImage = 2 * TriggerWidth + 4 * BarHight + Seperation + ObjectZ + j*ImageVolume[2]/Resolution[2]
            
            HittingPoints = np.zeros(3)
            
            HittingPoints[0] = (ZImage - bx) / ax #XImage  
            HittingPoints[1] = (ZImage - by) / ay #YImage
            HittingPoints[2] = ZImage #ZImage
            
            Iind = int(round((HittingPoints[0] + ImageVolume[0]/2)*(Resolution[0]-1)/ImageVolume[0]))
            Jind = int(round((HittingPoints[1] + ImageVolume[1]/2)*(Resolution[1]-1)/ImageVolume[1]))
#            Iind = int(round((HittingPoints[0] + ImageLayerSize[0]/2)*(Resolution[0]-1)/ImageLayerSize[0]))
#            Jind = int(round((HittingPoints[1] + ImageLayerSize[1]/2)*(Resolution[1]-1)/ImageLayerSize[1]))
#            Iind = int(round((HittingPoints[0])*(Resolution[0]-1)/ImageLayerSize[0]))
#            Jind = int(round((HittingPoints[1])*(Resolution[1]-1)/ImageLayerSize[1]))
        
            if Iind > 0 and Jind > 0 and Iind < len(DetectorCounts[0]) and Jind < len(DetectorCounts[1]):
                DetectorCounts[Iind][Jind][j] += 1
                    
    return DetectorCounts 


    
#---------------------------------- Miscellaneous Analysis Functions --------------------------------------


        
def GroupOverlaps(Data):
    #Expects 4D array for Data
    
    #--------------------------------------------------------------------
    # Checks to see if each pair of object data arrays overlap by seeing 
    # if subtracting one from the other changes any of the values in the 
    # first array.  
    # Then forms groups of overlapping sets of data (each of which is 
    # interpretted as an object)
    #--------------------------------------------------------------------
    
    Data = np.array(Data)       
           
    N = len(Data)
    OverlapLists = []
    
    for i in range(N):
        for j in range(N):
            if i != j: 
                Minus = (Data[i] - Data[j])
                Minus[Minus < 0] = 0 # sets negative pixels to zero
    
                if np.any(Data[i] != Minus):
                    OverlapLists.append([i,j])    
    
    ObjectGroups = []
    
    for i in range(len(OverlapLists)):
        if len(ObjectGroups) == 0:
            ObjectGroups.append([OverlapLists[i][0],OverlapLists[i][1]])
    
        Appended = 0 #Used to keep track of whether element is already in ObjectGroups across if gates
    
        for j in range(len(ObjectGroups)):
            if not OverlapLists[i][0] in ObjectGroups[j]:
                if OverlapLists[i][1] in ObjectGroups[j]:
                    ObjectGroups[j].append(OverlapLists[i][0])
                    Appended += 1                
    
            elif not OverlapLists[i][1] in ObjectGroups[j]:
                if OverlapLists[i][0] in ObjectGroups[j]:
                    ObjectGroups[j].append(OverlapLists[i][1])
                    Appended += 1
            
            if OverlapLists[i][0] in ObjectGroups[j]:
                if OverlapLists[i][1] in ObjectGroups[j]:
                    Appended += 1
                    
        if Appended == 0:
            ObjectGroups.append([OverlapLists[i][0],OverlapLists[i][1]])
            
    return ObjectGroups, OverlapLists
        


def ScaleGroups(Counts, Groups, Maxima, OverlapCutoff):
    
    #--------------------------------------------------------------------
    # Scales all beams identically so regions of interference can be            
    # identified and decides whether pairs of 3D arrays overlap to 
    # create Target. Removes the need for the classification algorithm 
    # with this step.
    #--------------------------------------------------------------------
    
    DetectorCounts = np.zeros((len(Groups),ProjectionPixel[0],ProjectionPixel[1],ProjectionPixel[2]))
    AddedMaxima = []
    Targets = np.zeros((len(Groups),ProjectionPixel[2]))
    
    for i in range(len(Groups)):
        Temp = []
        
        for j in Groups[i]: # Decides how many pairs of beams overlap at each layer by permuting them within their groups
            for k in Groups[i]:
                if j != k: 
                    for m in range(ProjectionPixel[2]):
                        Minus = (Counts[j,:,:,m] - Counts[k,:,:,m])
                        Minus[Minus < 0] = 0
            
                        #if np.any(Counts[j,:,:,m] != Minus):
                        if abs(np.sum(Counts[j,:,:,m] - Minus)) > np.max(Counts[j,:,:,m]) * OverlapCutoff: 
                            Targets[i,m] += 1
                        
                        else:
                            Targets[i,m] += 2
                        
            if Maxima[j] != 0 and not Maxima[j] in Temp: # assumes that if maxima are the same then clustered array is a duplicate
                ScaledData = Counts[j] * np.max(Maxima)/Maxima[j] 
            
                DetectorCounts[i] += ScaledData
    
                Temp.append(Maxima[j])
        
        AddedMaxima.append(Temp)
        
    return DetectorCounts, AddedMaxima, Targets



def ScaleLayers(Counts, Scale):
    
    if Scale == True:
        Shape = np.shape(Counts)
        TempCounts = np.zeros(Shape)
        '''
        Counts[Counts > 0] = 100
        
        TempCounts = Counts
        '''
        for i in range(Shape[0]):
            Max = np.max(Counts[i])
            
            for j in range(ProjectionPixel[2]):
                    TempCounts[i,:,:,j] = Counts[i,:,:,j] * Max / np.max(Counts[i,:,:,j]) if np.max(Counts[i,:,:,j]) != 0 \
                                                                                        else Counts[i,:,:,j]

    else:
        TempCounts = Counts
    
    return TempCounts
    


#---------------------------------- Run Functions --------------------------------------
    


def ReadDataFiles():
    '''
    print('Do you want to use the last input? \n If so, input "yes" otherwise input any string')

    Input = input()
    
    if Input == 'yes':
        RealFiles = load('RealFiles.joblib')
        SkyFiles = load('SkyFiles.joblib')
        XPositions = load('XPositions.joblib')
        YPositions = load('YPositions.joblib')
        Seperations = load('Seperations.joblib')
        
    else:
        RealFiles = []
        SkyFiles = []
        XPositions = []
        YPositions = []
        Seperations = []
        
        Input = ' '
        
        while Input != 'done':
            print('\n Input: /path/to/Real_RowData.out: ') 
            RealFiles.append(input())
            
            print('\n Input: /path/to/Sky_RowData.out')
            SkyFiles.append(input())
        
            print('\n Input detector X position [cm]')
            XPositions.append(float(input()))
            
            print('\n Input detector Y position [cm]')
            YPositions.append(float(input()))
            
            print('\n Input layer seperation [cm]')
            Seperations.append(float(input()))
            
            print('\n To add more data, input any string \n', 'otherwise input "done"')
            Input = input()
     
        dump(RealFiles,'RealFiles.joblib')
        dump(SkyFiles,'SkyFiles.joblib')
        dump(XPositions,'XPositions.joblib')
        dump(YPositions,'YPositions.joblib')
        dump(Seperations,'Seperations.joblib')
        
    '''    
    XPositions = [500, -500, 330]
    YPositions = [-500, 500, 330]
    
    Seperations = [25, 25, 25]
    
    Dir = '/Users/keegan/Desktop/Research/visualisation/useful_data/Fri_May_14_07-01-29_2021/'
    
    RealFiles = [Dir+name for name in ['RDR_[500,-500]cm_30m_25cm_0a_0b_0c_[0,0,10]m.out',
                                       'RDR_[-500,500]cm_30m_25cm_0a_0b_0c_[0,0,10]m.out',
                                       'RDR_[330,330]cm_30m_25cm_0a_0b_0c_[0,0,10]m.out']]
    
    SkyFiles = [Dir+name for name in ['RDS_[500,-500]cm_30m_25cm_0a_0b_0c_[0,0,10]m.out',
                                      'RDS_[-500,500]cm_30m_25cm_0a_0b_0c_[0,0,10]m.out',
                                      'RDS_[330,330]cm_30m_25cm_0a_0b_0c_[0,0,10]m.out']]
    
    # Dir_name = load('time.joblib')

    # names = [filename[filename.find('/')+4:] for filename in glob.iglob('../RowData/{}/RDR*.out'.format(Dir_name), recursive=True)]
    
    # RealFiles = [Dir_name+'/'+'RDR'+name for name in names]
    # SkyFiles = [Dir_name+'/'+'RDS'+name for name in names]
    # ZPositions = [float(names[0].split('m_')[1]) for i in range(len(names))] # Accept ZPositions as argument for functions
    # Seperations = [float(names[i].split('cm')[1][-2:]) for i in range(len(names))]
    
    SkyCountList = []
    RealCountList = []
    CountList = []
    RowSkyList = []
    RowRealList = []
    IterateCountList = []
    
    for i in range(len(RealFiles)):
        RDSky = ReadRowDataFileFastest(SkyFiles[i], [XPositions[i],YPositions[i]])
        RDReal = ReadRowDataFileFastest(RealFiles[i], [XPositions[i],YPositions[i]])
    
        DCS = PazAnalysis(RDSky,Seperations[i],Iterate) 
        DCR = PazAnalysis(RDReal,Seperations[i],Iterate)
        
        PlotQuick(DCS,False)
        PlotQuick(DCR,False)
        
#        DCdatPlus = (DCR-DCS) 
        DCdatPlus = (DCS-DCR) 
        DCdatPlus[DCdatPlus < 0] = 0
        
#        return DCdatPlus
        
        PlotQuick(DCdatPlus,Iterate)
        
        RowSkyList.append(RDSky)
        RowRealList.append(RDReal)
        SkyCountList.append(DCS) 
        RealCountList.append(DCR)
        
        if Iterate == True:
            IterateCountList.append(DCdatPlus)    
            CountList.append(DCdatPlus[:,:,ClusterLayer])
            
        else:
            IterateCountList = None
            CountList.append(DCdatPlus)
        
    ReadDict = {'Row Sky List':RowSkyList, \
                'Row Real List':RowRealList, \
                'Sky Count List':SkyCountList, \
                'Real Count List':RealCountList, \
                'Subtracted Count List':CountList, \
                'Seperations':Seperations}
    
    return ReadDict
    


def AnalyseData(ReadDict):   
    
    HittingData = []
    ClusterList = []
    
    ValueList = []
    IndexList = []
    ObjectCounts = []
    ObjectMax = []
    #LayerDictList = []
    ClusteredPixels = []
    
    '''
    Shouldn't we just cluster object counts? 
    Ie. cluster after they have been interfered
    '''
    
    for i in range(len(ReadDict['Row Sky List'])):

        TempHittingData = []
        TempClusterList = []
        TempObjectCounts = []
        TempObjectMax = []
        
        
        Value, Index = LocalMaxIndices(ReadDict['Subtracted Count List'][i], LocalCutoff, Divide)

        ValueList.append(Value)
        IndexList.append(Index)
        
        Clustered_pixels = set()
        
#        print("Max values shape is: ",np.shape(Value))
#        print("Max values are: ",Value)
        # print("Max indices shape is: ",np.shape(Index))
        # print("Max indices are: ",Index)

        for j in range(len(Value)): # Clusters each of the extrema in the List of local extrema (Indices / Values)
            Max = np.max(Value)
            
#            print("Current array index:",i)
#            print("Current indices ",[Index[j][0],Index[j][1]])
#            print("clustered pixels are ",Clustered_pixels)
            
            if (Index[j][0],Index[j][1]) in Clustered_pixels:
                continue # this region has already been clustered
            
            ClusterDict = ClusterAlgorithm(ReadDict['Subtracted Count List'][i][:,:], PercentCutoff * Max, [Index[j][0],Index[j][1]])
            
            if np.count_nonzero(ClusterDict['Clustered Array']) > 5:
            # just did this
                for lists in ClusterDict['Active Indices']:
                    for indices in lists:
                        Clustered_pixels.add(tuple(indices))
                
                
#                print("Active indices shape is: ",np.shape(ClusterDict['Active Indices']))
#                print("Active indices are: ",ClusterDict['Active Indices'])
                
#                PlotQuick(ClusterDict['Clustered Array'],False)
        
#                if dense:
#                    PixelHits = ClusteredHittingPoints(ReadDict['Row Real List'][i], ClusterDict['Active Indices'], ClusterLayer, ReadDict['Seperations'][i])
                
#                else: 
                
#                print("Test")
                
#                PixelHits = ClusteredHittingPoints(ReadDict['Row Sky List'][i], ClusterDict['Active Indices'], ClusterLayer, ReadDict['Seperations'][i])
                
#                TempCounts = ObjectView(PixelHits, ProjectionPixel, ObjectZ, ImageVolume, ReadDict['Seperations'][i])
        
#                TempHittingData.append(PixelHits)       
#                TempObjectCounts.append(TempCounts)
#                TempObjectMax.append(np.max(TempCounts))
                TempClusterList.append(ClusterDict)
                
        ClusteredPixels.append(Clustered_pixels)
        
        HittingData.append(TempHittingData)
        ClusterList.append(TempClusterList)
        ObjectCounts.append(TempObjectCounts)
        ObjectMax.append(TempObjectMax)
        
        
        
#    ObjectCounts = np.array(ObjectCounts)
#    
#    ObjectCounts = ScaleLayers(ObjectCounts, True)
#    
#    ObjectGroups, OverlapLists = GroupOverlaps(ObjectCounts)          
#    
#    DetectorCounts, AddedMaxima, Targets = ScaleGroups(ObjectCounts, ObjectGroups, ObjectMax, OverlapCutoff) # Scales the array from each beam identically and  
#                                                                                                             # creates Target without need for the classifier                                                                                            
#                    
    AnalysisDict = {'Hitting Data':HittingData, \
                    'Cluster Dict List':ClusterList, \
#                    'Detector Counts':DetectorCounts, \
#                    'Targets':Targets, \
#                    'Object Counts':ObjectCounts, \
#                    'Object Groups':ObjectGroups, \
                    'Cluster Indices':IndexList, \
                    'Cluster Values':ValueList, \
                    'All Indices':ClusteredPixels}
    
    return AnalysisDict
    
    

def VisualiseObjects(AnalysisDict, ReadDict):

    #AllClusters = []
    #GroupClusters = []
    Shape = np.shape(AnalysisDict['Object Counts'])
    Maximized = np.zeros(Shape)
    IsolatedObjects = np.zeros((ProjectionPixel[0],ProjectionPixel[1],ProjectionPixel[2]))

    for i in range(len(AnalysisDict['Object Groups'])):
        #for i in AnalysisDict['Object Groups'][group]:
        Max = np.max(AnalysisDict['Detector Counts'][i])
        #MaxIndex = [np.where(ProgramDict['Decay Volume Slices'][1] == Max)[0][0]

        for j in range(ProjectionPixel[2]):
            Layer_Max = np.max(AnalysisDict['Detector Counts'][i,:,:,j])
            
#            Value, Index = LocalMaxIndices(AnalysisDict['Detector Counts'][i,:,:,j], LocalCutoff3D * Max / Layer_Max, Divide)  
#            Value, Index = LocalMaxIndices(AnalysisDict['Detector Counts'][i,:,:,j], Layer_Max * LocalCutoff3D, Divide)  
            Value, Index = LocalMaxIndices(AnalysisDict['Detector Counts'][i,:,:,j], Max * LocalCutoff3D, Divide)  


            Clustered_pixels = set()

            if len(Index) != 0:
                for k in range(len(Index)):

#                    LayerDict = ClusterAlgorithm(AnalysisDict['Detector Counts'][i,:,:,j], PercentCutoff3D * Max / Layer_Max, Index[k]) 
#                    LayerDict = ClusterAlgorithm(AnalysisDict['Detector Counts'][i,:,:,j], Layer_Max * PercentCutoff3D, Index[k]) 
                    LayerDict = ClusterAlgorithm(AnalysisDict['Detector Counts'][i,:,:,j], Max * PercentCutoff3D, Index[k]) 

                    if np.count_nonzero(LayerDict['Clustered Array']) > 5:
                        for lists in LayerDict['Active Indices']:
                            for indices in lists:
                                Clustered_pixels.add(tuple(indices))

#                ClusteredArray = np.zeros(np.shape(LayerDict['Clustered Array']))
                
                for ind_tup in Clustered_pixels:
                    Maximized[i,ind_tup[0],ind_tup[1],j] += \
                        AnalysisDict['Detector Counts'][i,ind_tup[0],ind_tup[1],j]

#                        if k == 0: # and np.count_nonzero(LayerDict['Clustered Array']) > 5: # only place it is being reset to zero ==> this is the issue seperate second condition
#                            Temp = [LayerDict['Clustered Array']]
#    
#                        else: #if np.count_nonzero(LayerDict['Clustered Array']) > 5:
#                            Temp = np.concatenate((Temp,[LayerDict['Clustered Array']]),axis=0)
#    
#                        Temp = np.array(Temp)
#    
#                    for l in range(ProjectionPixel[0]):
#                        for m in range(ProjectionPixel[1]):
#                            Maximized[i,l,m,j] += np.max(Temp[:,l,m]) 
                
#            else:
#                Temp = np.zeros((1,ProjectionPixel[0],ProjectionPixel[1]))
#            
        IsolatedObjects += Maximized[i]
            
            #GroupClusters.append(Temp)
        
        #IsolatedObjects += Maximized[i]
        #AllClusters.append(GroupClusters)
        
    #IsolatedObjects = np.zeros((ProjectionPixel[0],ProjectionPixel[1],ProjectionPixel[2]))
    '''
    for i in range(len(AnalysisDict['Object Groups'])):
        IsolatedObjects += Maximized[i]
    '''

    #GroupHitting = []
    
    
    for i in range(len(AnalysisDict['Object Groups'])):   
        TempHitting = []
        
        for j in AnalysisDict['Object Groups'][i]:
            for k in range(len(AnalysisDict['Hitting Data'][j])):
                TempHitting.append(AnalysisDict['Hitting Data'][j][k])
            
        AlterHittingPoints(TempHitting, True, 500, Which, [0,0])
    
    
    ScatterDistance(IsolatedObjects, Cutoff, ObjectZ, ImageVolume, ReadDict['Seperations'][0]) 
    
    
        #GroupHitting.append(TempHitting)
    
    ObjectDict = {'Isolated Objects':IsolatedObjects, \
                  'Maximized':Maximized}
                 
    return ObjectDict



def VisualiseObjects_old(AnalysisDict, ReadDict):

    #AllClusters = []
    #GroupClusters = []
    Shape = np.shape(AnalysisDict['Object Counts'])
    Maximized = np.zeros(Shape)
    IsolatedObjects = np.zeros((ProjectionPixel[0],ProjectionPixel[1],ProjectionPixel[2]))

    for i in range(len(AnalysisDict['Object Groups'])):
        #for i in AnalysisDict['Object Groups'][group]:
        Max = np.max(AnalysisDict['Detector Counts'][i])
        #MaxIndex = [np.where(ProgramDict['Decay Volume Slices'][1] == Max)[0][0]

        for j in range(ProjectionPixel[2]):
#            Value, Index = LocalMaxIndices(AnalysisDict['Detector Counts'][i,:,:,j], Max * LocalCutoff3D, Divide)  
            Value, Index = LocalMaxIndices(AnalysisDict['Detector Counts'][i,:,:,j], LocalCutoff3D, Divide)  

            if len(Index) != 0:
                for k in range(len(Index)):

#                    LayerDict = ClusterAlgorithm(AnalysisDict['Detector Counts'][i,:,:,j], Max * PercentCutoff3D, Index[k]) 
                    LayerDict = ClusterAlgorithm(AnalysisDict['Detector Counts'][i,:,:,j], PercentCutoff3D, Index[k]) 

                    if k == 0: # and np.count_nonzero(LayerDict['Clustered Array']) > 5: # only place it is being reset to zero ==> this is the issue seperate second condition
                        Temp = [LayerDict['Clustered Array']]

                    else: #if np.count_nonzero(LayerDict['Clustered Array']) > 5:
                        Temp = np.concatenate((Temp,[LayerDict['Clustered Array']]),axis=0)

                    Temp = np.array(Temp)

                for l in range(ProjectionPixel[0]):
                    for m in range(ProjectionPixel[1]):
                        Maximized[i,l,m,j] += np.max(Temp[:,l,m]) 
                
            else:
                Temp = np.zeros((1,ProjectionPixel[0],ProjectionPixel[1]))
            
            IsolatedObjects += Maximized[i]
            
            #GroupClusters.append(Temp)
        
        #IsolatedObjects += Maximized[i]
        #AllClusters.append(GroupClusters)
        
    #IsolatedObjects = np.zeros((ProjectionPixel[0],ProjectionPixel[1],ProjectionPixel[2]))
    '''
    for i in range(len(AnalysisDict['Object Groups'])):
        IsolatedObjects += Maximized[i]
    '''

    #GroupHitting = []
    
    for i in range(len(AnalysisDict['Object Groups'])):   
        TempHitting = []
        
        for j in AnalysisDict['Object Groups'][i]:
            for k in range(len(AnalysisDict['Hitting Data'][j])):
                TempHitting.append(AnalysisDict['Hitting Data'][j][k])
            
        AlterHittingPoints(TempHitting, True, 500, Which, [0,0])
    
    
    ScatterDistance(IsolatedObjects, Cutoff, ObjectZ, ImageVolume, ReadDict['Seperations'][0]) 
    
    
        #GroupHitting.append(TempHitting)
    
    ObjectDict = {'Isolated Objects':IsolatedObjects, \
                  'Maximized':Maximized}
                 
    return ObjectDict



#---------------------------------- Call Run Functions --------------------------------------



#-----------------------------------------------------------------
# Runs the imaging procedure on data specified by the user on
# the command line. Creates 2D images as well as tracked beams
# and estimates of the object sizes and shapes visually
#-----------------------------------------------------------------

if __name__ == "__main__":

    ReadDictionary = ReadDataFiles()
    
    AnalysisDictionary = AnalyseData(ReadDictionary)
    
    ObjectDictionary = VisualiseObjects(AnalysisDictionary, ReadDictionary) 
    
#    imageio.mimsave('oc.gif', AnalysisDictionary['Detector Counts'][0].transpose())

    tictoc = datetime.datetime.now() - begin_time
    print('The total runtime of the program was: ', tictoc)
