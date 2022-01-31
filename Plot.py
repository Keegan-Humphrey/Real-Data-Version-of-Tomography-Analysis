import matplotlib.pyplot as plt
from joblib import load
import numpy as np
import glob
import os
from scipy import ndimage

def PlotQuick(Data, Save=False, Title=None):   
    #Expects Boolean value for ThreeD
    
    #--------------------------------------------------------------------
    # Plots each image layer in Data as a list of colormapped images
    #--------------------------------------------------------------------    
    
    def PlotQuick2D(Data):
        
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(1,1,1)
        
        ax.set_title(Title)
        plt.imshow(Data, alpha = 0.5)
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.patch.set_alpha(0)
        ax.set_frame_on(False)
        plt.colorbar(orientation='vertical')
        
        if Save:
            plt.savefig(Title.replace('.joblib','.png'))
            
        plt.show()
        
    
    try:
        Shape = np.shape(Data)
        
        for i in range(Shape[2]):
            PlotQuick2D(Data[:,:,i])
    
    except:
        PlotQuick2D(Data)


if __name__ == '__main__':
    save = True
    
    Dir = load('time.joblib') #'Sat_Mar_13_09-50-39_2021/error = 0 mev'
    #Dir = 'Fri_Mar_12_22-36-06_2021'
    #Dir = 'Sun_Feb_14_20-16-48_2021'
    
    Directory = os.getcwd() + '/' + Dir + '/' #load('time.joblib')
    
    names = [filename[filename.find('RDR')+3:] for filename in glob.iglob(Directory+'RDR*.joblib', recursive=True)]
    
    
    for i in range(len(names)):
        
        for k in range(1):
            sky = np.array(load(Directory + 'RDR' + names[i]), dtype=int)
            real = np.array(load(Directory + 'RDS' + names[i]), dtype=int)
            
            sky = sky / np.max(sky)
            real = real / np.max(real)
            
            if k == 1:
                sky = ndimage.gaussian_filter(sky,1)
                real = ndimage.gaussian_filter(real,1)
            
#            DCdatPlus = (real-sky)
            DCdatPlus = (sky-real)
            DCdatPlus[DCdatPlus < 0] = 0
            
            for j in range(1): #range(np.shape(sky)[2]):
                PlotQuick(sky[:,:,j], Title='RDS'+names[i],Save=save)
                PlotQuick(real[:,:,j], Title='RDR'+names[i],Save=save)
                PlotQuick(DCdatPlus[:,:,j], Title='DCR'+names[i],Save=save)
    
#            if k == 1:
#                print('denoised')
#                
#            else:
#                print('noisy')
