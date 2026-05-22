# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:03:18 2026

@author: ON
"""

import os
import keras
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import tensorflow.keras.backend as K

np.set_printoptions(suppress=True)
np.set_printoptions(edgeitems=400, linewidth=1000)
np.set_printoptions(precision=4)

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

plt.rcParams['figure.figsize'] = (40, 20)
plt.rcParams.update({'font.size': 33})
#plt.style.context('dark_background')



class worldHypths:
    def __init__(self):
    
        self.__inputModel       = 'models/'
        self.__inputEvents      = 'database/events_data/'
        
        self.__datasetTest      = "database/evaluated/OMNI_20240101_20260331.csv"
        self.__datasetBaseline  = "database/evaluated/Colorado_20240101_20240331.csv"
        
        self.__treino_dim       = 19
        self.__target           = 19
        self.__dim              = 48
        self.__b_size           = 24
        
        self.__coef_mean        = [183.129893,11.500000,5.702753,0.023190,0.001778,
                                     -0.000807,-0.037346,426.414533,0.074147,-3.846173,
                                     6.277123,91932.280150,2.082737,0.016914,208.202730,
                                     -0.209666,1.576934,13.537718,0.132061,-11.817895]
        
        self.__coef_std         = [105.441549,6.922201,2.994710,3.369023,3.823655,
                                    2.817255,2.955366,98.941935,22.160388,19.264675,
                                    4.890508,89733.140224,1.606501,1.365420,59.579773,
                                    52.455670,14.288044,1.540540,0.589907,19.350337]
        
        self.__columns          = ["timeStamp","Year","Day","Hour","B",
                                    "BX_GSE_GSM","BY_GSE","BZ_GSE","BZ_GSM",
                                    "Speed","Vy","Vz","Proton_Density",
                                    "Proton_Temperature","Flow_Pressure",
                                    "Electric_Field","s_c_X_GSE","s_c_Y_GSE",
                                    "s_c_Z_GSE","BSN_location_X_GSE",
                                    "BSN_location_Z_GSE","Dst"]
        
        self.__columnsBaseline  = ["obsTime", 
                                   "DSTKyoto.dst", 
                                   "DSTKyoto_MagCNN.dst"]
        
    def get_model(self):
        return self.__inputModel
        
    def get_data(self):
        return self.__datasetTest, self.__datasetBaseline, self.__inputEvents
    
    def get_parameters(self):
        return self.__treino_dim, self.__target, self.__dim, self.__b_size
        
    def get_coefs(self):
        return self.__coef_mean, self.__coef_std
    
    def get_columns(self):
        return self.__columns, self.__columnsBaseline





class Dst:
    def __init__(self, inputModel, inputEvents):
        self.model  = inputModel
        self.events = inputEvents
    
    def description(self):
        return f"{self.model} - {self.events}"

# a classe model herda da classe Dst.
class Model(Dst):
    def __init__(self, model, events, color):
        super().__init__(model, events)
        self.color = color

    def description(self):
        return f"{self.model} is a {self.color}."
    
    

#------
@keras.saving.register_keras_serializable(package="MyCustomLayers")
class attention(tf.keras.layers.Layer):

    def __init__(self, return_sequences=True, **kwargs):
        self.return_sequences = return_sequences
        #super(attention,self).__init__()
        super(attention, self).__init__(**kwargs) # Crucial: Pass kwargs here

    def build(self, input_shape):

        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1), initializer="normal")

        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1), initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1), initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1), initializer="normal")

        super(attention,self).build(input_shape)



    def call(self, x):

        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)

        output = x*a

        if self.return_sequences:
            return output
        return K.sum(output, axis=1)
#------



#------
def create_dir_overwrite(path):
    # Check if the directory exists
    if os.path.exists(path):
        # If it exists, remove it and all its contents
        shutil.rmtree(path)
        print(f"Existing directory '{path}' removed.")
    
    # Create the new directory (and any necessary parent directories)
    os.makedirs(path)
    print(f"Directory '{path}' created.")
#------

#------
def create_sequence(dataset, length):
    data_sequences = []
    for index in range(len(dataset) - length):
        data_sequences.append(dataset[index: index + length])
    return np.asarray(data_sequences)
#------

#------
def attributeStatistics(attribute):
    print("     Shape:                  ", attribute.size)
    print("     Minimum:                ", '%.4f' % np.nanmin(attribute))
    print("     Maximum:                ", '%.4f' % np.nanmax(attribute))
    print("     Mean:                   ", '%.4f' % np.nanmean(attribute))
    print("     Median:                 ", '%.4f' % np.nanmedian(attribute))
    print("     Variance:               ", '%.4f' % np.nanvar(attribute))
    print("     Standard Deviation:     ", '%.4f' % np.nanstd(attribute))
#------

#------
def reverse_zscore(pandas_series, mean, std):
    yis = pandas_series*std+mean
    return yis
#------

#------
def plotPrediction(output, startDate, endDate, Dst_ON, colorado, kioto):

    plt.margins(x=0)
    plt.title(str(startDate)+"  -  "+str(endDate)+"\n")
    plt.xlabel('Hours', fontsize=43, labelpad=20)
    plt.ylabel('nT', fontsize=43, rotation=0, labelpad=25)
    plt.plot(Dst_ON, color="blue", linewidth="3", label="Dst_ON")
    plt.plot(colorado, color="green", linewidth="3", label="Dst_CNN")
    plt.plot(kioto, color="orange", linewidth="3", label="Dst_Kioto")
    plt.legend()

    plt.minorticks_on() 
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#CCCCCC', linestyle=':', alpha=0.5)
    
    plt.savefig(output+'Figure_'+str(startDate[:8])+'.png',
                bbox_inches='tight', 
                dpi=300)
    plt.show()
    #------
    

    #------
    fig, (ax1, ax2, ax3, ax0) = plt.subplots(4, 1, #sharex=True, 
                                             figsize=(30, 30), 
                                             constrained_layout=True)
    
    ax0.margins(x=0)
    ax0.set_title(str(startDate)+"  -  "+str(endDate)+"\n")
    ax0.set_xlabel('Hours', fontsize=43, labelpad=20)
    ax0.set_ylabel('nT', fontsize=43, rotation=0, labelpad=25)
    ax0.plot(Dst_ON, color="blue", linewidth="3", label="Dst_ON")
    ax0.plot(colorado, color="green", linewidth="3", label="Dst_CNN")
    ax0.plot(kioto, color="orange", linewidth="3", label="Dst_Kioto")
    ax0.legend()
    #ax0.grid()
    
    # Required to see minor grid
    ax0.minorticks_on() 
    ax0.grid(which='major', color='#666666', linestyle='-')
    ax0.grid(which='minor', color='#CCCCCC', linestyle=':', alpha=0.5)

    # Total integral using Simpson's rule
    total_integral_simpson = simpson(Dst_ON, kioto)
    total_integral_simpson_formx = np.abs(total_integral_simpson)
    total_integral_simpson_formy = f"{total_integral_simpson_formx:.2f}"
    #print(f"Total integral (Simpson's rule): {total_integral_simpson_formy}")
    labelPred = "Cumulative integral: "+ str(total_integral_simpson_formy)
    
    print()
    print('* * * * * * *')
    print('Cumulative integral using Simpson s rule:')
    print('* * * * * * *')
    print()
    
    print('Area of ​​difference between the [Dst_ON model] in relation to [Dst_Kioto model]')
    print(labelPred)
    print()

    ax1.margins(x=0)
    ax1.set_title("Area of ​​difference between the Dst_ON model in relation to Dst_Kioto model")
    ax1.fill_between(Dst_ON, kioto, label=labelPred, color="#5ECFFF")
    ax1.legend()

    ax1.minorticks_on() 
    ax1.grid(which='major', color='#666666', linestyle='-')
    ax1.grid(which='minor', color='#CCCCCC', linestyle=':', alpha=0.5)
    
    ax1.tick_params(left=True,
                        right=True,
                        labelleft=True,
                        labelbottom=False,
                        bottom=False)

    # Total integral using Simpson's rule
    total_integral_simpson2 = simpson(colorado, kioto)
    total_integral_simpson_form2x = np.abs(total_integral_simpson2)
    total_integral_simpson_form2y = f"{total_integral_simpson_form2x:.2f}"
    labelPred2 = "Cumulative integral: "+ str(total_integral_simpson_form2y)
    
    print('Area of ​​difference between the [Dst_CNN model] in relation to [Dst_Kioto model]')
    print(labelPred2)
    print()

    ax2.margins(x=0)
    ax2.set_title("Area of ​​difference between the Dst_CNN model in relation to Dst_Kioto model")
    ax2.fill_between(colorado, kioto, label=labelPred2, color="#9EFF5E")
    #ax2.set_ylabel('nT', fontsize=33, rotation=0, labelpad=25)
    ax2.legend()
    
    ax2.minorticks_on() 
    ax2.grid(which='major', color='#666666', linestyle='-')
    ax2.grid(which='minor', color='#CCCCCC', linestyle=':', alpha=0.5)
    
    ax2.tick_params(left=True,
                        right=True,
                        labelleft=True,
                        labelbottom=False,
                        bottom=False)

    
    # Total integral using Simpson's rule
    total_integral_simpson3 = simpson(Dst_ON, colorado)
    total_integral_simpson_form3x = np.abs(total_integral_simpson3)
    total_integral_simpson_form3y = f"{total_integral_simpson_form3x:.2f}"
    labelPred3 = "Cumulative integral: "+ str(total_integral_simpson_form3y)
   
    print('Area of ​​difference between the [Dst_ON model] in relation to [Dst_CNN model]')
    print(labelPred3)
    print()
    print('* * * * * * *')
    print()
    
    

    ax3.margins(x=0)
    ax3.set_title("Area of ​​difference between the Dst_ON model in relation to Dst_CNN model")
    ax3.fill_between(colorado, Dst_ON, label=labelPred3, color="#FFD75E")
    ax3.legend()
    
    ax3.minorticks_on() 
    ax3.grid(which='major', color='#666666', linestyle='-')
    ax3.grid(which='minor', color='#CCCCCC', linestyle=':', alpha=0.5)
    
    ax3.tick_params(left=True,
                        right=True,
                        labelleft=True,
                        labelbottom=False,
                        bottom=False)
    
    fig.tight_layout()
    
    plt.savefig(output+'Figure_painel_'+str(startDate[:8])+'.png',
                bbox_inches='tight', 
                dpi=150)
    
    plt.show()
#------

#------
def plotPredictionUncertainty(output,start_date, end_date, y,
                              ypred, mean_pred, uncertainty_std):

    plt.figure()
    

    plt.minorticks_on() 
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#CCCCCC', linestyle=':', alpha=0.5)

    plt.plot(y, mean_pred, color='#CCCCCC', linewidth=6, 
                     label='Mean Prediction')
    
    plt.plot(ypred, mean_pred, color='#000000', linewidth=6, 
                     label='Mean Prediction')
    
    plt.fill_between(y.flatten(),
                     (mean_pred - 3 * uncertainty_std).flatten(),
                     (mean_pred + 3 * uncertainty_std).flatten(),
                     color='red', alpha=0.3, 
                     label='3 Std Dev Uncertainty') 
                     
    plt.xlabel('Input', fontsize=33)
    plt.ylabel('Output', fontsize=33)
    plt.title('\nMC Dropout Uncertainty Distribution\n'+str(start_date)+'\n', fontsize=33)
    plt.legend()
    
    plt.savefig(output+str(start_date[:10])+' Std Dev Uncertainty phase.png',
                bbox_inches='tight', 
                dpi=150)
    plt.show()
    #------
    

    #------
    # Upper and lower bounds
    lower_bound = ypred.flatten() - (mean_pred - 3 * uncertainty_std).flatten()
    upper_bound = ypred.flatten() + (mean_pred + 3 * uncertainty_std).flatten()
    #------
    
    #------
    # Plotting
    plt.figure()
    
    plt.margins(x=0)
    timestepX = np.arange(0, len(y), 1)  
    
    plt.minorticks_on() 
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#CCCCCC', linestyle=':', alpha=0.5)
    
    plt.plot(timestepX, ypred, label='Dst_ON',
             color='blue', linewidth=4)
    
    plt.fill_between(timestepX, lower_bound, upper_bound,
                     color='red', alpha=0.3, 
                     label='3 Std Dev Uncertainty Band')
    #plt.title('\nPrediction with Uncertainty Band\n'+str(start_date)+'\n')
    plt.title('\n'+str(start_date)+'-'+str(end_date)+'\n')
    plt.xlabel('Hours', fontsize=43, labelpad=20)
    plt.ylabel('nT', fontsize=43, rotation=0, labelpad=25)
    
    plt.legend()
    
    plt.savefig(output+str(start_date[:10])+" Pred with uncertainty band.png",
                bbox_inches='tight', 
                dpi=150)
    
    plt.show()
#------