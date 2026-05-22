# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:12:26 2026

@author: ON
"""

import os
import sys
import keras
import sklearn
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from library import attention
from library import worldHypths
from library import reverse_zscore
from library import plotPrediction
from library import create_sequence
from platform import python_version
from library import attributeStatistics
from library import create_dir_overwrite
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

np.random.seed(1230)

# How to run: python script.py [model_version] {2022, 2024}
# Model version 2025 will be coming soon.

print()
print() 
print("--------------------------------------------------------")
print("National Observatory / MCTI")
print("Brazil")
print("--------------------------------------------------------")
print("--------------------------------------------------------")
print("Python version   ==      %s" % python_version())
print("Pandas           ==      %s" % pd.__version__)
print("Numpy            ==      %s" % np.__version__)
print("Sklearn          ==      %s" % sklearn.__version__)
print("Matplotlib       ==      %s" % matplotlib.__version__)
print("Tensorflow       ==      %s" % tf.__version__)
print("--------------------------------------------------------")
print()

#- - - - - -
model_instance = worldHypths()
#- - - - - -
model = model_instance.get_model() 
datasetTest, datasetBaseline, inputEvents = model_instance.get_data() 
coef_1, coef_2 = model_instance.get_coefs() 
treino_dim, target, dim, b_size = model_instance.get_parameters()
columns, columnsBaseline = model_instance.get_columns()
#- - - - - -

#- - - - - -
print(f"{model}\n{inputEvents}")
print('*.*.*')
print()

with os.scandir(model) as entries:
    for entry in entries:
        if entry.is_file():
            print()
            print()
            print('***')
            print(entry.name)
            print('***')
            print()
            print()

            print()
            print("Load Model:")
            model = keras.models.load_model(model+entry.name, 
                                                   custom_objects={'attention': attention})
            print()
            print(model.summary())
            print()

            output = 'output/'
            output_current = output+str(entry.name[:-6])+"/"
            create_dir_overwrite(output_current)
            output = output_current

            with os.scandir(inputEvents) as entries2:
                for entry2 in entries2:
                    if entry2.is_file():
                        print()
                        print()
                        print('***')
                        print(entry2.name)
                        print('***')
                        print()
                        print()

                        #.......
                        # Load data:
                        #.......
                        df          = pd.read_csv(inputEvents+entry2.name, encoding='utf-8', usecols=columns)
                        df_test     = pd.read_csv(datasetTest, encoding='utf-8', usecols=columns[1:len(columns)])
                        df_baseline = pd.read_csv(datasetBaseline, encoding='utf-8', usecols=columnsBaseline)
                        #.......

                        #.......
                        #  TimeStamp:
                        #.......
                        df_test.reset_index(drop=True, inplace=True)
                        df_test["Timestamp"]    = pd.to_datetime(df_test["Year"] * 100000 + df_test["Day"] * 100 + df_test["Hour"], format="%Y%j%H")
                        df_test.index           = df_test["Timestamp"] 

                        df_baseline.reset_index(drop=True, inplace=True)
                        df_baseline["Timestamp"]  = pd.to_datetime(df_baseline["obsTime"])
                        df_baseline.index         = df_baseline["Timestamp"]
                        #.......
                        
                        #.......
                        start_time = df.iloc[0,0]
                        start_date_plus_dim = pd.Timestamp(start_time) - pd.Timedelta(days=2)
                        end_time   = df.iloc[-1,0]
                        #.......

                        #.......
                        select_df_test      = df_test.loc[start_date_plus_dim:end_time]
                        select_df_baseline  = df_baseline.loc[start_time:end_time]
                        
                        select_df_test_ = select_df_test.drop('Timestamp', axis=1)
                        select_df_test_.drop(select_df_test_.columns[[0]], axis=1, inplace=True)
                        #.......
                        
                        #.......
                        select_df_test_Norm = (select_df_test_-coef_1)/coef_2
                        #.......
                       
                        #.......
                        X_test = select_df_test_Norm.iloc[:, 0:treino_dim]
                        y_test = select_df_test_Norm.iloc[:, target]
                        
                        
                        X_test = create_sequence(X_test, dim)
                        y_test = y_test[-X_test.shape[0]:]
                        #.......
                        
                        #.......
                        # Prediction
                        #.......
                        print('* * * * * *')
                        print("Prediction:")
                        print('* * * * * *')
                        prediction = model.predict(X_test)
                        
                        #. . . . .
                        print()
                        print('****')
                        print('Statistics {KIOTO} [normalized]:')
                        attributeStatistics(y_test)
                        print()
                        print('Statistics {Prediction} [normalized]:')
                        attributeStatistics(prediction)
                        print('****')
                        print()
                        #. . . . .
                        
                        #. . . . .
                        rsme = mean_squared_error(prediction, y_test)
                        rsme_formatted = f"{rsme:.4f}"
                        print('****')
                        print("[Prediction] RMSE [normaliz]:        ", rsme_formatted)
                        print()
                        
                        mae = mean_absolute_error(prediction, y_test)
                        mae_formatted = f"{mae:.4f}"
                        print("[Prediction] MAE [normaliz]:         ", mae_formatted)
                        print()
                        
                        y_test_nT = reverse_zscore(y_test,
                                                coef_1[-1],
                                                coef_2[-1])
                        
                        predictiont_nT = reverse_zscore(prediction,
                                                coef_1[-1],
                                                coef_2[-1])
                        
                        #. . . . .
                        print()
                        print('***')
                        print("Statistics {Dst_ON Prediction} [nT]:")
                        rsme = mean_squared_error(predictiont_nT, y_test_nT)
                        rsme_formatted = f"{rsme:.4f}"
                        print("[Prediction] RMSE [nT]:              ", rsme_formatted)
                        print()
                        
                        mae = mean_absolute_error(predictiont_nT, y_test_nT)
                        mae_formatted = f"{mae:.4f}"
                        print("[Prediction] MAE [nT]:               ", mae_formatted)
                        print('***')
                        print()
                        #. . . . .
                        
                        #. . . . .
                        print()
                        print('***')
                        print('Statistics {Dst_CNN} [nT]:')
                        rsme = mean_squared_error(select_df_baseline["DSTKyoto_MagCNN.dst"], y_test_nT)
                        rsme_formatted = f"{rsme:.4f}"
                        print("[Prediction] RMSE [nT]:              ", rsme_formatted)
                        print()
                        
                        mae = mean_absolute_error(select_df_baseline["DSTKyoto_MagCNN.dst"], y_test_nT)
                        mae_formatted = f"{mae:.4f}"
                        print("[Prediction] MAE [nT]:               ", mae_formatted)
                        print('***')
                        print()
                        #. . . . .
                        
                        #. . . . .
                        print()
                        print('****')
                        print('Statistics {CNN} [nT]:')
                        attributeStatistics(select_df_baseline["DSTKyoto_MagCNN.dst"])
                        print()
                        print('Statistics {KIOTO} [nT]:')
                        print('****')
                        attributeStatistics(y_test_nT)
                        print()
                        print('Statistics {Prediction} [nT]:')
                        attributeStatistics(predictiont_nT)
                        print('****')
                        print()
                        #. . . . .
                        
                        #. . . . .
                        rsme2 = np.nanmin(mean_squared_error(predictiont_nT, y_test_nT))
                        rsme_formatted2 = f"{rsme2:.4f}"
                        print('****')
                        print("Dst_ON x Dst_Kyoto [nT]:             ", rsme_formatted2)
                        print()
                        
                        rsme3 = np.nanmin(mean_squared_error(predictiont_nT, select_df_baseline["DSTKyoto_MagCNN.dst"]))
                        rsme_formatted3 = f"{rsme3:.4f}"
                        print("Dst_ON x Dst_CNN [nT]:               ", rsme_formatted3)
                        print()
                        
                        rsme4 = np.nanmin(mean_squared_error(y_test_nT, select_df_baseline["DSTKyoto_MagCNN.dst"]))
                        rsme_formatted4 = f"{rsme4:.4f}"
                        print("Dst_Kyoto x Dst_CNN [nT]:            ", rsme_formatted4)
                        print('****')
                        print()
                        #. . . . .
                        
                        #.......
                        predictiont_nT = pd.DataFrame(predictiont_nT)
                        predictiont_nT.reset_index(drop=True, inplace=True)
                        
                        y_test_nT = pd.DataFrame(y_test_nT)
                        y_test_nT.reset_index(drop=True, inplace=True)
                        
                        select_df_baseline.reset_index(drop=True, inplace=True)
                        
                        predictiont_nT  = predictiont_nT.iloc[:,0]
                        y_test_nT       = y_test_nT.iloc[:,0]
                        y_test_CNN_nT   = select_df_baseline["DSTKyoto_MagCNN.dst"]
                        #.......
                        
                        #.......
                        # Corr
                        #.......
                        dfCorr = pd.DataFrame()
                        dfCorr = pd.concat([predictiont_nT, 
                                            y_test_nT, 
                                            y_test_CNN_nT], 
                                           axis=1)
                        co_dfCorr = dfCorr.corr(numeric_only=True)
                        print('****')
                        print(co_dfCorr)
                        print('****')
                        print()
                        #.......
                        
                        #.......
                        # Figures
                        #.......
                        plotPrediction(output,
                                       start_time, end_time,
                                       predictiont_nT,
                                       y_test_CNN_nT,
                                       y_test_nT)
                        #.......

                        del(df_test)
                        del(df_baseline)
                        del(df)
print('*.*.*')
print('Done.')