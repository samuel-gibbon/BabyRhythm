#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 08:14:08 2021

@author: mahmoudkeshavarzi
"""


import tensorflow as tf	
import os
import numpy as np
import scipy
import scipy.io

np.random.seed(7)


from sklearn import svm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from scipy.signal import butter, lfilter






def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def filt_data(Data_input, lowcut, highcut, fs, order=6):
    Out_Data = Data_input
    Out_Power = np.zeros((Data_input.shape[0],60))
    for ii in range(0,Data_input.shape[0]):
        for j in range (0,Data_input.shape[1]):
            Out_Data[ii,j,:] = butter_bandpass_filter(np.transpose(Data_input[ii,j,:]), lowcut, highcut, fs, order=6)
            Out_Power[ii,j] = np.sum(Out_Data[ii,j,:]*Out_Data[ii,j,:])/200
    return Out_Data, Out_Power
            
            
    
N_Com = 30
Results=[]


q1list = os.listdir("/home/sam/Documents/epoched_v2/bad_epochs_rejected/drum")
    
for i in range(len(q1list)):
    
    filename = q1list[i]
    print("File index = ",i)
    
    q1Data = []
    q4Data = []
    q1Label = []
    q4Label = []
    
    data = scipy.io.loadmat("/home/sam/Documents/epoched_v2/bad_epochs_rejected/drum/"+filename)
    data = data['all_data']
    data = np.swapaxes(data,0,2) 
    data = np.swapaxes(data,1,2)
    
    Out_Data_Delta,Out_Power_Delta = filt_data(data, 1, 4, 100, order=6)
    Out_Data_Theta,Out_Power_Theta = filt_data(data, 4, 8, 100, order=6)
    Out_Data_Alpha,Out_Power_Alpha = filt_data(data, 8, 12, 100, order=6)  
    
    data=np.concatenate((Out_Power_Delta,Out_Power_Theta,Out_Power_Alpha),axis=1)

    
    label = [0 for i in range(data.shape[0])] 
    q1Label += label
    if len(q1Data) == 0:
        q1Data = data
    else:
        q1Data = np.concatenate((q1Data, data))                        
    
    filename = filename.replace("_Drum","_Ta")
    data = scipy.io.loadmat("/home/sam/Documents/epoched_v2/bad_epochs_rejected/ta/"+filename)
    data = data['all_data']
    data = np.swapaxes(data,0,2) 
    data = np.swapaxes(data,1,2)
    
    Out_Data_Delta,Out_Power_Delta = filt_data(data, 1, 4, 100, order=6)
    Out_Data_Theta,Out_Power_Theta = filt_data(data, 4, 8, 100, order=6)
    Out_Data_Alpha,Out_Power_Alpha = filt_data(data, 8, 12, 100, order=6)  
    
    data=np.concatenate((Out_Power_Delta,Out_Power_Theta,Out_Power_Alpha),axis=1)
    
    label = [1 for i in range(data.shape[0])] 
    q4Label += label
    if len(q4Data) == 0:
        q4Data = data
    else:
        q4Data = np.concatenate((q4Data, data))    
   
        
    All_Data = np.concatenate((q1Data,q4Data),axis=0)
    All_Label = np.concatenate((q1Label,q4Label),axis=0)

    All_Data, X_test, All_Label, y_test = train_test_split(All_Data, All_Label, test_size=0.00001,random_state=42)
    
    All_Data = np.concatenate((All_Data,X_test),axis=0)
    All_Label = np.concatenate((All_Label,y_test),axis=0)
    
    

    scaler = StandardScaler()
    All_Data= scaler.fit_transform(All_Data)                
    pca = PCA(n_components=N_Com,random_state = 42)
    pca.fit(All_Data)
    All_Data = pca.transform(All_Data)
    
 

    cv = StratifiedKFold(n_splits=5) 
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=42,))

    Roc_auc =np.zeros((1,5))
    for j, (train, test) in enumerate(cv.split(All_Data, All_Label)):
        Label_tr=tf.keras.utils.to_categorical(All_Label[train], num_classes=2)
        y_score = classifier.fit(All_Data[train], Label_tr).decision_function(All_Data[test])
        Label_te = tf.keras.utils.to_categorical(All_Label[test], num_classes=2)
        Roc_auc[0,j] = roc_auc_score(Label_te, y_score)
    Results.append(Roc_auc)  
    print(Roc_auc)    
    print(np.mean(Roc_auc))
