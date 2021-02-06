"""
This script will load the data, run the CNN, and save the results. 
You'll first need to put the data into seperate folders, and rename filepath lines accordingly.

"""

import os
import numpy as np
import scipy
from scipy import io
import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import LeakyReLU
from keras.optimizers import SGD
from keras.layers import BatchNormalization
from keras.models import load_model
from keras import models
from keras import layers
from keras import regularizers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.layers import InputLayer
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
import csv

results = []
seed = 42

cw = []

drumList = os.listdir('/home/sam/Documents/epoched_v2/bad_epochs_included/drum')
drumList.sort()
drumData = []
drumLabel = []

for DrumFilename in drumList: 
    data1 = scipy.io.loadmat('/home/sam/Documents/epoched_v2/bad_epochs_included/drum/'+DrumFilename)
    data1 = data1['all_data']
    data1 = np.swapaxes(data1,0,2)
    data1 = np.swapaxes(data1,1,2)
    label1 = [0 for i in range(data1.shape[0])]
    drumData.append(data1)
    drumLabel.append(label1)
    
taList = os.listdir('/home/sam/Documents/epoched_v2/bad_epochs_included/ta')  
taList.sort()
taData = []
taLabel = []
 
for TaFilename in taList:
    data2 = scipy.io.loadmat('/home/sam/Documents/epoched_v2/bad_epochs_included/ta/'+TaFilename)
    data2 = data2['all_data']
    data2 = np.swapaxes(data2,0,2)
    data2 = np.swapaxes(data2,1,2)
    label2 = [1 for i in range(data2.shape[0])]
    taData.append(data2)
    taLabel.append(label2)

EEG = []
for i in range(len(taData)):
    data = np.concatenate([taData[i], drumData[i]])
    data = tf.keras.utils.normalize(data,axis=-1,order=2)
    # reshape data to be compliant with what the model expects 
    data = np.expand_dims(data, axis=3)
    EEG.append(data)

LABEL = []
for i in range(len(taData)):
    data = np.concatenate([taLabel[i], drumLabel[i]])
    LABEL.append(data)

del(drumData,taData)

#from keras.callbacks import EarlyStopping
#es = EarlyStopping(monitor='val_auc', mode='max', patience=6)

#epoch_count = []
#epochs = []

for i in range(len(EEG)):
    cvscores = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    EEG1 = EEG[i]
    LABEL1 = LABEL[i]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(LABEL1), LABEL1)
    cw.append(class_weights)
    for train, test in kfold.split(EEG1, LABEL1):
        model = models.Sequential()
        model.add(Conv2D(12,(1,4),input_shape=(60,200,1)))
        model.add(Conv2D(12,(60,4),activation='elu',kernel_regularizer=regularizers.l2(0.01)))
        model.add(MaxPooling2D((1,4)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(32,activation='elu'))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', 
                      metrics=[keras.metrics.AUC(),#name='auc',dtype='float32',num_thresholds=3,thresholds=[0 - 1e-7, 0.5, 1 + 1e-7]),
                               keras.metrics.Precision(name='precision'),
                               keras.metrics.Recall(name='recall')])
        #LABEL1 = tf.keras.utils.to_categorical(LABEL1, num_classes=2)
        history = model.fit(EEG1[train], LABEL1[train], 
                  epochs=25, 
                  batch_size=32, 
                  class_weight=class_weights) 
                  #callbacks=[es],
                  #validation_data=(EEG1[test], LABEL1[test]))
        #n_epochs = len(history.history['loss'])
        #epoch_count.append(n_epochs)
        scores = model.evaluate(EEG1[test], LABEL1[test])
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1])
        #LABEL1 = np.argmax(LABEL1, axis=1)
    results.append((np.mean(cvscores), np.std(cvscores)))
    #epochs.append((np.mean(epoch_count), np.std(epoch_count)))
    tf.keras.backend.clear_session()
 
model.summary()

with open('classification_results_allepochs.csv','w+') as out:
    csv_out=csv.writer(out)
    for row in results:
        csv_out.writerow(row)

with open("classification_subjects_allepochs.txt", "w+") as subjects:
    for item in drumList:
        subjects.write("{}\n".format(item))
        
with open('classification_cw_allepochs.csv','w+') as out:
    csv_out=csv.writer(out)
    for row in cw:
        csv_out.writerow(row)

######### code below displays number of epochs for each stim

# drum epochs
epochs = []
for i in range(len(drumData)):
    epochs.append(len(drumData[i]))
sum = 0
for num in epochs:
    sum = sum+num
print(sum)

# ta epochs
epochs = []
for i in range(len(taData)):
    epochs.append(len(taData[i]))
sum = 0
for num in epochs:
    sum = sum+num
print(sum)
