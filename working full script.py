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

results = []
seed = 7

drumList = os.listdir('D:\\DrumTa\\dataEpoched_2\\dataEpoched\\Drum')
drumData = []
drumLabel = []

for DrumFilename in drumList: 
    data1 = scipy.io.loadmat('D:\\DrumTa\\dataEpoched_2\\dataEpoched\\Drum\\'+DrumFilename)
    data1 = data1['all_data']
    data1 = np.swapaxes(data1,0,2)
    data1 = np.swapaxes(data1,1,2)
    label1 = [0 for i in range(data1.shape[0])]
    drumData.append(data1)
    drumLabel.append(label1)
    
taList = os.listdir('D:\\DrumTa\\dataEpoched_2\\dataEpoched\\Ta')  
taData = []
taLabel = []
 
for TaFilename in taList:
    data2 = scipy.io.loadmat('D:\\DrumTa\\dataEpoched_2\\dataEpoched\\Ta\\'+TaFilename)
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

import time
t = time.time()
for i in range(len(EEG)):
    cvscores = []
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    EEG1 = EEG[i]
    LABEL1 = LABEL[i]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(LABEL1), LABEL1)
    for train, test in kfold.split(EEG1, LABEL1):
        model = models.Sequential()
        model.add(Conv2D(10,(1,5),input_shape=(64,200,1)))
        model.add(Conv2D(10,(64,5),activation='elu',kernel_regularizer=regularizers.l2(0.01)))
        model.add(MaxPooling2D((1,3)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(32,activation='elu'))
        model.add(Dense(2,activation='softmax'))
        keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.9)
        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        LABEL1 = tf.keras.utils.to_categorical(LABEL1, num_classes=2)
        model.fit(EEG1[train], LABEL1[train], epochs=100, batch_size=25, class_weight=class_weights)
        scores = model.evaluate(EEG1[test], LABEL1[test])
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        LABEL1 = np.argmax(LABEL1, axis=1)
    results.append((np.mean(cvscores), np.std(cvscores)))
elapsed = time.time() - t  

# save results     
with open("results.txt", 'w+') as file_handler:
    for item in results:
        file_handler.write("{}\n".format(item))

# save model
model.save("cv_model1.h5")

# save subject list
with open("subjects.txt", "w+") as subjects:
    for item in drumList:
        subjects.write("{}\n".format(item))

with open('ur file.csv','wb') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['name','num'])
    for row in data:
        csv_out.writerow(row)
        
import csv
with open('results_2.csv','w+') as out:
    csv_out=csv.writer(out)
    for row in results:
        csv_out.writerow(row)




