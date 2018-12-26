# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:31:34 2017

@author: lisssse14
"""

import cv2
import shutil
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import h5py
from glob import glob
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
#df = pd.read_csv('labels.csv')
#df = pd.read_csv('lab.csv')
#n = len(df)
#breed = set(df['breed'])
#n_class = len(breed)
n = len(glob('Images/*/*.jpg'))
#class_to_num = dict(zip(breed, range(n_class)))
#num_to_class = dict(zip(range(n_class), breed))
df_test = pd.read_csv('sample_submission.csv')
n_test = len(df_test)
synset = list(df_test.columns[1:])
n_class = len(synset)
#%%
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.callbacks import *
from keras.preprocessing.image import *
from keras.utils import np_utils

def write_gap(MODEL,image_size,lambda_function=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((width, height, 3))
    x = input_tensor
    if lambda_function:
        x = Lambda(lambda_function)(x)
    
    base_model = MODEL(input_tensor=x,include_top=False, weights='imagenet')
#    x = GlobalAveragePooling2D()(base_model)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    X = np.zeros((n, width, height, 3), dtype=np.uint8)
    y = np.zeros((n,), dtype=np.uint8)
#    y = np.zeros((n, n_class), dtype=np.uint8)
    X_test = np.zeros((n_test, width, height, 3), dtype=np.uint8)
    # train
    for i, file_name in tqdm(enumerate(glob('Images/*/*.jpg')), total=n):
        img = cv2.imread(file_name)
        img = cv2.resize(img, (width, height))
        X[i] = img
#    for i in tqdm(range(n)):
#        X[i] = cv2.resize(cv2.imread('images/%s.jpg' % df['id'][i]), (width, height))
#        X[i] = cv2.resize(cv2.imread('images/%s/%s.jpg' % (df['breed'][i],df['id'][i])), (width, height))
        y[i] = synset.index(file_name.split('\\')[1])
#        y[i][class_to_num[df['breed'][i]]] = 1
    for i, fname in tqdm(enumerate(df_test['id']), total=n_test):
        img = cv2.imread('test/%s.jpg' % fname)
        img = cv2.resize(img, (width, height))
        X_test[i] = img
#    for i in tqdm(range(n_test)):
#        X_test[i] = cv2.resize(cv2.imread('test/%s.jpg' % df2['id'][i]), (width, height))  

    train = model.predict(X, batch_size=32, verbose=1)
    test = model.predict(X_test, batch_size=32, verbose=1)
    with h5py.File("gap_%s.h5" % model.name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=y)
#%%
write_gap(ResNet50,(224,224))
write_gap(InceptionV3,(299,299),inception_v3.preprocess_input)
write_gap(Xception,(299,299),xception.preprocess_input)
#%%
#np.random.seed(2018)
X_train = []
for filename in ["gap_ResNet50.h5","gap_Xception.h5", "gap_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_train,y_train = shuffle(X_train,y_train)
y_train = np_utils.to_categorical(y_train,n_class)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
#%%
X_test = []
for filename in ["gap_ResNet50.h5","gap_Xception.h5", "gap_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X_test.append(np.array(h['test']))

X_test = np.concatenate(X_test, axis=1)
#%%
Adam = optimizers.Adam(lr=1e-3,decay=0.0005)
#SGD = optimizers.SGD(lr=0.1,momentum=0.9,decay=0.005,nesterov=True)
inputs = Input(X_train.shape[1:])
x = inputs
x= BatchNormalization()(x)
x = Dense(512,kernel_initializer='he_normal',use_bias=False)(x)
x= BatchNormalization(scale=False)(x)
x= Activation('relu')(x)
#x = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(1e-4))(x)
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax')(x)
model = Model(inputs, x)
model.compile(optimizer=Adam,loss='categorical_crossentropy',metrics=['accuracy'])
#h = model.fit(X_train, y_train, batch_size=128, epochs=10,validation_data = (X_val,y_val))
#h = model.fit(X_train, y_train, batch_size=128, epochs=30,validation_split=0.2)
h = model.fit(X_train, y_train, batch_size=256, epochs=80)
#%%
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')

plt.subplot(1, 2, 2)
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.ylabel('acc')
plt.xlabel('epoch')
#%%
y_pred = model.predict(X_test, batch_size=128)
df_pred = pd.read_csv('sample_submission.csv')
#for b in breed:
#    df2[b] = y_pred[:,class_to_num[b]]
for i, c in enumerate(df_pred.columns[1:]):
    df_pred[c] = y_pred[:,i]

df_pred.to_csv('pred.csv', index=None)