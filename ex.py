import warnings
warnings.filterwarnings("ignore")
import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from PIL import Image

import tensorflow as tf
from keras.layers import Dense,Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


TRAIN_DIR="./dataset2"
ORG_DIR="/Dataset/Train"
CLASS=['Agaricus','Amanita','Boletus','Cortinarius','Entoloma','Hygrocybe','Lactarius','Russula','Suillus']

for C in CLASS:
    DEST=os.path.join(TRAIN_DIR,C)
    
    if not os.path.exists(DEST):
        os.makedirs(DEST)
    
    for img_path in glob.glob(os.path.join(ORG_DIR,C)+"*"):
        SRC=img_path
        print("Copying", SRC, "to", DEST)
        
        shutil.copy(SRC,DEST)

base_model=InceptionV3(input_shape=(256,256,3),include_top=False)

for layer in base_model.layers:
    layer.trainable=False

X= Flatten()(base_model.output)
X= Dense(units=9,activation='softmax')(X)

#model


model=Model(inputs=base_model.input, outputs=X)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

train_datagen=ImageDataGenerator(featurewise_center=True,rotation_range=0.4,width_shift_range=0.3,horizontal_flip=True,preprocessing_function=preprocess_input,zoom_range=0.4, shear_range=0.4)

train_data=train_datagen.flow_from_directory(directory='./dataset1',target_size=(256,256),batch_size=36)

train_data.class_indices

t_img,label=train_data.next()

def plotImages(img_arr,label):
    for idx ,img in enumerate(img_arr):
        if idx<=10:
            plt.figure(figsize=(5,5))
            plt.imshow(img)
            plt.title(img.shape)
            plt.axis=False
            plt.show()

plotImages(t_img,label)

from keras.callbacks import ModelCheckpoint, EarlyStopping
mc=ModelCheckpoint(filepath="./best_model.h5",
                  monitor="accuracy",
                  verbose=1,
                  save_best_only=True)

es=EarlyStopping(monitor="accuracy",
                min_delta=0.01,
                patience=5,
                verbose=1)
cb=[mc,es]
"""
his= model.fit_generator(train_data,
                        steps_per_epoch=10,
                        epochs=5,
                        callbacks=cb)
"""
from keras.models import load_model
model=load_model("./best_model.h5")

#h=his.history
#h.keys()
"""
plt.plot(h['loss'])
plt.plot(h['accuracy'],'go--',c="red",)

plt.title("loss vs accuracy")
plt.show()
"""
path="img6.jpg"
img=load_img(path,target_size=(256,256))

i=img_to_array(img)
i=preprocess_input(i)

input_arr=np.array([i])
input_arr.shape

pred= np.argmax(model.predict(input_arr))

if pred==0:
    print("Agaricus")
if pred==1:
    print("Amanita")
if pred==2:
    print("Boletus")
if pred==3:
    print("Cortinarius")
if pred==4:
    print("Entoloma")
if pred==5:
    print("Hygrocybe")
if pred==6:
    print("Lactarius")
if pred==7:
    print("Russula")
if pred==8:
    print("Suillus")