"""
Model made for Histopathologic Cancer Detection Challenge
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from skimage.io import imread
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data():
    train_df = pd.read_csv("../input/train_labels.csv")
    train, validation = train_test_split(train_df, test_size=0.25)
    
    return train, validation
    
def augment_data(train, validation):
    train_dgen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True)
    valid_dgen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_dgen.flow_from_dataframe(
        dataframe=train,
        directory='../input/train/',
        x_col='id',
        y_col='label',
        has_ext=False,
        shuffle=True,
        batch_size=64,
        target_size=(96, 96),
        class_mode = 'binary')
    valid_gen = valid_dgen.flow_from_dataframe(
        dataframe=validation,
        directory='../input/train/',
        x_col='id',
        y_col='label',
        has_ext=False,
        batch_size=64,
        shuffle=True,
        target_size=(96, 96),
        class_mode = 'binary')
        
    return train_gen, valid_gen

def create_model():
    res_net = ResNet50(weights='imagenet',include_top=False,input_shape=(96,96,3))
    
    inputs = Input((96,96,3))

    outputs = GlobalAveragePooling2D()(res_net(inputs))
    outputs = Dropout(0.5)(outputs)
    outputs = Dense(1, activation="sigmoid")(outputs)

    model = Model(inputs, outputs)

    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['acc'])
    
    model.summary()
    
    return model
    
def train_model(model, train_gen, valid_gen):
    t_steps = train_gen.n//train_gen.batch_size
    v_steps = valid_gen.n//valid_gen.batch_size
    model.fit_generator(train_gen,
                    steps_per_epoch=t_steps ,
                    validation_data=valid_gen,
                    validation_steps=v_steps,
                    epochs=12)
                    
    return model
    
def test_model(model,batch=5000):
    testing_files = glob(os.path.join('../input/test/','*.tif'))
    submission = pd.DataFrame()
    for index in range(0, len(testing_files), batch):
        data_frame = pd.DataFrame({'path': testing_files[index:index+batch]})
        data_frame['id'] = data_frame.path.map(lambda x: x.split('/')[3].split(".")[0])
        data_frame['image'] = data_frame['path'].map(imread)
        images = np.stack(data_frame.image, axis=0)
        predicted_labels = [model.predict(np.expand_dims(image/255.0, axis=0))[0][0] for image in images]
        predictions = np.array(predicted_labels)
        data_frame['label'] = predictions
        submission = pd.concat([submission, data_frame[["id", "label"]]])
    submission.to_csv("submission.csv", index=False, header=True)

train, validation = get_data()
train_gen, valid_gen = augment_data(train, validation)
model = create_model()
model = train_model(model, train_gen, valid_gen)
test_model(model,batch=5000)