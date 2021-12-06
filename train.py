import pandas as pd
import numpy as np
import os
import pickle 
from utils import  prep_data,calc_nse,kge
from get_model import lstm_model
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint,TensorBoard

### Define initial and final timestamp
inicial='1981-01-01'
last='2009-12-31'

### define sequence length
time_steps = 120

max_epocs=100
lstm_batch_size=256
learning_rate=0.001
stop_patience=10

### input data
path='/home/pedrozamboni/Documentos/doutorado/dataset/cabra/v1'
basin_code=50
x_train, y_train, x_val, y_val,x_test, y_test=prep_data(path,basin_code,inicial,last,['PREC', 'ET'],'streamflow(mm/dia)',
train_split=0.6,val_split=0.2,test_split=0.2,time_steps=120)

### model 
experiment_name = ''
filepath = os.path.join(experiment_name, "model_{epoch:06d}.h5")
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, period=20, save_weights_only=False)
tensorboard = TensorBoard(os.path.join(experiment_name,'logs'), write_graph=False)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=stop_patience,
                                                mode='min')
callbacks_list = [early_stopping,checkpoint, tensorboard]

model = lstm_model(x_train.shape[1:],256,0.5,1)
model.summary()

customAdam = keras.optimizers.Adam(lr=learning_rate)   


model.compile(loss=tf.losses.MeanSquaredError(),
        optimizer=customAdam,
        metrics=[tf.metrics.MeanSquaredError()])

history = model.fit(x_train, y_train ,epochs=max_epocs,
                batch_size=lstm_batch_size,
                validation_data=(x_val,y_val), shuffle=True,
                callbacks=callbacks_list)

model.save(os.path.join(experiment_name,'model_final.h5'))