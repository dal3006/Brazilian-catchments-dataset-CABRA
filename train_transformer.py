import pandas as pd
import numpy as np
import os
import pickle 
from utils import  prep_data,calc_nse,kge
from get_model import build_model
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
path='/content/drive/MyDrive/cabra'

basin_code=50
x_train, y_train, x_val, y_val,x_test, y_test=prep_data(path,basin_code,inicial,last,['PREC', 'ET'],'streamflow(mm/dia)',
train_split=0.6,val_split=0.2,test_split=0.2,time_steps=120)

input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=10,
    ff_dim=4,
    num_transformer_blocks=6,
    mlp_units=[256],
    mlp_dropout=0.5,
    dropout=0.25,
)

model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["mean_squared_error"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)