# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 19:51:53 2018

@author: Aditya Raj
"""

import json
import pandas as pd
import pickle
from pandas import Series, DataFrame, HDFStore
import numpy as np
from binance.client import Client
import talib
from talib.abstract import *

import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

satvalue = 100000000
window_len = 10
test_size = 0.2
zero_base = True
#target_col = 'close'
# model params
lstm_neurons = 20
epochs = 50
batch_size = 1000
loss = 'mae'
dropout = 0.25
optimizer = 'adam'

def build_lstm_model(input_data, output_size, neurons=20, activ_func='linear',
                     dropout=0.25, loss='mae', optimizer='adam'):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


if __name__== "__main__":
    with open('tickerlist.pkl', 'rb') as f:
        tickerlist = pickle.load(f)
    
    with open('model_data/highdata.pickle', 'rb') as handle:
        highdata = pickle.load(handle)
    

    train = highdata['trainarr']
    test = highdata['testarr']
    X_train = highdata['X_trainarr']
    X_test = highdata['X_testarr']
    y_train = highdata['y_trainarr']
    y_test = highdata['y_testarr']
    
    model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,optimizer=optimizer)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    
    model.save('model_data/high_model.h5')  # creates a HDF5 file 'my_model.h5'
    #model = load_model('model_data/high_model.h5')
    
    targets = y_test
    preds = model.predict(X_test).squeeze()
    
    mean_absolute_error(preds, y_test)
    
    print("low low low low")
    #training for model predicting low
    with open('model_data/lowdata.pickle', 'rb') as handle:
        lowdata = pickle.load(handle)
    

    l_train = lowdata['trainarr']
    l_test = lowdata['testarr']
    l_X_train = lowdata['X_trainarr']
    l_X_test = lowdata['X_testarr']
    l_y_train = lowdata['y_trainarr']
    l_y_test = lowdata['y_testarr']
    
    l_model = build_lstm_model(
    l_X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,optimizer=optimizer)
    l_history = l_model.fit(l_X_train, l_y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    
    l_model.save('model_data/low_model.h5')  # creates a HDF5 file 'my_model.h5'
    #model = load_model('model_data/low_model.h5')
    
    l_targets = l_y_test
    l_preds = l_model.predict(l_X_test).squeeze()
    
    mean_absolute_error(l_preds, l_y_test)
    
    
