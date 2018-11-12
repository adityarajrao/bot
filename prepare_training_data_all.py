# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 22:41:13 2018

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
"""
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
"""
window_len = 200
test_size = 0.2
zero_base = False
#target_col = 'close'
# model params
lstm_neurons = 20
epochs = 50
batch_size = 4
loss = 'mae'
dropout = 0.25
optimizer = 'adam'

def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with respect to first entry. """
    return df / df.iloc[0] - 1

def normalise_min_max(df):
    """ Normalise dataframe column-wise min/max. """
    return (df - df.min()) / (df.max() - df.min())

def extract_window_data(df, window_len=200, zero_base=False):
    """ Convert dataframe to overlapping sequences/windows of len `window_data`.
    
        :param window_len: Size of window
        :param zero_base: If True, the data in each window is normalised to reflect changes
            with respect to the first entry in the window (which is then always 0)
    """
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    #return np.array(window_data)
    return window_data

def prepare_data_high(df, target_col, window_len=200, zero_base=False, test_size=0.2):
    """ Prepare data for LSTM. """
    # train test split
    train_data, test_data = train_test_split(df.iloc[:-1], test_size=test_size)
    
    # extract window data
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    
    # extract targets
    train_target, test_target = train_test_split(df.iloc[1:], test_size=test_size)
    y_train = train_target[target_col][window_len:].values
    y_test = test_target[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test
"""
def build_lstm_model(input_data, output_size, neurons=20, activ_func='linear',
                     dropout=0.25, loss='mae', optimizer='adam'):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model
"""
if __name__== "__main__":
    with open('tickerlist.pkl', 'rb') as f:
        tickerlist = pickle.load(f)
    
    train, test, X_train, X_test, y_train, y_test = [], [], [], [], [], []
    
    for symbol in tickerlist:
        #symbol = 'ADABTC'
        print(symbol)
        df = pd.read_hdf("added_params/"+symbol+".h5")
        df = df.drop(['time'], axis=1)
        df = df.dropna()
        
        l_train, l_test, l_X_train, l_X_test, l_y_train, l_y_test = prepare_data_high(
        df, target_col = 'high', window_len=window_len, zero_base=zero_base, test_size=test_size)
        
        if len(l_y_test) > 10 :
            symbolhighdata = {}
            symbolhighdata['l_train'] = l_train
            symbolhighdata['l_test'] = l_test
            symbolhighdata['l_X_train'] = l_X_train
            symbolhighdata['l_X_test'] = l_X_test
            symbolhighdata['l_y_train'] = l_y_train
            symbolhighdata['l_y_test'] = l_y_test
            
            with open('symbol_data/' + symbol + 'highdata.pickle', 'wb') as handle:
                pickle.dump(symbolhighdata, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #if len(l_y_test) > 240 :
            train.append(l_train)
            test.append(l_test)
            X_train.append(l_X_train)
            X_test.append(l_X_test)
            y_train.append(l_y_train)
            y_test.append(l_y_test)
    
    trainarr = np.concatenate([np.array(x) for x in train])
    testarr = np.concatenate([np.array(x) for x in test])
    X_trainarr = np.concatenate([np.array(x) for x in X_train])
    X_testarr = np.concatenate([np.array(x) for x in X_test])
    y_trainarr = np.concatenate([np.array(x) for x in y_train])
    y_testarr = np.concatenate([np.array(x) for x in y_test])
    
    highdata = {}
    highdata['trainarr'] = trainarr
    highdata['testarr'] = testarr
    highdata['X_trainarr'] = X_trainarr
    highdata['X_testarr'] = X_testarr
    highdata['y_trainarr'] = y_trainarr
    highdata['y_testarr'] = y_testarr
    
    with open('model_data/highdata.pickle', 'wb') as handle:
        pickle.dump(highdata, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    #with open('highdata.pickle', 'rb') as handle:
    #    highdata = pickle.load(handle)
    
    train, test, X_train, X_test, y_train, y_test = [], [], [], [], [], []
    
    for symbol in tickerlist:
        #symbol = 'ADABTC'
        print(symbol)
        df = pd.read_hdf("added_params/"+symbol+".h5")
        df = df.drop(['time'], axis=1)
        df = df.dropna()
        
        l_train, l_test, l_X_train, l_X_test, l_y_train, l_y_test = prepare_data_high(
        df, target_col = 'low', window_len=window_len, zero_base=zero_base, test_size=test_size)
        
        if len(l_y_test) > 10 :
            symbollowdata = {}
            symbollowdata['l_train'] = l_train
            symbollowdata['l_test'] = l_test
            symbollowdata['l_X_train'] = l_X_train
            symbollowdata['l_X_test'] = l_X_test
            symbollowdata['l_y_train'] = l_y_train
            symbollowdata['l_y_test'] = l_y_test
            
            with open('symbol_data/' + symbol + 'lowdata.pickle', 'wb') as handle:
                pickle.dump(symbollowdata, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
        #if len(l_y_test) > 240 :
            train.append(l_train)
            test.append(l_test)
            X_train.append(l_X_train)
            X_test.append(l_X_test)
            y_train.append(l_y_train)
            y_test.append(l_y_test)
    
    trainarr = np.concatenate([np.array(x) for x in train])
    testarr = np.concatenate([np.array(x) for x in test])
    X_trainarr = np.concatenate([np.array(x) for x in X_train])
    X_testarr = np.concatenate([np.array(x) for x in X_test])
    y_trainarr = np.concatenate([np.array(x) for x in y_train])
    y_testarr = np.concatenate([np.array(x) for x in y_test])
    
    lowdata = {}
    lowdata['trainarr'] = trainarr
    lowdata['testarr'] = testarr
    lowdata['X_trainarr'] = X_trainarr
    lowdata['X_testarr'] = X_testarr
    lowdata['y_trainarr'] = y_trainarr
    lowdata['y_testarr'] = y_testarr
    
    with open('model_data/lowdata.pickle', 'wb') as handle:
        pickle.dump(lowdata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    #model = build_lstm_model(
    #X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,optimizer=optimizer)
    #history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    
    #targets = y_test
    #preds = model.predict(X_test).squeeze()
    
    #mean_absolute_error(preds, y_test)
