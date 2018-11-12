# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:46:02 2018

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
from keras import layers
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error


satvalue = 100000000
window_len = 1
test_size = 0.2
zero_base = False
#target_col = 'close'
# model params
lstm_neurons = 256
epochs = 50
batch_size = 10
loss = 'mse'
dropout = 0.25
optimizer = 'adam'

def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18);

def line_plot_multiple(line1, line2, line3, line4, label1=None, label2=None, label3=None, label4=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.plot(line3, label=label3, linewidth=lw)
    ax.plot(line4, label=label4, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18);

def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row-1]
    test_data = df.iloc[split_row-1:-1]
    train_target = df.iloc[1:split_row]
    test_target = df.iloc[split_row:]
    return train_data, test_data, train_target, test_target

def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with respect to first entry. """
    return df / df.iloc[0] - 1

def normalise_min_max(df):
    """ Normalise dataframe column-wise min/max. """
    return (df - df.min()) / (df.max() - df.min())

def extract_window_data(df, window_len=1, zero_base=True):
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
    return np.array(window_data)
    #return window_data

def prepare_data_high(df, target_col, window_len=1, zero_base=False, test_size=0.2):
    """ Prepare data for LSTM. """
    # train test split
    train_data, test_data, train_target, test_target = train_test_split(df, test_size=test_size)
    
    # extract window data
    X_train = train_data.values
    X_test = test_data.values
    
    # extract targets
    #train_target, test_target = train_test_split(df.iloc[1:], test_size=test_size)
    y_train = train_target[target_col].values
    y_test = test_target[target_col].values
    if zero_base:
        y_train = y_train / train_target[target_col][:-window_len].values - 1
        y_test = y_test / test_target[target_col][:-window_len].values - 1

    return train_data, test_data, train_target, test_target, X_train, X_test, y_train, y_test

def build_lstm_model(input_data, output_size, neurons=256, activ_func='linear',
                     dropout=0.25, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(layers.Dense(3, activation = "relu", input_shape=(input_data.shape[1], )))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    #model.add(layers.Dense((input_data.shape[1]+1)/2, activation = "relu"))
    #model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(3, activation = "relu"))
    model.add(layers.Dense(1, activation = "linear"))
    
    #model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    #model.add(Dropout(dropout))
    #model.add(Dense(units=output_size))
    #model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

if __name__== "__main__":
    with open('tickerlist.pkl', 'rb') as f:
        tickerlist = pickle.load(f)
    
    train, test, X_train, X_test, y_train, y_test = [], [], [], [], [], []
    
    #for symbol in tickerlist:
    symbol = 'ADABTC'
    print(symbol)
    df = pd.read_hdf("added_params/"+symbol+".h5")
    df = df.drop(['time'], axis=1)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    
    #convert to float
    #df[['open','high','low','close','sma','upper','middle','lower','atr']] = df[['open','high','low','close']].apply(pd.to_numeric)
    
    #convert to sat range
    df.loc[:,'open'] *= satvalue
    df.loc[:,'high'] *= satvalue
    df.loc[:,'low'] *= satvalue
    df.loc[:,'close'] *= satvalue
    df.loc[:,'sma'] *= satvalue
    df.loc[:,'upper'] *= satvalue
    df.loc[:,'middle'] *= satvalue
    df.loc[:,'lower'] *= satvalue
    df.loc[:,'atr'] *= satvalue
    
    #df = df[['open', 'high', 'low', 'close', 'vol', 'upper', 'lower', 'middle']]
    df = df[['open', 'high', 'low', 'close', 'vol']]

        
    train, test, train_target, test_target, X_train, X_test, y_train, y_test = prepare_data_high(
    df, target_col = 'high', window_len=window_len, zero_base=zero_base, test_size=test_size)
    
    model = build_lstm_model(
        X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    history = model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    
    preds = model.predict(X_test).squeeze()
    #mean_absolute_error(preds, y_test)
    target = test_target['high'][window_len:]
    
    predic = test_target['high'].values[:-window_len] * (preds+1)
    
    predic = pd.Series(index=target.index, data=predic)
    
    #line_plot(target, predic, 'actual', 'prediction', lw=3)
    
    
    ##### LOW LOW LOW LOW
    l_train, l_test, l_train_target, l_test_target, l_X_train, l_X_test, l_y_train, l_y_test = prepare_data_high(
    df, target_col = 'low', window_len=window_len, zero_base=zero_base, test_size=test_size)
    
    l_model = build_lstm_model(
        l_X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    l_history = l_model.fit(
        l_X_train, l_y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    
    l_preds = l_model.predict(l_X_test).squeeze()
    #mean_absolute_error(l_preds, l_y_test)
    l_target = l_test_target['low'][window_len:]
    
    l_predic = l_test_target['low'].values[:-window_len] * (l_preds+1)
    
    l_predic = pd.Series(index=l_target.index, data=l_predic)
    
    #line_plot(target, predic, 'actual', 'prediction', lw=3)
    
    line_plot_multiple(target, predic, l_target, l_predic,  'actual_high', 'high_pred', 'actual_low', 'low_pred', lw=3)
    
    print("mae high prediction : " + str(mean_absolute_error(preds, y_test)))
    print("mae low prediction : " + str(mean_absolute_error(l_preds, l_y_test)))
