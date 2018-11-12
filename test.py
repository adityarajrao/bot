# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 23:13:43 2018

@author: Aditya Raj
"""

import json
import pandas as pd
from pandas import Series, DataFrame, HDFStore
import numpy as np
from binance.client import Client
import talib
from talib.abstract import *

satvalue = 1#00000000

if __name__== "__main__":
    symbol = 'ADABTC'
    df = pd.read_hdf("klines/"+symbol+".h5")
    
    df = df.drop(['qav', 'tbbav', 'tbqav', 'ig'], axis=1)
        #convert to sat range
    df.loc[:,'open'] *= satvalue
    df.loc[:,'high'] *= satvalue
    df.loc[:,'low'] *= satvalue
    df.loc[:,'close'] *= satvalue

    
    lopen, lhigh, llow, lclose, lvolume = ([], [], [], [], []) #listopen, listclose etc.
    
    lopen = df['open'].astype('float64') 
    lhigh = df['high'].astype('float64') 
    llow = df['low'].astype('float64') 
    lclose = df['close'].astype('float64') 
    lvolume = df['vol'].astype('float64') 
    
    a = np.asarray(lopen)
    b = np.random.random(100)

    inputs = {
            'open' : np.asarray(lopen),
            'high' : np.asarray(lhigh),
            'low' : np.asarray(llow),
            'close' : np.asarray(lclose),
            'volume' : np.asarray(lvolume),
            }
    
    # uses close prices (default)
    sma = SMA(inputs, timeperiod=25)

    # uses open prices
    opensma = SMA(inputs, timeperiod=25, price='open')
    
    # uses close prices (default)
    upper, middle, lower = BBANDS(inputs, 20, 2, 2)

    # uses high, low, close (default)
    slowk, slowd = STOCH(inputs, 5, 3, 0, 3, 0) # uses high, low, close by default
    
    # uses high, low, open instead
    #slowk, slowd = STOCH(inputs, 5, 3, 0, 3, 0, prices=['high', 'low', 'open'])
    
    df['sma'] = Series(np.asarray(sma), index=df.index)
    
    #Average True Range
    atr = ATR(inputs)
    df['atr'] = Series(np.asarray(atr), index=df.index)
"""    
    #candlestick pattern recognition
    #CDLDOJI - Doji
    doji = CDLDOJI(inputs)
    df['doji'] = Series(np.asarray(doji), index=df.index)
    
    #CDLMARUBOZU - Marubozu
    marubozu = CDLMARUBOZU(inputs)
    df['marubozu'] = Series(np.asarray(marubozu), index=df.index)
    
    #CDLHARAMI - Harami Pattern
    harami = CDLHARAMI(inputs)
    df['harami'] = Series(np.asarray(harami), index=df.index)
    
    #CDLENGULFING - Engulfing Pattern
    engulfing = CDLENGULFING(inputs)
    df['engulfing'] = Series(np.asarray(engulfing), index=df.index)
    
    #CDLDARKCLOUDCOVER - Dark Cloud Cover
    dcc = CDLDARKCLOUDCOVER(inputs)
    df['dcc'] = Series(np.asarray(dcc), index=df.index)
    
    #CDLHAMMER - Hammer
    hammer = CDLHAMMER(inputs)
    df['hammer'] = Series(np.asarray(hammer), index=df.index)
    
    #CDLINVERTEDHAMMER - Inverted Hammer
    ihammer = CDLINVERTEDHAMMER(inputs)
    df['ihammer'] = Series(np.asarray(ihammer), index=df.index)
    
    #CDLMORNINGSTAR - Morning Star
    mstar = CDLMORNINGSTAR(inputs)
    df['mstar'] = Series(np.asarray(mstar), index=df.index)
    
    #CDL3BLACKCROWS - Three Black Crows
    tbc = CDL3BLACKCROWS(inputs)
    df['tbc'] = Series(np.asarray(tbc), index=df.index)
    
    #CDLHIKKAKE - Hikkake Pattern
    hikkake = CDLHIKKAKE(inputs)
    df['hikkake'] = Series(np.asarray(hikkake), index=df.index)
"""

"""
def saveHistoricalData(symbol):
    global df
    global inputs
    global a
    global b
    global klines
    #klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE, "18 July, 2018")
    #df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'ctime', 'qav', 'ntrades', 'tbbav', 'tbqav', 'ig'])
    #df.drop(['qav', 'tbbav', 'tbqav', 'ig'], axis=1)
    #store = HDFStore(symbol+'.h5')
    #store['df'] = df  # save it
    df = pd.read_hdf("klines/"+symbol+".h5")
    
    #calculate and add indicators
    rsi, sma, upper, middle, lower = ([], [], [], [], [])
    lopen, lhigh, llow, lclose, lvolume = ([], [], [], [], [])    #listopen, listclose etc.
    for index, row in df.iterrows():
        if index<99 :
            rsi.append(None)
            sma.append(None)
            upper.append(None)
            middle.append(None)
            lower.append(None)
            lopen.append(float(row['open']))
            lhigh.append(float(row['high']))
            llow.append(float(row['low']))
            lclose.append(float(row['close']))
            lvolume.append(float(row['vol']))
            
        else:
            lopen.append(float(row['open']))
            lhigh.append(float(row['high']))
            llow.append(float(row['low']))
            lclose.append(float(row['close']))
            lvolume.append(float(row['vol']))
            a = np.asarray(lopen)
            b = np.random.random(100)
            inputs = {
                    'open' : np.asarray(lopen),
                    'high' : np.asarray(lhigh),
                    'low' : np.asarray(llow),
                    'close' : np.asarray(lclose),
                    'volume' : np.asarray(lvolume),
                    }
            
            # uses close prices (default)
            output = SMA(inputs, timeperiod=25)
            print(output)
            sma.append(output)
            # uses open prices
            output = SMA(inputs, timeperiod=25, price='open')
            
            # uses close prices (default)
            cupper, cmiddle, clower = BBANDS(inputs, 20, 2, 2)
            upper.append(cupper)
            middle.append(cmiddle)
            lower.append(clower)
            # uses high, low, close (default)
            slowk, slowd = STOCH(inputs, 5, 3, 0, 3, 0) # uses high, low, close by default
            
            # uses high, low, open instead
            slowk, slowd = STOCH(inputs, 5, 3, 0, 3, 0, prices=['high', 'low', 'open'])
            
            del lopen[0]
            del lhigh[0]
            del llow[0]
            del lclose[0]
            del lvolume[0]
    
    df['sma'] = Series(np.asarray(sma), index=df.index)
 """