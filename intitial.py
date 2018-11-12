# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
                    kline data
                     1499040000000,      # Open time
                    "0.01634790",       # Open
                    "0.80000000",       # High
                    "0.01575800",       # Low
                    "0.01577100",       # Close
                    "148976.11427815",  # Volume
                    1499644799999,      # Close time
                    "2434.19055334",    # Quote asset volume
                    308,                # Number of trades
                    "1756.87402397",    # Taker buy base asset volume
                    "28.46694368",      # Taker buy quote asset volume
                    "17928899.62484339" # Can be ignored
"""

import json
import pandas as pd
import pickle
from pandas import Series, DataFrame, HDFStore
import numpy as np
from binance.client import Client
import talib
from talib.abstract import *

#satvalue = 100000000

   
def addTALIBindicators(symbol):
    df = pd.read_hdf("klines/"+symbol+".h5")
    
    df = df.drop(['qav', 'tbbav', 'tbqav', 'ig', 'ctime'], axis=1)
    """
    #convert to sat range
    df.loc[:,'open'] *= satvalue
    df.loc[:,'high'] *= satvalue
    df.loc[:,'low'] *= satvalue
    df.loc[:,'close'] *= satvalue
    """
    lopen, lhigh, llow, lclose, lvolume = ([], [], [], [], []) #listopen, listclose etc.
    
    lopen = df['open'].astype('float64') 
    lhigh = df['high'].astype('float64') 
    llow = df['low'].astype('float64') 
    lclose = df['close'].astype('float64') 
    lvolume = df['vol'].astype('float64') 

    inputs = {
            'open' : np.asarray(lopen),
            'high' : np.asarray(lhigh),
            'low' : np.asarray(llow),
            'close' : np.asarray(lclose),
            'volume' : np.asarray(lvolume),
            }
    
    # uses close prices (default)
    sma = SMA(inputs, timeperiod=25)
    df['sma'] = Series(np.asarray(sma), index=df.index)

    # uses open prices
    #opensma = SMA(inputs, timeperiod=25, price='open')
    
    # uses close prices (default)
    upper, middle, lower = BBANDS(inputs, 20, 2, 2)
    df['upper'] = Series(np.asarray(upper), index=df.index)
    df['middle'] = Series(np.asarray(middle), index=df.index)
    df['lower'] = Series(np.asarray(lower), index=df.index)

    # uses high, low, close (default)
    slowk, slowd = STOCH(inputs, 5, 3, 0, 3, 0) # uses high, low, close by default
    df['slowk'] = Series(np.asarray(slowk), index=df.index)
    df['slowd'] = Series(np.asarray(slowd), index=df.index)
    
    # uses high, low, open instead
    #slowk, slowd = STOCH(inputs, 5, 3, 0, 3, 0, prices=['high', 'low', 'open'])
    
    #Average True Range
    atr = ATR(inputs)
    df['atr'] = Series(np.asarray(atr), index=df.index)
    
    #Rate of change Percentage: (price-prevPrice)/prevPrice
    rocp = ROCP(inputs)
    df['rocp'] = Series(np.asarray(rocp), index=df.index)
    
    return df




if __name__== "__main__":
    client = Client("", "")
    
    #saveHistoricalData("ADABTC")
    #df = addTALIBindicators("ADABTC")
    
    with open('tickerlist.pkl', 'rb') as f:
        tickerlist = pickle.load(f)
    
    for ticker in tickerlist:
        df = addTALIBindicators(ticker)
        store = HDFStore(ticker+'.h5')
        store['df'] = df  # save it
        
    
    """
    tickers = client.get_ticker()
    tickerlist = []
    for ticker in tickers:
        if 'BTC' in ticker['symbol']:
            tickerlist.append(ticker['symbol'])
    
    with open('tickerlist.pkl', 'wb') as f:
        pickle.dump(tickerlist, f)
    """
        
    """
    tickers = client.get_ticker()
    klines = client.get_historical_klines("ADABTC", Client.KLINE_INTERVAL_4HOUR, "1 Jan, 2017")
    for ticker in tickers:
        if 'BTC' in ticker['symbol']:
            symbol = ticker['symbol']
            print(symbol)
            klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_4HOUR, "1 Jan, 2017")
            df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'ctime', 'qav', 'ntrades', 'tbbav', 'tbqav', 'ig'])
            df.drop(['qav', 'tbbav', 'tbqav', 'ig'], axis=1)
            store = HDFStore(symbol+'.h5')
            store['df'] = df  # save it
            
    
    """
        