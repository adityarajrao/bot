# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 09:49:02 2018

@author: Aditya Raj
"""

import pandas as pd
from binance.client import Client

def saveHistoricalBTCpairKlines():
    client = Client("", "")
    tickers = client.get_ticker()
    klines = client.get_historical_klines("ADABTC", Client.KLINE_INTERVAL_4HOUR, "1 Jan, 2017")
    for ticker in tickers:
        if 'BTC' in ticker['symbol']:
            symbol = ticker['symbol']
            print(symbol)
            klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_4HOUR, "1 Jan, 2017")
            df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'ctime', 'qav', 'ntrades', 'tbbav', 'tbqav', 'ig'])
            df.drop(['qav', 'tbbav', 'tbqav', 'ig'], axis=1)
            store = pd.HDFStore('klines/'+symbol+'.h5')
            store['df'] = df  # save it
