from util import get_data, plot_data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# TODO Indicators must return a single results vector. As an example, the MACD indicator can only return one vector.
#  This means it must return a custom scalar array that you develop that provides the information you need,
#  the existing Signal line as an array, or the MACD line as an array.
#  Indicators can only be used once


"""MACD MOVING AVERAGE CONVERGENCE DIVERGENCE"""
def get_macd(symbol, market_data):

    # TODO . If the MACD line crosses from below to above the signal line, that would be a buy signal,
    #  and when the MACD line crosses from above to below, a sell signal is triggered.
    short_ema = market_data.ewm(span=12, adjust=False).mean() # 12 window EMA
    long_ema = market_data.ewm(span=26, adjust=False).mean() # 26 window EMA
    macd = short_ema - long_ema # MACD (12 window - 26 window)
    signal_line = macd.ewm(span=9, adjust=False).mean() # Signal Line (9 window EMA of MACD)

    macd_shifted = macd.shift(1)
    yesterday_macd_lower = (signal_line > macd_shifted)
    yesterday_macd_higher = (signal_line < macd_shifted)
    buy_change = (macd > signal_line) #.diff()
    buy = (buy_change & yesterday_macd_lower).astype(int)
    sell_change = (macd < signal_line)
    sell = (sell_change & yesterday_macd_higher).astype(int) * -1
    signals = sell + buy
    return signals[symbol] # macd, signal_line

"""CCI COMMODITY CHANNEL INDEX"""
def get_cci(symbol, sd, ed):
    sd_padding = sd - timedelta(days=14)
    market_data_high = get_data([symbol], pd.date_range(sd_padding, ed), addSPY=False, colname="High")
    market_data_high.dropna(inplace=True)
    market_data_low = get_data([symbol], pd.date_range(sd_padding, ed), addSPY=False, colname="Low")
    market_data_low.dropna(inplace=True)
    market_data_close = get_data([symbol], pd.date_range(sd_padding, ed), addSPY=False, colname="Close")
    market_data_close.dropna(inplace=True)
    typical_price = (market_data_high + market_data_low + market_data_close) / 3
    SMA = typical_price.rolling(20, min_periods=10).mean()
    mean_deviation = typical_price.rolling(20, min_periods=10).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (typical_price - SMA) / (0.015 * mean_deviation)
    sell = (cci > 100).astype(int) * -1
    buy = (cci < -100).astype(int)
    cci_indicators = buy + sell
    return cci_indicators[sd:ed][symbol]

"""RATE OF CHANGE (ROC)"""
def get_roc(symbol, sd, ed):
    sd_padding = sd - timedelta(days=15)
    market_data = get_data([symbol], pd.date_range(sd_padding, ed), addSPY=False)
    market_data.dropna(inplace=True)
    roc = ((market_data / market_data.shift(15)) - 1) * 100
    buy = (roc > 0).astype(int)
    sell = (roc < 0).astype(int) * -1
    indicators = buy + sell
    return indicators[sd:ed][symbol] # return only within the original sd and ed

"""RSI (Relative Strength Index)"""
def get_rsi(symbol, sd, ed):
    sd_padding = sd - timedelta(days=15)
    market_data = get_data([symbol], pd.date_range(sd_padding, ed), addSPY=False)
    market_data.dropna(inplace=True)
    difference_in_price = market_data.diff(1)
    change_up = difference_in_price.copy()
    change_down = difference_in_price.copy()
    change_up[change_up < 0] = 0
    change_down[change_down > 0] = 0
    avg_loss = change_down.rolling(window=14, min_periods=1).mean().abs()
    avg_gain = change_up.rolling(window=14, min_periods=1).mean()
    rs = (avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    sell = (rsi > 70).astype(int) * -1
    buy = (rsi < 30).astype(int)
    indicators = buy + sell
    return indicators[sd:ed][symbol]

"""SIMPLE MOVING AVERAGE (SMA) AND BOLLINGER BANDS"""
def get_sma_bollinger_bands(market_data, symbol):
    SMA = pd.DataFrame(market_data[symbol].rolling(window=20, min_periods=1).mean()) # SMA = Simple moving average
    std = market_data[symbol].rolling(window=20, min_periods=1).std()
    two_std = std * 2
    upper_band = pd.DataFrame(SMA).add(pd.DataFrame(two_std))
    lower_band = pd.DataFrame(SMA).sub(pd.DataFrame(two_std))
    bands_df = (pd.DataFrame(market_data['JPM'])
                .join(SMA, rsuffix='_sma')
                .join(lower_band,  rsuffix='_lower')
                .join(upper_band, rsuffix='_upper'))
    BBP = (market_data - lower_band) / (upper_band - lower_band)
    buy = (BBP < 0.0).astype(int)
    sell = (BBP > 1.0).astype(int) * -1
    indicators = buy + sell
    return indicators[symbol] # bands_df, BBP

def get_bollinger_bands(df):
    # Bollinger band feature: bb_value[t] = (price[t] - SMA[t])/(2 * stdev[t])
    SMA = df.rolling(20).mean() # SMA = Simple moving average
    std = df.rolling(20).std()
    two_std = std * 2
    upper_band = SMA + two_std
    lower_band = SMA - two_std
    return upper_band, lower_band



def author():
  return 'jtyndale3'

def main():
  print('hello world')

if __name__ == "__main__":
    main()
