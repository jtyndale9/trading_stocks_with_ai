"""An improved version of your marketsim code accepts a “trades” DataFrame (instead of a file).
 More info on the trades data frame is below. It is OK not to submit this file if you have subsumed its functionality
 into one of your other required code files. This file has a different name and a slightly different setup than
 your previous project. However, that solution can be used with several edits for the new requirements.  """

"""
df_trades: A single column data frame, indexed by date, whose values represent trades for each trading day 
(from the start date to the end date of a given period). Legal values are +1000.0 indicating a BUY of 1000 shares, 
-1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING. Values of +2000 and -2000 for trades are also 
legal so long as net holdings are constrained to -1000, 0, and 1000. Note: The format of this data frame differs 
from the one developed in a prior project.
"""

import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import math

# TODO the 'marketsimcode.py' accepts a 'trades' dataframe instead of a file.
# TODO One thing I might say differently is that df_trades is the trading activity (rather than net stock position) for each day.

def compute_portvals(
    orders_df,  # Path of the order file or the file object
    start_val=100000,  # (int) – The starting value of the portfolio
    commission=9.95,  # was 9.95. (float) – The fixed amount in dollars charged for each transaction (both entry and exit)
    impact=0.005,  # was 0.005 (float) – amount the price moves against the trader compared to historical data at each transaction
):
    # TODO the 'marketsimcode.py' accepts a 'trades' dataframe

    # TODO: get the start date and end date of orders_file (they max not be in order, get min max)
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()

    # TODO: read the market data using Util.py
    #market_data_to_read = orders_df.Symbol.unique()
    market_data_to_read = orders_df.columns[0]
    symbol = orders_df.columns[0]
    market_data = get_data([market_data_to_read], pd.date_range(start_date, end_date))

    # TODO: add another column onto dataframe 1 ("prices_df") at the end for cash. Fill with 1's?
    market_data['cash'] = 1.0

    # TODO: Make a copy of dataframe 1, initially filled with zeroes, (this is dataframe 2 - "trades_df_2").
    #           Everytime a change is made (order is made), log that change in stock and in cash
    #           This is an Orders dataframe
    trades_df = market_data.copy()
    trades_df[trades_df.columns] = 0.0000

    pd.options.display.float_format = '{:.4f}'.format

    for i, row in orders_df.iterrows():
        shares = row[symbol]
        if shares != 0:
            if i not in trades_df.index:
                a, b = i, i
                # set the change in shares in the trades dataframe (plus since we are buying)
                while a not in trades_df.index and b not in trades_df.index:
                    a = a + pd.Timedelta(1, unit="d")
                    b = b - pd.Timedelta(1, unit="d")
                if a in trades_df.index:
                    i = a
                elif b in trades_df.index:
                    i = b
            trades_df.loc[i][symbol] = trades_df.loc[i][symbol] + shares
            trades_df.loc[i]["cash"] = trades_df.loc[i]["cash"] + (market_data.loc[i][symbol] * (-shares))

            trades_df.loc[i]["cash"] = trades_df.loc[i]["cash"] - (impact * market_data.loc[i][symbol] * abs(shares))
            # (impact * market_data.loc[i][symbol])
            trades_df.loc[i]["cash"] = trades_df.loc[i]["cash"] - commission  # Commision code!!!!

    # TODO: Make another dataframe (dataframe 3). Also called "holdings_df_3".
    #           Initialize with all zeroes except with starting cash. So cash column row 0 is starting cash
    #       Method: Add trades_df_2[0] to holdings_df_3[0] and forward fill all I think?
    #               Then do that again and again for each row. add forward fill, add forward fill
    holdings_df = market_data.copy()
    holdings_df[holdings_df.columns] = 0
    holdings_df.iloc[0]['cash'] = start_val

    # TODO I might need to sort the dates
    for i in range(len(holdings_df)):
        holdings_df.iloc[i] = holdings_df.iloc[i] + trades_df.iloc[i]
        if i != len(holdings_df)-1:
            holdings_df.iloc[i+1] = holdings_df.iloc[i]

    # TODO: make another dataframe (dataframe 4) called "values_df". This is prices * holdings
    values_df = pd.DataFrame(holdings_df[market_data_to_read] * market_data[market_data_to_read])
    values_df['cash'] = holdings_df['cash']


    # TODO: To get total portfolio value on each day, sum the rows of dataframe 4 "values_df" (sum axis=1)
    portfolio = values_df.sum(axis=1)

    portfolio_df = pd.DataFrame({}, index=values_df.index)
    portfolio_df['portfolio'] = portfolio

    """
    We suggest that you analyze and confirm the following factors: 
        - Plot the price history over the trading period. 
        - Sharpe ratio (Always assume you have 252 trading days in a year. And risk-free rate = 0) of the total portfolio 
            - The Sharpe ratio uses the sample standard deviation.
        - Cumulative return of the total portfolio 
        - Standard deviation of daily returns of the total portfolio 
        - Average daily return of the total portfolio 
        - Ending value of the portfolio 
    """

    # the result (portvals) as a single-column dataframe, containing the value of the portfolio
    # for each trading day in the first column from start_date to end_date, inclusive. As a dataframe

    return portfolio_df


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats

    start_date = portvals.index.min()
    end_date = portvals.index.max()

    """Calculate stats for our portfolio"""
    cumulative_return = (portvals[-1] - portvals[0])
    daily_returns = (portvals / portvals.shift(1)) - 1
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    sharpe_ratio = (math.sqrt(252) * ((avg_daily_ret) / daily_returns.std()))

    """Calculate stats for SPY"""
    # TODO: this is a bit off....
    spy = get_data('', pd.date_range(start_date, end_date))
    spy = spy[spy.columns[0]]
    #cum_ret_SPY = (spy[-1] - spy[0])
    cum_ret_SPY = spy[-1] / spy[0] - 1
    cum_ret_SPY = (spy[-1] - spy[0]) - 1
    spy_daily_returns = (spy / spy.shift(1)) - 1
    avg_daily_ret_SPY = spy_daily_returns.mean()
    std_daily_ret_SPY = spy_daily_returns.std()
    sharpe_ratio_SPY = (math.sqrt(252) * ((avg_daily_ret_SPY) / spy_daily_returns.std()))

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cumulative_return}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")

def author():
  return 'jtyndale3'


def main():
  print('hello world')

if __name__ == "__main__":
    main()

