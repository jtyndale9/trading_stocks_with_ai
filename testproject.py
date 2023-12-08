"""
Code initializing/running all necessary files for the report.
NOTE: You will have to create the contents of this file yourself
"""

"""
Create testproject.py. Testproject.py is the entry point to your project, and it should implement the necessary calls 
(following each respective API) to Manual Strategy.py, StrategyLearner.py, experiment1.py, and experiment2.py with the 
appropriate parameters to run everything needed for the report in a single Python call: 

PYTHONPATH=../:. python testproject.py 
"""

# this is the entry point file!!
# initialize learner, pull in data, send everything over, run strategy, run experiments, create graphs...

"""
SIMPLE MOVING AVERAGE (SMA) AND BOLLINGER BANDS
RATE OF CHANGE (ROC)
RSI (Relative Strength Index)
CCI COMMODITY CHANNEL INDEX 
MACD MOVING AVERAGE CONVERGENCE DIVERGENCE 
"""

import experiment1
import experiment2
import ManualStrategy as manual
import StrategyLearner as sl

from marketsimcode import compute_portvals
import time
import pandas as pd
import numpy as np
import math, statistics
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
import random
import util as ut

# TODO For your report, trade only the symbol JPM
# TODO The in-sample period is January 1, 2008 to December 31, 2009.
# TODO The out-of-sample/testing period is January 1, 2010 to December 31, 2011.
# TODO Starting cash is $100,000.
start = time.time()
def main():

    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)
    out_of_sample_sd = dt.datetime(2010, 1, 1)
    out_of_sample_ed = dt.datetime(2011, 12, 31)

    this_symbol ="JPM"

    manual_learning = True
    model_learning = False
    verbose = True

    if manual_learning:
        """Get Benchmark Performance"""
        benchmark_df_in_sample = benchmark(in_sample_sd, in_sample_ed, this_symbol)
        benchmark_df_out_of_sample = benchmark(out_of_sample_sd, out_of_sample_ed, this_symbol)
        #plot(benchmark_df_in_sample, "benchmark.png")

        # TODO call manual strategy, and get the returned DF strategy, and run market sim on it to get porfolio values, then plot
        """In Sample"""
        ms = manual.ManualStrategy()
        manual_strategy_trades = ms.testPolicy(symbol=this_symbol, sd=in_sample_sd, ed=in_sample_ed, sv=100000)
        portfolio_val_in_sample = compute_portvals(orders_df=manual_strategy_trades, start_val=100000, commission=0, impact=0)
        #plot(portfolio_val_in_sample, "portfolio_val.png")
        manual_strategy_plot(benchmark_df_in_sample, manual_strategy_trades, portfolio_val_in_sample, this_symbol,
                             "In_sample_manual_strategy.png", 'In Sample')

        """Out of Sample"""
        ms = manual.ManualStrategy()
        manual_strategy_trades = ms.testPolicy(symbol=this_symbol, sd=out_of_sample_sd, ed=out_of_sample_ed, sv=100000)
        portfolio_val_out_of_sample = compute_portvals(orders_df=manual_strategy_trades, start_val=100000, commission=0, impact=0)
        #plot(portfolio_val_out_of_sample, "portfolio_val.png")
        manual_strategy_plot(benchmark_df_out_of_sample, manual_strategy_trades, portfolio_val_out_of_sample, this_symbol,
                             "Out_of_sample_manual_strategy.png", 'Out of Sample')

        if verbose:
            print('IN SAMPLE BENCHMARK')
            get_table_calcs(benchmark_df_in_sample.portfolio)
            print('IN SAMPLE STRATEGY')
            get_table_calcs(portfolio_val_in_sample.portfolio)
            print()
            print()
            print("OUT OF SAMPLE BENCHMARK")
            get_table_calcs(benchmark_df_out_of_sample.portfolio)
            print("OUT OF SAMPLE STRATEGY")
            get_table_calcs(portfolio_val_out_of_sample.portfolio)


    if model_learning:
        # TODO call strategy learner, and get the returned DF
        strategy_lerner = sl.StrategyLearner(verbose=False, impact=0.0, commission=0.0)
        strategy_lerner.add_evidence(symbol=this_symbol, sd=in_sample_sd, ed=in_sample_ed,sv=100000) # training phase

        strategy_trades_in_sample = strategy_lerner.testPolicy(symbol=this_symbol,sd=in_sample_sd, ed=in_sample_ed,
                                                               sv=100000)
        strat_portfolio_val_in_sample = compute_portvals(orders_df=strategy_trades_in_sample, start_val=100000, commission=0,
                                                   impact=0)
        manual_strategy_plot(benchmark_df_in_sample, strategy_trades_in_sample, strat_portfolio_val_in_sample,
                             this_symbol, "QLEARNING_in_sample.png", 'In Sample')

        strategy_trades_out_of_sample = strategy_lerner.testPolicy(symbol=this_symbol,sd=out_of_sample_sd,
                                                                   ed=out_of_sample_ed, sv=100000)
        strat_portfolio_val_out_of_sample = compute_portvals(orders_df=strategy_trades_out_of_sample, start_val=100000,
                                                         commission=0, impact=0)
        manual_strategy_plot(benchmark_df_out_of_sample, strategy_trades_out_of_sample, strat_portfolio_val_out_of_sample,
                             this_symbol, "QLEARNING_out_of_sample.png", 'Out of Sample')

        if verbose:
            print("IN SAMPLE STRATEGY")
            get_table_calcs(strat_portfolio_val_in_sample.portfolio)
            print("OUT OF SAMPLE STRATEGY")
            get_table_calcs(strat_portfolio_val_out_of_sample.portfolio)


    experiment1.main()
    experiment2.main()
    end = time.time()
    print(end - start)


def benchmark(sd, ed, symbol):
    benchmark_trades_df = pd.DataFrame({symbol: [1000, 0]},
                                       index=[sd, ed])

    # call marketism's compute_portvals with the dataframe
    benchmark_returns = compute_portvals(orders_df=benchmark_trades_df, start_val=100000, commission=0, impact=0)
    return benchmark_returns


# TODO CREATE THE CHARTS IN THIS FILE
def plot(df, name):
    ax = df.plot.line()
    ax.figure.savefig(name)


def manual_strategy_plot(benchmark_df, manual_strategy_trades, portfolio_val, this_symbol, filename, title):
    # Benchmark purple line
    # Performance of manual strat red line
    # normalized to 1.0
    # blue for long entry
    # black for short entry

    fig, ax = matplotlib.pyplot.subplots()
    benchmark_normalized = benchmark_df.portfolio / benchmark_df.portfolio[0]
    portfolio_val_normalized = portfolio_val.portfolio / portfolio_val.portfolio[0]
    trades = manual_strategy_trades[manual_strategy_trades[this_symbol] != manual_strategy_trades[this_symbol].shift()]

    plt.figure(figsize=(14, 8))
    plt.plot(benchmark_df.index, benchmark_normalized, color='purple', label="Benchmark")
    plt.plot(portfolio_val.index, portfolio_val_normalized, color='red', label="Manual Strategy")
    plt.legend()

    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()
    #print(portfolio_val)

    for i,position in trades.iterrows():
        if position[this_symbol] > 0:
            y_min = portfolio_val_normalized.iloc[portfolio_val_normalized.index.get_loc(i,method='backfill')] - 0.015
            y_max = portfolio_val_normalized.iloc[portfolio_val_normalized.index.get_loc(i,method='backfill')] + 0.015
            plt.vlines(i, ymin=y_min, ymax=y_max, color='blue', linestyle='-', label='BUY')
        elif position[this_symbol] < 0 :
            y_min = portfolio_val_normalized.iloc[portfolio_val_normalized.index.get_loc(i, method='backfill')] - 0.015
            y_max = portfolio_val_normalized.iloc[portfolio_val_normalized.index.get_loc(i, method='backfill')] + 0.015
            plt.vlines(i, ymin=y_min, ymax=y_max, color='black', linestyle='-', label='SELL')

    plt.xlabel("Dates")
    plt.ylabel("Portfolio Value (Normalized)")
    plt.title(f"Normalized Portfolio Values for {this_symbol} - ({title})")
    plt.savefig(filename)  # save as png



def get_table_calcs(returns):
    calculations = {}

    start_date = returns.index.min()
    end_date = returns.index.max()

    """Calculate stats for our portfolio"""
    calculations['cumulative_return'] = (returns[-1] - returns[0])
    calculations['daily_returns'] = (returns / returns.shift(1)) - 1
    calculations['avg_daily_ret'] = calculations['daily_returns'].mean()
    calculations['std_daily_ret'] = calculations['daily_returns'].std()
    calculations['sharpe_ratio'] = (math.sqrt(252) * ((calculations['avg_daily_ret']) / calculations['daily_returns'].std()))

    print(f"Date Range: {start_date} to {end_date}")
    print(f"Sharpe Ratio of Fund: {calculations['sharpe_ratio']}")
    print(f"Cumulative Return of Fund: {calculations['cumulative_return']}")
    print(f"Standard Deviation of Fund: {calculations['std_daily_ret']}")
    print(f"Average Daily Return of Fund: {calculations['avg_daily_ret']}")
    print(f"Final Portfolio Value: {returns[-1]}")
    print()
    print()
    return calculations


def author(self):
    return "jtyndale3"

if __name__ == "__main__":
    main()
