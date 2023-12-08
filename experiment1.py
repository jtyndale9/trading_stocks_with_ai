import ManualStrategy as manual
import StrategyLearner as sl
from marketsimcode import compute_portvals

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt


def main():
    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)
    out_of_sample_sd = dt.datetime(2010, 1, 1)
    out_of_sample_ed = dt.datetime(2011, 12, 31)

    this_symbol ="JPM"


    """In Sample Benchmark"""
    benchmark_df_in_sample = benchmark(in_sample_sd, in_sample_ed, this_symbol)
    """Out of Sample Benchmark"""
    benchmark_df_out_of_sample = benchmark(out_of_sample_sd, out_of_sample_ed, this_symbol)


    """In Sample Manual Strategy"""
    ms = manual.ManualStrategy()
    manual_strategy_trades_in_sample = ms.testPolicy(symbol=this_symbol, sd=in_sample_sd, ed=in_sample_ed, sv=100000)
    portfolio_manual_in_sample = compute_portvals(orders_df=manual_strategy_trades_in_sample, start_val=100000, commission=0,
                                               impact=0)


    #print('\n\n\n')
    """Out of Sample Manual Strategy"""
    ms = manual.ManualStrategy()
    manual_strategy_trades_out_of_sample = ms.testPolicy(symbol=this_symbol, sd=out_of_sample_sd, ed=out_of_sample_ed, sv=100000)
    portfolio_manual_out_of_sample = compute_portvals(orders_df=manual_strategy_trades_out_of_sample, start_val=100000, commission=0,
                                                   impact=0)

    #print(manual_strategy_trades_out_of_sample.shape)
    #print(manual_strategy_trades_out_of_sample.head())
    #print(manual_strategy_trades_out_of_sample.tail())


    """Train Strategy Model"""
    strategy_lerner = sl.StrategyLearner(verbose=False, impact=0.0, commission=0.0)
    strategy_lerner.add_evidence(symbol=this_symbol, sd=in_sample_sd, ed=in_sample_ed,sv=100000) # training phase


    """In Sample Model Strategy"""
    strategy_trades_in_sample = strategy_lerner.testPolicy(symbol=this_symbol,
                                           sd=in_sample_sd, ed=in_sample_ed, sv=100000)  # testing phase
    strat_portfolio_val_in_sample = compute_portvals(orders_df=strategy_trades_in_sample, start_val=100000, commission=0,
                                               impact=0)
    #print(strategy_trades_in_sample.shape)


    """Out of Sample Model Strategy"""
    strategy_trades_out_of_sample = strategy_lerner.testPolicy(symbol=this_symbol,
                                                           sd=out_of_sample_sd, ed=out_of_sample_ed, sv=100000)
    strat_portfolio_val_out_of_sample = compute_portvals(orders_df=strategy_trades_out_of_sample, start_val=100000,
                                                     commission=0,impact=0)
    #print(strategy_trades_out_of_sample.shape)



    """Plot In Sample"""
    plot(benchmark_df_in_sample, manual_strategy_trades_in_sample, portfolio_manual_in_sample,
         strategy_trades_in_sample, strat_portfolio_val_in_sample, this_symbol,
         "In_Sample_Experiment_1.png", 0.025, "In Sample")

    """Plot Out of Sample"""
    plot(benchmark_df_out_of_sample, manual_strategy_trades_out_of_sample, portfolio_manual_out_of_sample,
         strategy_trades_out_of_sample, strat_portfolio_val_out_of_sample, this_symbol,
         "Out_of_Sample_Experiment_1.png", 0.01, "Out of Sample")


def plot(benchmark_df, manual_strategy_trades, manual_portfolio_val, strategy_trades, strategy_portfolio,
         this_symbol, filename, length, title):
    fig, ax = matplotlib.pyplot.subplots()

    benchmark_normalized = benchmark_df.portfolio / benchmark_df.portfolio[0]
    manual_portfolio_val_normalized = manual_portfolio_val.portfolio / manual_portfolio_val.portfolio[0]
    model_portfolio_val_normalized = strategy_portfolio.portfolio / strategy_portfolio.portfolio[0]

    manual_trades = manual_strategy_trades[manual_strategy_trades[this_symbol] != manual_strategy_trades[this_symbol].shift()]
    model_trades = strategy_trades[strategy_trades[this_symbol] != strategy_trades[this_symbol].shift()]

    plt.figure(figsize=(14, 8))
    plt.plot(benchmark_df.index, benchmark_normalized, color='purple', label='Benchmark')
    plt.plot(manual_portfolio_val.index, manual_portfolio_val_normalized, color='red', label='Manual Trading')
    plt.plot(strategy_portfolio.index, model_portfolio_val_normalized, color='green', label='Q-Learning')
    #first_legend = plt.legend(handles=[line1], loc=1)
    #ax = plt.gca().add_artist(first_legend)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()

    """Plot Manual Trade indicators"""
    for i,position in manual_trades.iterrows():
        if position[this_symbol] == 'BUY':
            y_min = manual_portfolio_val_normalized.iloc[manual_portfolio_val_normalized.index.get_loc(i,method='backfill')] - length
            y_max = manual_portfolio_val_normalized.iloc[manual_portfolio_val_normalized.index.get_loc(i,method='backfill')] + length
            plt.vlines(i, ymin=y_min, ymax=y_max, color='blue', linestyle='-', label='BUY')
        elif position[this_symbol] == 'SELL':
            y_min = manual_portfolio_val_normalized.iloc[manual_portfolio_val_normalized.index.get_loc(i, method='backfill')] - length
            y_max = manual_portfolio_val_normalized.iloc[manual_portfolio_val_normalized.index.get_loc(i, method='backfill')] + length
            plt.vlines(i, ymin=y_min, ymax=y_max, color='black', linestyle='-', label='SELL')

    """Plot Model based trade indicators"""
    for i, position in model_trades.iterrows():
        if position[this_symbol] > 0:
            y_min = model_portfolio_val_normalized.iloc[
                        model_portfolio_val_normalized.index.get_loc(i, method='backfill')] - length
            y_max = model_portfolio_val_normalized.iloc[
                        model_portfolio_val_normalized.index.get_loc(i, method='backfill')] + length
            plt.vlines(i, ymin=y_min, ymax=y_max, color='blue', linestyle='-', label='BUY')
        elif position[this_symbol] < 0:
            y_min = model_portfolio_val_normalized.iloc[
                        model_portfolio_val_normalized.index.get_loc(i, method='backfill')] - length
            y_max = model_portfolio_val_normalized.iloc[
                        model_portfolio_val_normalized.index.get_loc(i, method='backfill')] + length
            plt.vlines(i, ymin=y_min, ymax=y_max, color='black', linestyle='-', label='SELL')


    #legend2 = plt.legend([1,2], ["algo1", "algo2"], loc=1)
    #plt.gca().add_artist(legend2)
    #red_patch = mpatches.Patch(color='red', label='The red data')
    #plt.legend(handles=[red_patch])

    plt.xlabel("Dates")
    plt.ylabel("Portfolio Value (Normalized)")
    plt.title(f"Normalized Portfolio Values for {this_symbol} ({title})")
    plt.savefig(filename)

def benchmark(sd, ed, symbol):
    benchmark_trades_df = pd.DataFrame({symbol: [1000, -1000]},
                                       index=[sd, ed])
    benchmark_returns = compute_portvals(orders_df=benchmark_trades_df, start_val=100000, commission=0, impact=0)
    return benchmark_returns

def author(self):
    return "jtyndale3"

if __name__ == "__main__":
    pass
    #main()
