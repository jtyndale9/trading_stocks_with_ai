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

    """Train Strategy Model"""
    strategy_lerner = sl.StrategyLearner(verbose=False, impact=0.0, commission=0.0)
    strategy_lerner.add_evidence(symbol=this_symbol, sd=in_sample_sd, ed=in_sample_ed,sv=100000)

    """In Sample Model Strategy IMPACT 0"""
    strategy_trades_in_sample = strategy_lerner.testPolicy(symbol=this_symbol,sd=in_sample_sd,
                                                           ed=in_sample_ed, sv=100000)
    strategy_one = compute_portvals(orders_df=strategy_trades_in_sample, start_val=100000, commission=0,impact=0)

    """In Sample Model Strategy IMPACT 0.005"""
    strategy_two = compute_portvals(orders_df=strategy_trades_in_sample, start_val=100000, commission=0,impact=0.005)

    """In Sample Model Strategy IMPACT 0"""
    strategy_three = compute_portvals(orders_df=strategy_trades_in_sample, start_val=100000, commission=0,impact=0.01)

    """In Sample Model Strategy IMPACT 0"""
    strategy_four = compute_portvals(orders_df=strategy_trades_in_sample, start_val=100000, commission=0,impact=0.001)

    """Plot"""
    plot(benchmark_df_in_sample, strategy_one, strategy_two,
         strategy_three, strategy_four, this_symbol,
         "Experiment_2.png")


def plot(benchmark_df, strategy_one, strategy_two, strategy_three, strategy_four,
         this_symbol, filename):
    fig, ax = matplotlib.pyplot.subplots()

    benchmark_normalized = benchmark_df.portfolio / benchmark_df.portfolio[0]
    strategy_one_normalized = strategy_one.portfolio / strategy_one.portfolio[0]
    strategy_two_normalized = strategy_two.portfolio / strategy_two.portfolio[0]
    strategy_three_normalized = strategy_three.portfolio / strategy_three.portfolio[0]
    strategy_four_normalized = strategy_four.portfolio / strategy_four.portfolio[0]

    plt.figure(figsize=(14, 8))
    plt.plot(benchmark_df.index, benchmark_normalized, color='purple', label='Benchmark')
    plt.plot(strategy_one.index, strategy_one_normalized, color='red', label='0')
    plt.plot(strategy_two.index, strategy_two_normalized, color='green', label='0.005')
    plt.plot(strategy_three.index, strategy_three_normalized, color='orange', label='0.01')
    plt.plot(strategy_four.index, strategy_four_normalized, color='blue', label='0.001')

    plt.legend()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()

    plt.xlabel("Dates")
    plt.ylabel("Portfolio Value (Normalized)")
    plt.title(f"Portfolio Values for {this_symbol} with different impact")
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
