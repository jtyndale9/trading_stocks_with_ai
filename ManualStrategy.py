# Build a Manual Strategy, implemented as a class, that combines a minimum of 3 out of the 5 indicators from Project 6

"""
Code implementing a ManualStrategy object (your Manual Strategy) in the strategy_evaluation/ directory.
It should implement testPolicy() which returns a trades data frame (see below).
The main part of this code should call marketsimcode as necessary to generate the plots used in the report.
NOTE: You will have to create this file yourself.
"""

# TODO implement testPolicy() which returns a trades data frame

# TODO main part of this code should call marketsimcode as necessary to generate the plots used in the report


# TODO For your report, trade only the symbol JPM
# TODO The in-sample period is January 1, 2008 to December 31, 2009.
# TODO The out-of-sample/testing period is January 1, 2010 to December 31, 2011.
# TODO Starting cash is $100,000.

# TODO Allowable positions are: 1000 shares long, 1000 shares short, 0 shares.

# TODO Benchmark: The performance of a portfolio starting with $100,000 cash,
#  investing in 1000 shares of the symbol in use on the first trading day,
#  and holding that position. Include transaction costs.

# TODO There is no limit on leverage. This means that you do not need to confirm that you have the capital to make your trade
#  All trades can be executed without validating available cash in your portfolio.

# TODO Transaction costs:
#  ManualStrategy and StrategyLearner: Commission: $9.95, Impact: 0.005 (unless stated otherwise in an experiment).
#  Auto-Grader Commission will always be $0.00, Impact may vary, and will be passed in as a parameter to the learner

# TODO  All indicators must be used in some way to determine a buy/sell signal.
#  You cannot use a single indicator for all signals.


"""
df_trades = ms.testPolicy(symbol = "JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000) 
"""
import datetime as dt
import random
import util as ut
import pandas as pd
import numpy as np
from marketsimcode import compute_portvals
import indicators as indicator

class ManualStrategy(object):


    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    # this method should create a QLearner, and train it for trading
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 1, 1), sv=10000):

        dates = pd.date_range(sd, ed)
        market_data = ut.get_data([symbol], dates, addSPY=False)
        market_data.dropna(inplace=True)

        """Indicators!"""
        macd_signal = indicator.get_macd(symbol, market_data)
        cci_signal = indicator.get_cci(symbol, sd, ed)
        roc_signal = indicator.get_roc(symbol, sd, ed)
        rsi_signal = indicator.get_rsi(symbol, sd, ed)
        bbp_signal = indicator.get_sma_bollinger_bands(market_data, symbol)

        macd_signal.rename('MACD', inplace=True)
        cci_signal.rename('CCI', inplace=True)
        roc_signal.rename('ROC', inplace=True)
        rsi_signal.rename('RSI', inplace=True)
        bbp_signal.rename('BBP', inplace=True)
        result = pd.concat([macd_signal, cci_signal, roc_signal, rsi_signal, bbp_signal], axis=1)
        result['vote'] = result.sum(axis=1)

        result['action'] = ['BUY' if i > 0 else 'SELL' if i < 0 else "no action" for i in result['vote']]
        trade_index = [index for index,row in result.iterrows()]
        orders = [row['action'] for index,row in result.iterrows()]
        # Allowable positions are: 1000 shares long, 1000 shares short, 0 shares.
        # TODO How do I make the shares array? Do I always trade 1000 shares?
        #  what if I have 1000 shares already so I need to sell 2000? What if I want to do nothing?
        #  should I keep the do nothing in the dataframe so I can make sure my shares are at 0?
        #  didnt the rubric say I have unlimited money though?... idk
        manual_strategy_trades = pd.DataFrame({symbol: orders},
                                           index=trade_index)

        position = 0
        orders_array = []

        for i, row in manual_strategy_trades.iterrows():
            if row[symbol] == 'BUY':
                if position == 0:
                    orders_array.append(1000)
                    position = 1000
                elif position == -1000:
                    orders_array.append(2000)
                    position = 1000
                elif position == 1000:
                    orders_array.append(0)
                    position = 1000
            elif row[symbol] == 'SELL':
                if position == 0:
                    orders_array.append(-1000)
                    position = -1000
                elif position == -1000:
                    orders_array.append(0)
                    position = -1000
                elif position == 1000:
                    orders_array.append(-2000)
                    position = -1000
            elif row[symbol] == 'no action':
                #orders_array.append(0)
                if position == 0:
                    orders_array.append(0)
                    position = 0
                elif position == -1000:
                    orders_array.append(1000)
                    position = 0
                elif position == 1000:
                    orders_array.append(-1000)
                    position = 0

        manual_strategy_trades = pd.DataFrame({symbol: orders_array},
                                              index=trade_index)
        return manual_strategy_trades


    def author(self):
        return "jtyndale3"

# TODO The manual strategy should have the same api as the strategy learner,
#  but you do not need to implement the add_evidence function.
