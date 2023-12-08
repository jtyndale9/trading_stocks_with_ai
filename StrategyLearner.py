""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  

Student Name: Joshua Tyndale
GT User ID: jtyndale3
GT ID: 903767547
"""

import datetime as dt
import random
import numpy as np
import pandas as pd
import util as ut
import QLearner as ql
import indicators as indicator


class StrategyLearner(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """
    # constructor  		  	   		  		 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.state_mapping = self.create_state_mapper()


    def create_state_mapper(self):
        all_possible_combinations = []
        for a in range(-1, 2):
            for b in range(-1, 2):
                for c in range(-1, 2):
                    for d in range(-1, 2):
                        #for e in range(-1, 2):
                        all_possible_combinations.append(str(a)+str(b)+str(c)+str(d))
        return pd.DataFrame({'indicator': all_possible_combinations})


    # this method should create a QLearner, and train it for trading
    def add_evidence(self, symbol="IBM", sd=dt.datetime(2008, 1, 1),
                     ed=dt.datetime(2009, 1, 1), sv=10000):
        #print("Training...")

        dates = pd.date_range(sd, ed)
        market_data = ut.get_data([symbol], dates, addSPY=False)
        market_data.dropna(inplace=True)

        """Indicators"""
        macd_signal = indicator.get_macd(symbol, market_data)
        cci_signal = indicator.get_cci(symbol, sd, ed)
        roc_signal = indicator.get_roc(symbol, sd, ed)
        rsi_signal = indicator.get_rsi(symbol, sd, ed)
        #bbp_signal = indicator.get_sma_bollinger_bands(market_data, symbol)

        macd_signal.rename('MACD', inplace=True)
        cci_signal.rename('CCI', inplace=True)
        roc_signal.rename('ROC', inplace=True)
        rsi_signal.rename('RSI', inplace=True)
        #bbp_signal.rename('BBP', inplace=True)
        result = pd.concat([macd_signal, cci_signal, roc_signal, rsi_signal], axis=1)
        #result = pd.concat([macd_signal, cci_signal, roc_signal, rsi_signal], axis=1)

        result['MACD'] = result['MACD'].astype(str)
        result['CCI'] = result['CCI'].astype(str)
        result['ROC'] = result['ROC'].astype(str)
        result['RSI'] = result['RSI'].astype(str)
        #result['BBP'] = result['BBP'].astype(str)

        result['state'] = result.sum(axis=1)


        generator = result.iterrows()
        day_one = generator.__next__()
        current_indicators = day_one[1].state

        # 4 = 81
        # 5 = 243
        self.learner = ql.QLearner(num_states=81, num_actions=3, alpha=0.2, gamma=0.7, rar=0.98, radr=0.999, dyna=10,
                              verbose=False)

        reward=0
        action=1

        clean_market_data = market_data.dropna()
        clean_indicators = clean_market_data.join(result)

        for epoch in range(10):
            first = True
            for df_index, row in clean_indicators.iterrows():
                current_indicators = row.state
                current_state = self.discretize(current_indicators)
                # Actions will be 0 1 or 2. 0 = sell. 1 = do nothing. 2 = buy.
                if first:
                    action = self.learner.querysetstate(current_state)  # get first action
                    first = False # we get a buy, sell, or do nothing
                else:
                    if action == 1: # if do nothing
                        reward = -0.5
                    if action == 0: # if sell
                        reward = self.sell_reward(df_index, clean_indicators, symbol) * 2
                    if action == 2: # if buy
                        reward = self.sell_reward(df_index, clean_indicators, symbol) * -2
                    action = self.learner.query(current_state, reward)

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="IBM", sd=dt.datetime(2009, 1, 1),
                   ed=dt.datetime(2010, 1, 1), sv=10000):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data
        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """

        dates = pd.date_range(sd, ed)
        market_data = ut.get_data([symbol], dates, addSPY=False)
        market_data.dropna(inplace=True)
        macd_signal = indicator.get_macd(symbol, market_data)
        cci_signal = indicator.get_cci(symbol, sd, ed)
        roc_signal = indicator.get_roc(symbol, sd, ed)
        rsi_signal = indicator.get_rsi(symbol, sd, ed)
        #bbp_signal = indicator.get_sma_bollinger_bands(market_data, symbol)

        macd_signal.rename('MACD', inplace=True)
        cci_signal.rename('CCI', inplace=True)
        roc_signal.rename('ROC', inplace=True)
        rsi_signal.rename('RSI', inplace=True)
        #bbp_signal.rename('BBP', inplace=True)

        result = pd.concat([macd_signal, cci_signal, roc_signal, rsi_signal], axis=1)
        result['MACD'] = result['MACD'].astype(str)
        result['CCI'] = result['CCI'].astype(str)
        result['ROC'] = result['ROC'].astype(str)
        result['RSI'] = result['RSI'].astype(str)
        #result['BBP'] = result['BBP'].astype(str)

        result['state'] = result.sum(axis=1)

        clean_market_data = market_data.dropna()
        clean_indicators = clean_market_data.join(result)

        first = True

        actions = []
        for df_index, row in clean_indicators.iterrows():
            current_indicators = row.state
            current_state = self.discretize(current_indicators)
            if first:
                action = self.learner.just_query(current_state)  # get first action
                actions.append(action)
                first = False  # we get a buy, sell, or do nothing
            else:
                if action == 1:  # if do nothing
                    reward = 0
                if action == 0:  # if sell
                    reward = self.sell_reward(df_index, clean_indicators, symbol)
                if action == 2:  # if buy
                    reward = self.sell_reward(df_index, clean_indicators, symbol) * -1
                action = self.learner.just_query(current_state)
                actions.append(action)

        clean_indicators['action'] = actions

        clean_indicators['action_words'] = ['BUY' if i == 2 else 'SELL' if i == 0 else "no action" for i in clean_indicators['action']]
        #trade_index = [index for index, row in clean_indicators.iterrows() if row['action'] != 1]
        #orders = [row['action_words'] for index, row in clean_indicators.iterrows() if row['action_words'] != 'no action']
        trade_index = [index for index, row in clean_indicators.iterrows()]
        orders = [row['action_words'] for index, row in clean_indicators.iterrows()]
        symbols = [symbol] * len(orders)
        shares = [2000] * len(orders)
        shares[0] = 1000

        # 'Symbol': symbols,
        manual_strategy_trades = pd.DataFrame({symbol: orders},
                                              index=trade_index)
        #print(manual_strategy_trades.shape)
        #print(len(manual_strategy_trades))
        #sys.exit()

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



    def discretize(self, indicator):
        """
        convert the stuff to a single integer
        :param pos: the position to discretize. type int
        :return: the discretized position. rtype: int
        """
        state = self.state_mapping.loc[self.state_mapping['indicator'] == indicator].index[0]
        return state



    def sell_reward(self, df_index, market_data, symbol):
        yesterday = df_index - pd.Timedelta(1, unit="d")
        while True:
            try:
                yesterday_value = market_data.loc[yesterday][symbol]
            except:
                yesterday = yesterday - pd.Timedelta(1, unit="d")
            else:
                break
        today_value = market_data.loc[df_index][symbol]
        returns = yesterday_value - today_value
        return returns


    def author(self):
        return "jtyndale3"


if __name__ == "__main__":
    print("One does not simply think up a strategy")

