import math

import pandas as pd
import numpy as np

from logic_layer.base_class_daily_trading_backtester import BaseClassDailyTradingBacktester


class RawAlgoDailyTradingBacktester(BaseClassDailyTradingBacktester):

    def __init__(self):
        pass


    #region Private Methods

    def __summarize_trading_positions__(self, result_df, portfolio_size, net_commissions):
        """
        Summarizes trading positions from the result_df DataFrame.

        Parameters:
        result_df (pd.DataFrame): DataFrame containing trading data with columns ['trading_symbol', 'formatted_date', 'action', 'trading_symbol_price']
        portfolio_size (float): The total size of the portfolio in monetary terms.
        net_commissions (float): The net commissions to be deducted from the profit.

        Returns:
        pd.DataFrame: DataFrame summarizing the trading positions with columns ['symbol', 'open', 'close', 'side', 'price_open', 'price_close', 'unit_gross_profit', 'total_gross_profit', 'total_net_profit', 'portfolio_size', 'pos_size']
        """

        # Initialize an empty DataFrame to store the trading summary
        trading_summary_df = pd.DataFrame(columns=[
            'symbol', 'open', 'close', 'side', 'price_open', 'price_close', 'unit_gross_profit', 'total_gross_profit',
            'total_net_profit', 'portfolio_size', 'pos_size'
        ])

        # Variables to keep track of the open position
        position_open = False
        position_side = None
        entry_price = None
        entry_time = None
        entry_symbol = None

        # Iterate through each row in the result_df DataFrame
        for index, row in result_df.iterrows():
            current_symbol = row['trading_symbol']
            current_time = row['formatted_date']
            current_action = row['action']
            current_price = row['trading_symbol_price']

            # If a new position is opened
            if (not position_open) and (current_action in [self._LONG_POS,self._SHORT_POS]):
                # Calculate position size
                pos_size = math.floor(portfolio_size / current_price)

                # Open a new position
                position_open = True
                position_side = current_action
                entry_price = current_price
                entry_time = current_time
                entry_symbol = current_symbol

                # Create a new record in trading_summary_df for the opened position
                new_row = pd.DataFrame({
                    'symbol': [entry_symbol],
                    'open': [entry_time],
                    'close': [None],
                    'side': [position_side],
                    'price_open': [entry_price],
                    'price_close': [None],
                    'unit_gross_profit': [None],
                    'total_gross_profit': [None],
                    'total_net_profit': [None],
                    'portfolio_size': [portfolio_size],
                    'pos_size': [pos_size]
                })

                trading_summary_df = pd.concat([trading_summary_df, new_row], ignore_index=True)

            # If an open position is closed
            elif position_open and (
                    (position_side == self._LONG_POS and current_action in [self._SHORTw__POS, self._FLAT_POS]) or
                    (position_side == self._SHORT_POS and current_action in [self._LONG_POS, self._FLAT_POS])
            ):
                # Calculate unit gross profit
                unit_gross_profit = current_price - entry_price if position_side == self._LONG_POS else entry_price - current_price
                total_gross_profit = unit_gross_profit * pos_size
                total_net_profit = total_gross_profit - net_commissions

                # Update the record in trading_summary_df with the closing details
                trading_summary_df.at[trading_summary_df.index[-1], 'close'] = current_time
                trading_summary_df.at[trading_summary_df.index[-1], 'price_close'] = current_price
                trading_summary_df.at[trading_summary_df.index[-1], 'unit_gross_profit'] = unit_gross_profit
                trading_summary_df.at[trading_summary_df.index[-1], 'total_gross_profit'] = total_gross_profit
                trading_summary_df.at[trading_summary_df.index[-1], 'total_net_profit'] = total_net_profit

                # Close the position
                position_open = False
                position_side = None
                entry_price = None
                entry_time = None
                entry_symbol = None

        # Handle the case where the last row is still an open position
        if position_open:
            # Use the last available time in result_df for closing
            last_time = result_df.iloc[-1]['formatted_date']
            last_price = result_df.iloc[-1]['trading_symbol_price']

            unit_gross_profit = last_price - entry_price if position_side == self._LONG_POS else entry_price - last_price
            total_gross_profit = unit_gross_profit * pos_size
            total_net_profit = total_gross_profit - net_commissions

            # Update the last position to close it
            trading_summary_df.at[trading_summary_df.index[-1], 'close'] = last_time
            trading_summary_df.at[trading_summary_df.index[-1], 'price_close'] = last_price
            trading_summary_df.at[trading_summary_df.index[-1], 'unit_gross_profit'] = unit_gross_profit
            trading_summary_df.at[trading_summary_df.index[-1], 'total_gross_profit'] = total_gross_profit
            trading_summary_df.at[trading_summary_df.index[-1], 'total_net_profit'] = total_net_profit

        # Return the trading summary DataFrame
        return trading_summary_df



    #endregion


    #region Public Methods

    def backtest_daily_predictions(self,rnn_predictions_df,portf_size,trade_comm):

        trading_summary_df = self.__summarize_trading_positions__(rnn_predictions_df, portf_size, trade_comm)

        return self.__calculate_day_trading_summary__(trading_summary_df)


    import numpy as np



    #endregion