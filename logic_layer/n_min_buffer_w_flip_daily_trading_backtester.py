import pandas as pd
import math
from datetime import timedelta

from logic_layer.base_class_daily_trading_backtester import BaseClassDailyTradingBacktester


class NMinBufferWFlipDailyTradingBacktester(BaseClassDailyTradingBacktester):

    _N_BUFFER=10
    def __init__(self):
        pass

    def __summarize_trading_positions__(self, result_df, pos_size, net_commissions, n_min):
        """
        Summarizes trading positions using the N_MIN_BUFFER_W_FLIP algorithm.

        Parameters:
        result_df (pd.DataFrame): DataFrame containing trading data with columns ['trading_symbol', 'formatted_date', 'action', 'trading_symbol_price']
        portfolio_size (float): The total size of the portfolio in monetary terms.
        net_commissions (float): The net commissions to be deducted from the profit.
        n_min (int): The number of minutes to wait before flipping positions (N-minute buffer).

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
            current_time = pd.to_datetime(row['formatted_date'])
            current_action = row['action']
            current_price = row['trading_symbol_price']

            # Logic to handle flat positions
            if not position_open and current_action == self._LONG_POS:
                # Start waiting for n_min to confirm the LONG position
                wait_time = current_time + timedelta(minutes=n_min)
                potential_side = self._LONG_POS
                continue

            if not position_open and current_action == self._SHORT_POS:
                # Start waiting for n_min to confirm the SHORT position
                wait_time = current_time + timedelta(minutes=n_min)
                potential_side = self._SHORT_POS
                continue

            # Logic for LONG positions
            if position_open and position_side == self._LONG_POS:
                if current_action == self._SHORT_POS:
                    # Wait N minutes and if the signal remains SHORT, flip the position
                    wait_time = current_time + timedelta(minutes=n_min)
                    if wait_time >= current_time and result_df.iloc[index]['action'] == self._SHORT_POS:
                        position_open = False  # Close LONG position
                        # Logic to close the LONG and open SHORT
                        pass

            # Logic for SHORT positions
            if position_open and position_side == self._SHORT_POS:
                if current_action == self._LONG_POS:
                    # Wait N minutes and if the signal remains LONG, flip the position
                    wait_time = current_time + timedelta(minutes=n_min)
                    if wait_time >= current_time and result_df.iloc[index]['action'] == self._LONG_POS:
                        position_open = False  # Close SHORT position
                        # Logic to close the SHORT and open LONG
                        pass

            # Handle closing positions at the end of the day
            if index == len(result_df) - 1:  # Last row of the dataframe
                if position_open:
                    # Logic to close any open position
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


    #region Public Methods

    def backtest_daily_predictions(self,rnn_predictions_df,portf_size,trade_comm):

        trading_summary_df = self.__summarize_trading_positions__(rnn_predictions_df, portf_size, trade_comm,
                                                                  self._N_BUFFER)

        return self.__calculate_day_trading_summary__(trading_summary_df)
    #endregion