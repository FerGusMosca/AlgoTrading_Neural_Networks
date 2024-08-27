import math

import pandas as pd
import numpy as np
class DailyTradingBacktester:

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
            if (not position_open) and (current_action in ['LONG', 'SHORT']):
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
                    (position_side == 'LONG' and current_action in ['SHORT', 'FLAT']) or
                    (position_side == 'SHORT' and current_action in ['LONG', 'FLAT'])
            ):
                # Calculate unit gross profit
                unit_gross_profit = current_price - entry_price if position_side == 'LONG' else entry_price - current_price
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

            unit_gross_profit = last_price - entry_price if position_side == 'LONG' else entry_price - last_price
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

    def __calculate_day_trading_summary__(self, trading_summary_df):
        """
        This method calculates the daily trading summary including:
        - Total net profit for the day.
        - Total number of positions.
        - Maximum drawdown, defined as the maximum cumulative loss during the day.

        Parameters:
        trading_summary_df (pd.DataFrame): A DataFrame with trading positions containing the following columns:
                                           'close', 'price_close', 'unit_gross_profit', 'total_gross_profit',
                                           'total_net_profit'

        Returns:
        daily_net_profit (float): The sum of all 'total_net_profit' values for the day.
        total_positions (int): The number of positions closed in the day.
        max_cum_drawdown (float): The maximum drawdown defined as the maximum cumulative loss during the day.
        """
        # 1. Sum all values in the 'total_net_profit' column
        daily_net_profit = trading_summary_df['total_net_profit'].sum()

        # 2. Count the total number of positions (number of rows in the DataFrame)
        total_positions = len(trading_summary_df)

        # 3. Calculate the maximum drawdown as the maximum cumulative loss during the day
        max_cum_drawdown = 0
        current_drawdown = 0

        for profit in trading_summary_df['total_net_profit']:
            if profit < 0:
                current_drawdown += profit
                max_cum_drawdown = min(max_cum_drawdown, current_drawdown)
            else:
                current_drawdown = 0  # Reset the drawdown when there's a profit

        return daily_net_profit, total_positions, max_cum_drawdown, trading_summary_df

    #endregion


    #region Public Methods


    def backtest_daily_predictions(self,rnn_predictions_df,portf_size,trade_comm):

        trading_summary_df = self.__summarize_trading_positions__(rnn_predictions_df, portf_size, trade_comm)

        return self.__calculate_day_trading_summary__(trading_summary_df)

    import numpy as np

    def calculate_max_total_drawdown(self, daily_profits):
        """
        This function calculates the maximum drawdown over a period based on daily profits.

        Parameters:
        daily_profits (list or np.array): Array containing daily profits (positive or negative).

        Returns:
        float: The maximum drawdown over the period.
        """
        # Convert daily profits to a numpy array if it's not already
        daily_profits = np.array(daily_profits)

        # Initialize variables to track the maximum drawdown
        max_drawdown = 0
        cumulative_drawdown = 0

        # Loop through daily profits to calculate the maximum drawdown
        for profit in daily_profits:
            cumulative_drawdown += profit
            if cumulative_drawdown > 0:
                cumulative_drawdown = 0
            if cumulative_drawdown < max_drawdown:
                max_drawdown = cumulative_drawdown

        return -1* abs(max_drawdown)

    #endregion