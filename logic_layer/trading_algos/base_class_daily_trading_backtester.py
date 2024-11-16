import math
import pandas as pd
import numpy as np
class BaseClassDailyTradingBacktester:


    _LONG_POS="LONG"
    _SHORT_POS="SHORT"
    _FLAT_POS="FLAT"
    def __init__(self):
        pass


    def __initialize_dataframe__(self):

        # Initialize an empty list to store the rows of the trading summary
        trading_summary_df = pd.DataFrame(columns=[
            'symbol', 'open', 'close', 'side', 'price_open', 'price_close', 'unit_gross_profit', 'total_gross_profit',
            'total_net_profit', 'portfolio_size', 'pos_size'
        ])

        return  trading_summary_df

    def __append_position_row__(self,entry_symbol,entry_time,position_side,current_time,current_price,entry_price,
                           pos_size,unit_gross_profit,total_gross_profit,total_net_profit,summary_rows):

        new_row = pd.DataFrame({
            'symbol': [entry_symbol],
            'open': [entry_time],
            'close': [current_time],
            'side': [position_side],
            'price_open': [entry_price],
            'price_close': [current_price],
            'unit_gross_profit': [unit_gross_profit],
            'total_gross_profit': [total_gross_profit],
            'total_net_profit': [total_net_profit],
            'portfolio_size': [pos_size],
            'pos_size': [pos_size]
        })

        trading_summary_df = pd.concat([summary_rows, new_row], ignore_index=True)
        return  trading_summary_df


    def __update_position_row__(self,entry_symbol,entry_time,position_side,current_time,current_price,
                                pos_size,unit_gross_profit,total_gross_profit,total_net_profit,summary_rows):

        # Filter the row that matches entry_symbol, entry_time, and position_side
        row_mask = (summary_rows['symbol'] == entry_symbol) & \
                   (summary_rows['open'] == entry_time) & \
                   (summary_rows['side'] == position_side)

        # Check if at least one matching row is found
        if not row_mask.any():
            raise ValueError("No row found with the provided values for symbol, open, and side.")

        # Update the matching row(s) with the new values
        summary_rows.loc[row_mask, 'close'] = current_time
        summary_rows.loc[row_mask, 'price_close'] = current_price
        summary_rows.loc[row_mask, 'unit_gross_profit'] = unit_gross_profit
        summary_rows.loc[row_mask, 'total_gross_profit'] = total_gross_profit
        summary_rows.loc[row_mask, 'total_net_profit'] = total_net_profit
        summary_rows.loc[row_mask, 'portfolio_size'] = pos_size
        summary_rows.loc[row_mask, 'pos_size'] = pos_size

        # Return the updated DataFrame (optional, since `summary_rows` is modified in place)
        return summary_rows


    def __close_final_position__(self,entry_symbol,entry_time,position_side,current_time,current_price,entry_price,
                           portf_size,net_commissions,summary_rows):

        pos_size = int(portf_size / entry_price)
        last_time = current_time
        last_price = current_price

        if position_side == self._LONG_POS:
            unit_gross_profit = last_price - entry_price
        else:
            unit_gross_profit = entry_price - last_price
        total_gross_profit = unit_gross_profit * pos_size
        total_net_profit = total_gross_profit - net_commissions


        return  self.__update_position_row__(entry_symbol,entry_time,position_side,current_time,current_price,entry_price,
                                             pos_size,unit_gross_profit,total_gross_profit,total_net_profit,summary_rows)

    def __close_portf_position__(self,portf_pos,current_time,current_price,portf_size,net_commissions,summary_rows):
        portf_pos.date_close=current_time
        portf_pos.price_close=current_price

        pos_size = int(portf_size / portf_pos.price_open)
        last_time = current_time
        last_price = current_price

        if portf_pos.side==self._LONG_POS:
            unit_gross_profit = last_price - portf_pos.price_open
        elif  portf_pos.side==self._SHORT_POS:
            unit_gross_profit = portf_pos.price_open - last_price
        else:
            raise Exception(f"INVALID SIDE :{portf_pos.side}!! ")

        total_gross_profit = unit_gross_profit * pos_size
        total_net_profit = total_gross_profit - net_commissions

        return self.__update_position_row__(portf_pos.symbol, portf_pos.date_open, portf_pos.side,
                                            portf_pos.date_close,
                                            portf_pos.price_close,
                                            pos_size, unit_gross_profit, total_gross_profit, total_net_profit,
                                            summary_rows)

    def __close_position__(self,entry_symbol,entry_time,position_side,current_time,current_price,entry_price,
                           portf_size,net_commissions,summary_rows):

        pos_size=int(portf_size/entry_price)
        last_time = current_time
        last_price = current_price
        unit_gross_profit = last_price - entry_price
        total_gross_profit = unit_gross_profit * pos_size
        total_net_profit = total_gross_profit - net_commissions

        return self.__update_position_row__(entry_symbol, entry_time, position_side, current_time,
                                            last_time,last_price,pos_size, unit_gross_profit, total_gross_profit,
                                            total_net_profit,summary_rows)


    def __open_portf_position__(self,portf_pos,portf_size,trading_summary_df):
        pos_size = math.floor(portf_size / portf_pos.price_open)

        new_row = pd.DataFrame({
            'symbol': [portf_pos.symbol],
            'open': [portf_pos.date_open],
            'close': [None],
            'side': [portf_pos.side],
            'price_open': [portf_pos.price_open],
            'price_close': [None],
            'unit_gross_profit': [None],
            'total_gross_profit': [None],
            'total_net_profit': [None],
            'portfolio_size': [portf_size],
            'pos_size': [pos_size]
        })

        trading_summary_df = pd.concat([trading_summary_df, new_row], ignore_index=True)
        return  trading_summary_df

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

            if profit is None:
                continue

            if profit < 0:
                current_drawdown += profit
                max_cum_drawdown = min(max_cum_drawdown, current_drawdown)
            else:
                current_drawdown = 0  # Reset the drawdown when there's a profit

        return daily_net_profit, total_positions, max_cum_drawdown, trading_summary_df