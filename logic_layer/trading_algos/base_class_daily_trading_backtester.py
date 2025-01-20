import math
import pandas as pd
import numpy as np

from common.util.financial_calculation_helper import FinancialCalculationsHelper


class BaseClassDailyTradingBacktester:

    _DATE_COL="date"
    _LONG_ACTION="LONG"
    _SLOPE_UNITS_COL="slope_units"
    _TRADING_SYMBOL_COL="trading_symbol"
    _END_PORTF_SIZE_COL= "end_portfolio"
    _CLOSE_COL_PREFIX="close"
    _SLOPE_POSFIX="Slope"

    _TRADE_COMM_PCT_KEY="trade_comm_pct"
    _TRADE_COMM_NOM_KEY="trade_comm"

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

    def __extract_commission__(self,last_portf_size,n_algo_param_dict):

        if BaseClassDailyTradingBacktester._TRADE_COMM_PCT_KEY in n_algo_param_dict:
            comm_pct = float(n_algo_param_dict[BaseClassDailyTradingBacktester._TRADE_COMM_PCT_KEY])
            comm= last_portf_size*comm_pct
            new_portf_size=last_portf_size-comm
            return new_portf_size,comm
        elif BaseClassDailyTradingBacktester._TRADE_COMM_NOM_KEY in n_algo_param_dict:
            comm= float(n_algo_param_dict[BaseClassDailyTradingBacktester._TRADE_COMM_NOM_KEY])
            new_portf_size=last_portf_size-comm
            return  new_portf_size,comm
        else:
            return 0

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
                                pos_size,unit_gross_profit,total_gross_profit,total_net_profit,summary_rows,
                                init_portf_size=None):

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

        end_portfolio_size=init_portf_size + total_net_profit if init_portf_size is not None else 0

        summary_rows.loc[row_mask, 'end_portfolio'] = end_portfolio_size
        summary_rows.loc[row_mask, 'portfolio_size'] = end_portfolio_size
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
                                             pos_size,unit_gross_profit,total_gross_profit,total_net_profit,summary_rows,
                                             init_portf_size=portf_size)

    def __close_portf_position__(self,portf_pos,current_time,current_price,portf_size,nom_net_comm,summary_rows):
        portf_pos.date_close=current_time
        portf_pos.price_close=current_price

        pos_size=summary_rows.iloc[-1]["pos_size"]
        last_time = current_time
        last_price = current_price


        if portf_pos.side==self._LONG_POS:
            unit_gross_profit = last_price - portf_pos.price_open
        elif  portf_pos.side==self._SHORT_POS:
            unit_gross_profit = portf_pos.price_open - last_price
        else:
            raise Exception(f"INVALID SIDE :{portf_pos.side}!! ")

        total_gross_profit = unit_gross_profit * pos_size
        total_net_profit = total_gross_profit - nom_net_comm

        return self.__update_position_row__(portf_pos.symbol, portf_pos.date_open, portf_pos.side,
                                            portf_pos.date_close,
                                            portf_pos.price_close,
                                            pos_size, unit_gross_profit, total_gross_profit, total_net_profit,
                                            summary_rows,
                                            init_portf_size=portf_size)

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
                                            total_net_profit,summary_rows,init_portf_size=portf_size)


    def __open_portf_position__(self,portf_pos,init_portf_size,trading_summary_df,apply_round_units=False):

        if apply_round_units:
            pos_size = math.floor(init_portf_size / portf_pos.price_open)
        else:
            pos_size= init_portf_size / portf_pos.price_open

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
            'end_portfolio':None,
            'portfolio_size': [init_portf_size],
            'pos_size': [pos_size]
        })

        trading_summary_df = pd.concat([trading_summary_df, new_row], ignore_index=True)
        return  trading_summary_df

    def __calculate_day_trading_summary__(self, trading_summary_df, prices_df):
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

        if prices_df is not None:#we calculate the max drawdown more accurately
            max_cum_drawdown=FinancialCalculationsHelper.calculate_max_drawdown_with_prices(trading_summary_df,prices_df,trading_summary_df["symbol"].iloc[0])

        return daily_net_profit, total_positions, max_cum_drawdown, trading_summary_df