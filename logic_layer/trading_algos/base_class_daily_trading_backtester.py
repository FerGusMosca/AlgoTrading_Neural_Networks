import copy
import math
import pandas as pd
import numpy as np

from business_entities.porf_mult_positions import PortfMultPositions
from business_entities.portf_position import PortfolioPosition
from common.dto.etf_position_dto import ETFPositionDTO
from common.dto.strategy_summary_dto import StrategySummaryDTO
from common.enums.columns_prefix import ColumnsPrefix
from common.enums.sliding_window_strategy import SlidingWindowStrategy
from common.util.date_handler import DateHandler
from common.util.financial_calculation_helper import FinancialCalculationsHelper
from common.util.light_logger import LightLogger


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

    def __eval_reuse_reference_price__(self,algo,last_trading_dict,side,new_date,new_ref_price):
        try:
            if(last_trading_dict is not None):
                #1- We get the last trade of the algo
                res = last_trading_dict[algo]

                sorted_positions = sorted(res.portf_pos_summary, key=lambda x: x.date_close, reverse=True)

                if sorted_positions is not None and len(sorted_positions)>0:
                    last_pos = sorted_positions[0]
                    if DateHandler.evaluate_consecutive_days(last_pos.date_close,new_date) and last_pos.side==side:
                        #We have to consecutive days, we can use the old ref-price as opening price
                        return last_pos.price_close
        except Exception as e:
            raise Exception("Error evaluating previous day for algo {} for date {}:{}".format( algo,new_date,str(e)))

        return  new_ref_price

    def __validate_regime__(self, pos_regime_df, current_date,is_pos=True):
        """
        Returns False if any regime-switch indicator is active (non-zero) as of current_date.
        Assumes each regime symbol reports data at different frequencies (daily or monthly).
        """
        if pos_regime_df is None or pos_regime_df.empty:
            return True  # No regime filters, always valid

        current_date = pd.to_datetime(current_date)
        symbols = pos_regime_df["symbol"].unique()

        for symbol in symbols:
            # Filter data for this symbol up to the current date
            symbol_df = pos_regime_df[(pos_regime_df["symbol"] == symbol) & (pos_regime_df["date"] <= current_date)]

            if symbol_df.empty:
                continue  # No data available yet for this regime variable

            # Use the most recent value up to this date (important for monthly data like FEDFUNDS)
            last_row = symbol_df.sort_values("date").iloc[-1]

            # If any OHLC value is non-zero and not NaN, we consider a regime switch is active
            for col in ["open", "high", "low", "close"]:
                val = last_row[col]

                if is_pos and  pd.notna(val) and (val < 0):
                    LightLogger.do_log(f"Positive Regime switch triggered by {symbol} on {last_row['date']}")
                    return False

                if not is_pos and pd.notna(val) and ( val!=0):
                    LightLogger.do_log(f"Regime switch triggered by {symbol} on {last_row['date']}")
                    return False

        return True  # No regime switch detected

    def __validate_bias__(self,side,bias):
        if bias==SlidingWindowStrategy.NONE.value:
            return True
        else:
            return side==bias


    def __eval_exists_value_on_df__(self,panda_df,key,key_val,val_col):
        if panda_df[key] is not None:
            row_df= panda_df[panda_df[key]==key_val]

            if row_df is not None and len(row_df)>0:
                return  True
            else:
                return  False
        else:
            return  False
    def __extract_value_from_df__(self,panda_df,key,key_val,val_col ):

        if panda_df[key] is not None:
            row_df= panda_df[panda_df[key]==key_val]

            if row_df is not None and len(row_df)>0:
                return  row_df[val_col]
            else:
                raise Exception("Could not find column {} for a row with key value {}".format(val_col,key))
        else:
            raise Exception("Could not find row wiht key {}".format(key))

    def __extract_value_from_row__(self,panda_df,key,key_val,val_col ):

        if panda_df[key] is not None:
            row_df= panda_df[panda_df[key]==key_val]

            if row_df is not None and len(row_df)>0:
                return  row_df[val_col].values[0]
            else:
                raise Exception("Could not find column {} for a row with key value {}".format(val_col,key))
        else:
            raise Exception("Could not find row wiht key {}".format(key))

    def __initialize_dataframe__(self):

        # Initialize an empty list to store the rows of the trading summary
        trading_summary_df = pd.DataFrame(columns=[
            'symbol', 'open', 'close', 'side', 'price_open', 'price_close', 'unit_gross_profit', 'total_gross_profit',
            'total_net_profit', 'init_portfolio', 'end_portfolio','pos_size','max_drawdown',
            'max_drawdown_pct','pct_profit'
        ])

        return  trading_summary_df

    def __extract_commission__(self,last_portf_size,n_algo_param_dict):

        if BaseClassDailyTradingBacktester._TRADE_COMM_PCT_KEY in n_algo_param_dict:
            comm_pct = float(n_algo_param_dict[BaseClassDailyTradingBacktester._TRADE_COMM_PCT_KEY])
            comm= round( last_portf_size*comm_pct,2)
            new_portf_size= round( last_portf_size-comm,2)
            return  new_portf_size,comm
        elif BaseClassDailyTradingBacktester._TRADE_COMM_NOM_KEY in n_algo_param_dict:
            comm= round( float(n_algo_param_dict[BaseClassDailyTradingBacktester._TRADE_COMM_NOM_KEY]),2)
            new_portf_size= round( last_portf_size-comm,2)
            return new_portf_size,comm
        else:
            return 0

    def __apply_commisions__(self,MTM, net_commissions):
        return MTM-net_commissions


    def __extract_trading_symbols_from_multiple_portf__(self,df, col_prefix="close_"):
        """
            Extracts all column names that start with a given prefix and removes the prefix.

            :param df: Input dataframe
            :param col_prefix: Prefix to filter and remove from column names
            :return: A list of symbols without the prefix
            """
        return [col[len(col_prefix):] for col in df.columns if col.startswith(col_prefix)]


    def __initiate_portfolio_multiple_positions__(self,symbols_csv,side,date,portf_size,curr_pos_portf_arr,prices_row):

        portf_pos_arr=[]
        curr_ETF_MTM=0
        for curr_pos_portf in curr_pos_portf_arr:
            if curr_pos_portf.active:
                if prices_row[ColumnsPrefix.CLOSE_PREFIX.value + curr_pos_portf.symbol] is not None:
                    curr_price=  prices_row[ColumnsPrefix.CLOSE_PREFIX.value + curr_pos_portf.symbol]
                    curr_portf_pos_size=portf_size*curr_pos_portf.active_weight

                    curr_portf_pos_units=curr_portf_pos_size/curr_price

                    portf_pos=PortfolioPosition(curr_pos_portf.symbol)
                    portf_pos.open_pos(side,date,curr_price,units=curr_portf_pos_units)

                    portf_pos_arr.append(portf_pos)

                    curr_ETF_MTM+= curr_portf_pos_units * curr_price
                else:
                    curr_date=prices_row["date"]
                    raise Exception(f"CRITICAL error calculating MTM for security {curr_pos_portf.symbol}: no price found for date {curr_date}")
        return PortfMultPositions(date,symbols_csv,side,portf_pos_arr,curr_ETF_MTM)
        #return curr_ETF_MTM

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
                                init_portf_size=None,max_drawdown=None,max_drawdown_pct=None,pct_profit=None):

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
        #summary_rows.loc[row_mask, 'portfolio_size'] = end_portfolio_size
        summary_rows.loc[row_mask, 'pos_size'] = pos_size
        summary_rows.loc[row_mask, 'max_drawdown'] = max_drawdown
        summary_rows.loc[row_mask, 'max_drawdown_pct'] = max_drawdown_pct
        summary_rows.loc[row_mask, 'pct_profit'] = pct_profit

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
                                             init_portf_size=portf_size,pct_profit=None)

    def __close_portf_position__(self,portf_pos,current_time,current_price,init_portf_size,
                                 nom_net_comm,summary_rows,last_portf_size=None):
        portf_pos.date_close=current_time
        portf_pos.price_close=current_price

        pos_size=summary_rows.iloc[-1]["pos_size"]
        last_price = current_price


        if portf_pos.side==self._LONG_POS:
            unit_gross_profit = round( last_price - portf_pos.price_open,2)
        elif  portf_pos.side==self._SHORT_POS:
            unit_gross_profit = round( portf_pos.price_open - last_price,2)
        else:
            raise Exception(f"INVALID SIDE :{portf_pos.side}!! ")

        total_gross_profit = round( unit_gross_profit * pos_size,2)
        total_net_profit = round( total_gross_profit - nom_net_comm,2)

        #Calculate the max drawdown
        max_drawdown=None
        max_drawdown_pct=None
        if(portf_pos.daily_MTMs is not None and len(portf_pos.daily_MTMs)>0):
            max_drawdown_pct=FinancialCalculationsHelper.max_drawdown_on_MTM(portf_pos.daily_MTMs)
            max_drawdown= str(round(max_drawdown_pct*100,2))+" %"

        #Calculate the pct profit
        profit_pct=None
        if last_portf_size is not None:
            profit_pct = str(round(((last_portf_size / init_portf_size) - 1 )* 100, 2)) + " %"


        return self.__update_position_row__(portf_pos.symbol, portf_pos.date_open, portf_pos.side,
                                            portf_pos.date_close,
                                            portf_pos.price_close,
                                            pos_size, unit_gross_profit, total_gross_profit, total_net_profit,
                                            summary_rows,
                                            init_portf_size=init_portf_size,
                                            max_drawdown=max_drawdown,
                                            max_drawdown_pct=max_drawdown_pct,
                                            pct_profit=profit_pct)

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
            'init_portfolio': [round(init_portf_size,2)],
            'end_portfolio':None,
            'pos_size': [pos_size],
            'max_drawdown':None,
            'max_drawdown_pct':None,
            'pct_profit':None
        })

        trading_summary_df = pd.concat([trading_summary_df, new_row], ignore_index=True)
        return  trading_summary_df

    def __calculate_day_trading_multiple_pos_summary__(self, algo ,trading_summary_df, column_name="end_portfolio"):

        #0 Calculate max drowdown
        #Inter positions
        max_drawdown_inter=FinancialCalculationsHelper.calculate_max_drawdown_on_different_positions(trading_summary_df,column_name)
        #Intra Positions
        max_drowdown_intra = trading_summary_df["max_drawdown_pct"].min()
        max_cum_drawdown = min(max_drawdown_inter,max_drowdown_intra)

        # 1. Sum all values in the 'total_net_profit' column
        daily_net_profit = trading_summary_df['total_net_profit'].sum()

        # 2. Count the total number of positions (number of rows in the DataFrame)
        total_positions = len(trading_summary_df)

        summary_dto=StrategySummaryDTO(algo,daily_net_profit,total_positions,max_cum_drawdown,trading_summary_df)
        return summary_dto

    def __calculate_day_trading_single_pos_summary__(self,algo ,trading_summary_df, prices_df):
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

        if prices_df is not None :#we calculate the max drawdown more accurately
            max_cum_drawdown=FinancialCalculationsHelper.calculate_max_drawdown_with_prices(trading_summary_df,prices_df,trading_summary_df["symbol"].iloc[0])

        summary_dto=StrategySummaryDTO(algo,daily_net_profit,total_positions,max_cum_drawdown,trading_summary_df)
        return summary_dto


    def __extract__active__positions_for_day__(self,row,etf_comp_dto_arr):

        etfs_active_const_arr=[]
        for etf_comp_dto in etf_comp_dto_arr:
            etf_comp_dto_curr_day = copy.deepcopy(etf_comp_dto)
            etf_comp_dto_curr_day.active = not pd.isna( row[f"{BaseClassDailyTradingBacktester._CLOSE_COL_PREFIX}_{etf_comp_dto.symbol}"])
            etf_comp_dto_curr_day.active_weight=etf_comp_dto_curr_day.weight if etf_comp_dto_curr_day.active else 0
            etfs_active_const_arr.append(etf_comp_dto_curr_day)

        ETFPositionDTO.recalculate_weights(etf_comp_dto_arr,etfs_active_const_arr)

        return etfs_active_const_arr

