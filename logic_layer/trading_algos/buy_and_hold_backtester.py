import pandas as pd

from common.enums.columns_prefix import ColumnsPrefix
from common.util.dataframe_filler import DataframeFiller
from common.util.slope_calculator import SlopeCalculator
from logic_layer.trading_algos.base_class_daily_trading_backtester import BaseClassDailyTradingBacktester
from logic_layer.trading_algos.slope_backtester import SlopeBacktester


class BuyAndHoldBacktester(SlopeBacktester):



    def __init__(self):
        pass


    def long_signal(self,current_value,current_slope):
        return True

    def close_long_signal(self,current_value,current_slope):
        return  False

    def get_algo_name(self):
        return "buy_and_hold"



    def __run_trades_mult_pos__(self, predictions_df, init_portf_size,slope_indicator,
                                      n_algo_param_dict,etf_comp_dto_arr=None):
        """
        Summarizes trading positions using the N_MIN_BUFFER_W_FLIP algorithm.

        Parameters:
        result_df (pd.DataFrame): DataFrame containing trading data with columns ['trading_symbol', 'formatted_date', 'action', 'trading_symbol_price']
        pos_size (float): The size of each position in monetary terms.
        n_min (int): The number of minutes to wait before flipping positions (N-minute buffer).

        Returns:
        pd.DataFrame: DataFrame summarizing the trading positions with columns ['symbol', 'open', 'close', 'side', 'price_open', 'price_close', 'unit_gross_profit', 'total_gross_profit', 'total_net_profit', 'portfolio_size', 'pos_size']
        """
        # Initialize an empty list to store the rows of the trading summary
        trading_summary_df =  self.__initialize_dataframe__()

        portf_pos = None
        net_commissions=0
        new_portf_size=round(init_portf_size,2)
        last_portf_size=round(init_portf_size,2)
        trading_symbols=self.__extract_trading_symbols_from_multiple_portf__(predictions_df,ColumnsPrefix.CLOSE_PREFIX.value)
        current_symbol = ",".join(trading_symbols)

        portf_positions=[]

        for index, row in predictions_df.iterrows():
            #1- We calculate the active positions for Today
            etfs_active_const_arr=self.__extract__active__positions_for_day__(row,etf_comp_dto_arr)
            current_date = pd.to_datetime(row[SlopeBacktester._DATE_COL])
            print(f"Processing row {index}: DATE={current_date} for symbols {current_symbol}")


            if portf_pos is None:
                if self.long_signal(0,0):
                    # Open position
                    new_portf_size, net_commissions = self.__extract_commission__(last_portf_size, n_algo_param_dict)
                    portf_pos= self.__initiate_portfolio_multiple_positions__(current_symbol,SlopeBacktester._LONG_ACTION,
                                                                                    current_date,new_portf_size,
                                                                                    etfs_active_const_arr,row) #Mark To Market

                    trading_summary_df = self.__open_portf_position__(portf_pos, new_portf_size, trading_summary_df,apply_round_units=False)
                    continue

            else: #Position Closed
                portf_pos.calculate_and_append_MTM(row, current_date,error_if_missing=False)

            # Handle closing positions at the end of the day
            if index == len(predictions_df) - 1:
                if portf_pos is not None:
                    final_MTM=portf_pos.calculate_and_append_MTM(row, current_date, error_if_missing=False)
                    if final_MTM is None:
                        raise Exception(f"No price for ETF on date {current_date}")

                    trading_summary_df = self.__close_portf_position__(portf_pos, current_date, final_MTM,
                                                                       new_portf_size, net_commissions,
                                                                       trading_summary_df)
                    portf_positions.append(portf_pos)
                    portf_pos = None
                    break


        return trading_summary_df,portf_positions


    def backtest(self,series_df,indicator,portf_size,n_algo_param_dict,etf_comp_dto_arr=None):


        #1-Expand and repeat the prev values for missing values
        series_df = DataframeFiller.fill_missing_values(series_df,col=indicator)  # We fill missing values with the last one

        #2- We drop weekends and holidays
        series_df = series_df.dropna(how='all', subset=[col for col in series_df.columns if
                                                        col.startswith(ColumnsPrefix.CLOSE_PREFIX.value)])

        trading_summary_df,portf_pos = self.__run_trades_mult_pos__(series_df, portf_size, indicator, n_algo_param_dict,
                                                          etf_comp_dto_arr)
        summary_dto= self.__calculate_day_trading_multiple_pos_summary__("mult_pos_algo", trading_summary_df)

        return [summary_dto,portf_pos]

