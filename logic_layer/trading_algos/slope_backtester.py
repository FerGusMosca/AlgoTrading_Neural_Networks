import pandas as pd

from business_entities.portf_position import PortfolioPosition
from common.enums.columns_prefix import ColumnsPrefix
from common.util.dataframe_filler import DataframeFiller
from common.util.slope_calculator import SlopeCalculator
from logic_layer.trading_algos.base_class_daily_trading_backtester import BaseClassDailyTradingBacktester


class SlopeBacktester(BaseClassDailyTradingBacktester):

    def __init__(self):
        pass

    def long_signal(self, current_value,current_slope):
        raise Exception("long_signal not implemented in base class!")


    def close_long_signal(self,current_value,current_slope):
        raise Exception("close_long_signal not implemented in base class!")

    def get_algo_name(self):
        raise Exception("get_algo_name not implemented in base class!")

    '''
    def __run_trades_single_pos__(self, predictions_df, init_portf_size,slope_indicator,n_algo_param_dict):
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
        last_portf_size=round(init_portf_size,2)
        net_commissions=0
        new_portf_size=init_portf_size
        portf_positions=[]

        for index, row in predictions_df.iterrows():
            current_symbol = row[SlopeBacktester._TRADING_SYMBOL_COL]
            current_date = pd.to_datetime(row[SlopeBacktester._DATE_COL])
            current_price = row[f"{SlopeBacktester._CLOSE_COL_PREFIX}_{current_symbol}"]
            current_slope = row[f"{slope_indicator}_{SlopeBacktester._SLOPE_POSFIX}"]
            current_value=row[f"{slope_indicator}"]

            print(f"Processing row {index}: DATE={current_date}, slope={current_slope}, price={current_price}")

            if not pd.notna(current_slope) or not pd.notna(current_price):
                continue#We wait to have available data

            if portf_pos is None:
                if self.long_signal(current_value,current_slope):
                    # Open position
                    portf_pos = PortfolioPosition(current_symbol)
                    new_portf_size, net_commissions = self.__extract_commission__(last_portf_size, n_algo_param_dict)
                    portf_pos.open_pos(SlopeBacktester._LONG_ACTION, current_date, current_price,units=new_portf_size/current_price)

                    last_portf_size=init_portf_size if trading_summary_df.empty else trading_summary_df.iloc[-1][SlopeBacktester._END_PORTF_SIZE_COL]
                    new_portf_size,net_commissions= self.__extract_commission__(last_portf_size,n_algo_param_dict)
                    trading_summary_df = self.__open_portf_position__(portf_pos, new_portf_size, trading_summary_df,
                                                                      apply_round_units=False)
                    continue

            else: #Position Closed

                if self.close_long_signal(current_value,current_slope):
                    final_MTM = portf_pos.calculate_and_append_MTM(current_date,current_price)
                    last_portf_size = final_MTM - net_commissions

                    trading_summary_df = self.__close_portf_position__(portf_pos, current_date, current_price,
                                                                       new_portf_size, net_commissions,
                                                                       trading_summary_df)
                    portf_positions.append(portf_pos)
                    portf_pos = None
                else:
                    portf_pos.calculate_and_append_MTM(current_date,current_price)

            # Handle closing positions at the end of the day
            if index == len(predictions_df) - 1:
                if portf_pos is not None:
                    trading_summary_df = self.__close_portf_position__(portf_pos, current_date, current_price,
                                                                       new_portf_size, net_commissions,
                                                                       trading_summary_df)
                    portf_positions.append(portf_pos)
                    portf_pos = None

        return trading_summary_df,portf_positions


    '''

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
            current_slope = row[f"{slope_indicator}_{SlopeBacktester._SLOPE_POSFIX}"]
            current_value = row[f"{slope_indicator}"]

            print(f"Processing row {index}: DATE={current_date}, slope={current_slope} for symbols ={current_symbol}")

            if not pd.notna(current_slope) :
                continue#We wait to have available data

            if portf_pos is None:
                if self.long_signal(current_value,current_slope):
                    # Open position
                    new_portf_size, net_commissions = self.__extract_commission__(last_portf_size, n_algo_param_dict)
                    portf_pos= self.__initiate_portfolio_multiple_positions__(current_symbol,SlopeBacktester._LONG_ACTION,
                                                                                    current_date,new_portf_size,
                                                                                    etfs_active_const_arr,row) #Mark To Market

                    trading_summary_df = self.__open_portf_position__(portf_pos, new_portf_size, trading_summary_df,apply_round_units=False)
                    continue

            else: #Position Closed

                if self.close_long_signal(current_value,current_slope):
                    final_MTM = portf_pos.calculate_and_append_MTM(row,current_date)
                    last_portf_size= final_MTM - net_commissions
                    trading_summary_df = self.__close_portf_position__(portf_pos, current_date, final_MTM,
                                                                       new_portf_size, net_commissions,
                                                                       trading_summary_df,last_portf_size=last_portf_size)

                    portf_positions.append(portf_pos)
                    portf_pos = None
                else:
                    portf_pos.calculate_and_append_MTM(row, current_date,error_if_missing=False)

        # Close all the open positions
        if portf_pos is not None:
            final_MTM = portf_pos.calculate_and_append_MTM(row, current_date)
            last_portf_size = final_MTM - net_commissions
            trading_summary_df = self.__close_portf_position__(portf_pos, current_date, final_MTM,
                                                               new_portf_size, net_commissions,
                                                               trading_summary_df,
                                                               last_portf_size=last_portf_size)
            portf_positions.append(portf_pos)
            portf_pos = None

        return trading_summary_df,portf_positions

    def backtest(self,series_df,indicator,portf_size,n_algo_param_dict,etf_comp_dto_arr=None):

        # 1-We calculate the slope value
        series_df=SlopeCalculator.calculate_indicator_slope(series_df,
                                                            int(n_algo_param_dict[BaseClassDailyTradingBacktester._SLOPE_UNITS_COL]),
                                                            indicator)

        #2-Expand and repeat the prev values for missing values
        series_df = DataframeFiller.fill_missing_values(series_df,col=indicator)  # We fill missing values with the last one

        #3- We drop weekends and holidays
        series_df = series_df.dropna(how='all', subset=[col for col in series_df.columns if
                                                        col.startswith(ColumnsPrefix.CLOSE_PREFIX.value)])

        trading_summary_df, port_pos = self.__run_trades_mult_pos__(series_df, portf_size, indicator, n_algo_param_dict,
                                                                    etf_comp_dto_arr)
        return self.__calculate_day_trading_multiple_pos_summary__("mult_pos_algo", trading_summary_df), port_pos