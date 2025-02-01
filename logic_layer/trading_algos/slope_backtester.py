import pandas as pd

from business_entities.portf_position import PortfolioPosition
from common.enums.columns_prefix import ColumnsPrefix
from logic_layer.trading_algos.base_class_daily_trading_backtester import BaseClassDailyTradingBacktester


class SlopeBacktester(BaseClassDailyTradingBacktester):

    def __init__(self):
        pass

    def long_signal(self, current_slope):
        raise Exception("long_signal not implemented in base class!")


    def close_long_signal(self,slope):
        raise Exception("close_long_signal not implemented in base class!")


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
        net_commissions=0
        new_portf_size=init_portf_size

        for index, row in predictions_df.iterrows():
            current_symbol = row[SlopeBacktester._TRADING_SYMBOL_COL]
            current_date = pd.to_datetime(row[SlopeBacktester._DATE_COL])
            current_price = row[f"{SlopeBacktester._CLOSE_COL_PREFIX}_{current_symbol}"]
            current_slope = row[f"{slope_indicator}_{SlopeBacktester._SLOPE_POSFIX}"]

            print(f"Processing row {index}: DATE={current_date}, slope={current_slope}, price={current_price}")

            if not pd.notna(current_slope) or not pd.notna(current_price):
                continue#We wait to have available data

            if portf_pos is None:
                if self.long_signal(current_slope):
                    # Open position
                    portf_pos = PortfolioPosition(current_symbol)
                    portf_pos.open_pos(SlopeBacktester._LONG_ACTION, current_date, current_price)

                    last_portf_size=init_portf_size if trading_summary_df.empty else trading_summary_df.iloc[-1][SlopeBacktester._END_PORTF_SIZE_COL]
                    new_portf_size,net_commissions= self.__extract_commission__(last_portf_size,n_algo_param_dict)
                    trading_summary_df = self.__open_portf_position__(portf_pos, new_portf_size, trading_summary_df,
                                                                      apply_round_units=False)
                    continue

            else: #Position Closed

                if self.close_long_signal(current_slope):

                    trading_summary_df = self.__close_portf_position__(portf_pos, current_date, current_price,
                                                                       new_portf_size, net_commissions,
                                                                       trading_summary_df)

                    portf_pos = None

            # Handle closing positions at the end of the day
            if index == len(predictions_df) - 1:
                if portf_pos is not None:
                    trading_summary_df = self.__close_portf_position__(portf_pos, current_date, current_price,
                                                                       new_portf_size, net_commissions,
                                                                       trading_summary_df)
                    portf_pos = None

        return trading_summary_df



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
        new_portf_size=init_portf_size
        trading_symbols=self.__extract_trading_symbols_from_multiple_portf__(predictions_df,ColumnsPrefix.CLOSE_PREFIX)

        for index, row in predictions_df.iterrows():
            current_symbol = trading_symbols
            current_date = pd.to_datetime(row[SlopeBacktester._DATE_COL])
            current_price = row[f"{SlopeBacktester._CLOSE_COL_PREFIX}_{current_symbol}"]
            current_slope = row[f"{slope_indicator}_{SlopeBacktester._SLOPE_POSFIX}"]

            print(f"Processing row {index}: DATE={current_date}, slope={current_slope}, price={current_price}")

            if not pd.notna(current_slope) or not pd.notna(current_price):
                continue#We wait to have available data

            if portf_pos is None:
                if self.long_signal(current_slope):
                    # Open position
                    portf_pos = PortfolioPosition(current_symbol)
                    portf_pos.open_pos(SlopeBacktester._LONG_ACTION, current_date, current_price)

                    last_portf_size=init_portf_size if trading_summary_df.empty else trading_summary_df.iloc[-1][SlopeBacktester._END_PORTF_SIZE_COL]
                    new_portf_size,net_commissions= self.__extract_commission__(last_portf_size,n_algo_param_dict)
                    trading_summary_df = self.__open_portf_position__(portf_pos, new_portf_size, trading_summary_df,
                                                                      apply_round_units=False)
                    continue

            else: #Position Closed

                if self.close_long_signal(current_slope):

                    trading_summary_df = self.__close_portf_position__(portf_pos, current_date, current_price,
                                                                       new_portf_size, net_commissions,
                                                                       trading_summary_df)

                    portf_pos = None

            # Handle closing positions at the end of the day
            if index == len(predictions_df) - 1:
                if portf_pos is not None:
                    trading_summary_df = self.__close_portf_position__(portf_pos, current_date, current_price,
                                                                       new_portf_size, net_commissions,
                                                                       trading_summary_df)
                    portf_pos = None

        return trading_summary_df