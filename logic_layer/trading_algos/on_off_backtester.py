from common.enums.columns_prefix import ColumnsPrefix
from common.util.pandas_dataframes.dataframe_filler import DataframeFiller
from common.util.financial_calculations.slope_calculator import SlopeCalculator
from logic_layer.trading_algos.slope_backtester import SlopeBacktester


class OnOffBacktester(SlopeBacktester):


    def __init__(self):
        pass


    def long_signal(self,current_value,current_slope):
        return current_value>0

    def close_long_signal(self,current_value,current_slope):
        return  current_value<0

    def get_algo_name(self):
        return "on_off_value"


    def backtest(self,series_df,indicator,portf_size,n_algo_param_dict,etf_comp_dto_arr=None):

        #1-We calculate the on off value
        series_df=SlopeCalculator.calculate_indicator_on_off_slope(series_df,indicator)

        #2-Expand and repeat the prev values for missing values
        series_df = DataframeFiller.fill_missing_values(series_df,col=indicator)  # We fill missing values with the last one

        # 3- We drop weekends and holidays
        series_df = series_df.dropna(how='all', subset=[col for col in series_df.columns if
                                                        col.startswith(ColumnsPrefix.CLOSE_PREFIX.value)])

        trading_summary_df, portf_positions = self.__run_trades_mult_pos__(series_df, portf_size, indicator,
                                                                           n_algo_param_dict, etf_comp_dto_arr)
        return self.__calculate_day_trading_multiple_pos_summary__("mult_pos_algo", trading_summary_df), portf_positions

