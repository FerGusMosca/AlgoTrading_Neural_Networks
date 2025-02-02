import numpy as np
import pandas as pd
from scipy.stats import linregress

from business_entities.portf_position import PortfolioPosition
from common.enums.columns_prefix import ColumnsPrefix
from common.util.slope_calculator import SlopeCalculator
from logic_layer.trading_algos.base_class_daily_trading_backtester import BaseClassDailyTradingBacktester
from logic_layer.trading_algos.slope_backtester import SlopeBacktester


class DirectSlopeBacktester(SlopeBacktester):

    def __init__(self):
        pass


    def long_signal(self,slope):
        return slope>0

    def close_long_signal(self,slope):
        return  slope<0

    def backtest_slope(self,series_df,indicator,portf_size,n_algo_param_dict,etf_comp_dto_arr=None):

        series_df=SlopeCalculator.calculate_indicator_slope(series_df,
                                                            int(n_algo_param_dict[DirectSlopeBacktester._SLOPE_UNITS_COL]),
                                                            indicator)
        if sum(col.startswith(ColumnsPrefix.CLOSE_PREFIX.value) for col in series_df.columns)<=1:
            trading_summary_df = self.__run_trades_single_pos__(series_df, portf_size, indicator, n_algo_param_dict)
            return self.__calculate_day_trading_single_pos_summary__("base algo", trading_summary_df, series_df)
        else:
            trading_summary_df = self.__run_trades_mult_pos__(series_df, portf_size, indicator, n_algo_param_dict,etf_comp_dto_arr)
            return self.__calculate_day_trading_multiple_pos_summary__("mult_pos_algo",trading_summary_df)





