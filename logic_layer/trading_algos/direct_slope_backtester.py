import numpy as np
import pandas as pd
from scipy.stats import linregress

from business_entities.portf_position import PortfolioPosition
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

    def backtest_slope(self,series_df,trading_symbol,indicator,portf_size,n_algo_param_dict):

        series_df=SlopeCalculator.calculate_indicator_slope(series_df,
                                                            int(n_algo_param_dict[DirectSlopeBacktester._SLOPE_UNITS_COL]),
                                                            indicator)

        trading_summary_df = self.__summarize_trading_positions__(series_df, portf_size, indicator, n_algo_param_dict)

        return self.__calculate_day_trading_summary__(trading_summary_df)


