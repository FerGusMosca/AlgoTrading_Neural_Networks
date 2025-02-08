import numpy as np
import pandas as pd
from scipy.stats import linregress

from business_entities.portf_position import PortfolioPosition
from common.enums.columns_prefix import ColumnsPrefix
from common.util.slope_calculator import SlopeCalculator
from logic_layer.trading_algos.base_class_daily_trading_backtester import BaseClassDailyTradingBacktester
from logic_layer.trading_algos.slope_backtester import SlopeBacktester


class InvSlopeBacktester(SlopeBacktester):

    def __init__(self):
        pass


    def long_signal(self,current_value,current_slope):
        return current_slope<0

    def close_long_signal(self,current_value,current_slope):
        return  current_slope>0

    def get_algo_name(self):
        return "inv_slope"




