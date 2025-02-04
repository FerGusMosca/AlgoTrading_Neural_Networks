import numpy as np

from common.enums.columns_prefix import ColumnsPrefix
from common.enums.indicator_type import IndicatorType
from common.enums.on_off_indicator_values import OnOffIndicatorValue
from common.util.slope_calculator import SlopeCalculator
from logic_layer.trading_algos.inv_slope_backtester import InvSlopeBacktester
from logic_layer.trading_algos.slope_backtester import SlopeBacktester


class SintheticIndicatorCreator():

    def __init__(self):
        pass



    def build_sinthetic_indicator(self,indicators_series_df,indicator_type_arr,n_algo_param_dict):

        for indicator in indicator_type_arr:
            if (indicator.type==IndicatorType.DIRECT_SLOPE.value
                    or indicator.type==IndicatorType.INV_SLOPE.value):
                indicators_series_df=SlopeCalculator.calculate_indicator_slope(indicators_series_df,
                                                                    int(n_algo_param_dict[InvSlopeBacktester._SLOPE_UNITS_COL]),
                                                                    f"{ColumnsPrefix.CLOSE_PREFIX.value}{indicator.indicator}")

        for index,row in indicators_series_df.iterrows():
            all_on_ind=True
            for indicator in indicator_type_arr:

                if indicator.type==IndicatorType.DIRECT_SLOPE.value:
                    current_slope = row[f"{ColumnsPrefix.CLOSE_PREFIX.value}{indicator.indicator}_{SlopeBacktester._SLOPE_POSFIX}"]
                    all_on_ind=False if np.isnan(current_slope) or current_slope<0 else all_on_ind
                elif indicator.type==IndicatorType.INV_SLOPE.value:
                    current_slope = row[f"{ColumnsPrefix.CLOSE_PREFIX.value}{indicator.indicator}_{SlopeBacktester._SLOPE_POSFIX}"]
                    all_on_ind = False if np.isnan(current_slope) or current_slope > 0 else all_on_ind
                else:
                    raise Exception(f"Could not recognize indicator type {indicator.type}")

                if not all_on_ind:
                    break


            #TODO better structure this fields
            indicators_series_df.at[index, 'signal'] = OnOffIndicatorValue.ON_NUMERIC.value if all_on_ind \
                                                                else OnOffIndicatorValue.OFF_NUMERIC.value


        return indicators_series_df