import numpy as np

from common.enums.parameters.arima_parameters import ArimaParameters
from common.enums.columns_prefix import ColumnsPrefix
from common.enums.indicator_type import IndicatorType
from common.enums.on_off_indicator_values import OnOffIndicatorValue
from common.enums.parameters.positive_threshold_parameters import PositiveThresholdParameters
from common.enums.parameters.sudden_stop_parameters import SuddenStopParameters
from common.util.slope_calculator import SlopeCalculator
from framework.common.logger.message_type import MessageType
from logic_layer.ARIMA_models_analyzer import ARIMAModelsAnalyzer
from logic_layer.indicator_algos.sudden_stop_indicator import SuddenStopIndicator
from logic_layer.trading_algos.inv_slope_backtester import InvSlopeBacktester
from logic_layer.trading_algos.slope_backtester import SlopeBacktester


class SintheticIndicatorCreator():

    def __init__(self,logger=None):
        self.sudden_stop_ind={}
        self.logger=logger


    def __extract_variable__(self,key,n_algo_param_dict,indicator_type,optional=False,def_val=None):

        if key in n_algo_param_dict:
            return  n_algo_param_dict[key]
        else:
            if not optional:
                raise Exception(f"Could not find necessary parameter {key} for indicator type {indicator_type}")
            else:
                return  def_val


    def __proces__direct_slope__indicator__(self,row,indicator):
        current_slope = row[f"{ColumnsPrefix.CLOSE_PREFIX.value}{indicator.indicator}_{SlopeBacktester._SLOPE_POSFIX}"]
        all_on_ind = False if np.isnan(current_slope) or current_slope < 0 else True
        return all_on_ind

    def __proces__inv_slope__indicator__(self,row,indicator):
        current_slope = row[f"{ColumnsPrefix.CLOSE_PREFIX.value}{indicator.indicator}_{SlopeBacktester._SLOPE_POSFIX}"]
        all_on_ind = False if np.isnan(current_slope) or current_slope > 0 else True
        return all_on_ind

    def __process_threshold_indicator__(self,indicators_series_df,row,indicator,n_algo_param_dict, inv_ind=False):
        date=row["date"]
        filtered_df = indicators_series_df[indicators_series_df['date'] <= date]

        self.logger.do_log(f"Processing predictions for {date} for indicator {indicator.indicator}",MessageType.INFO)
        pos_threshold = self.__extract_variable__(PositiveThresholdParameters.pos_threshold.value, n_algo_param_dict, indicator.type)

        curr_val = row[f"{ColumnsPrefix.CLOSE_PREFIX.value}{indicator.indicator}"]

        self.logger.do_log(
            f"Predictions for {date} for indicator {indicator.indicator} successfully processed:",
            MessageType.INFO)

        return curr_val>pos_threshold if inv_ind is False else curr_val<pos_threshold

    def __process_sudden_top__(self,indicators_series_df,row,indicator,n_algo_param_dict, inv_ind=False):
        date=row["date"]
        key = indicator.indicator.strip().upper()
        if key not in  self.sudden_stop_ind:
            st_units = self.__extract_variable__(SuddenStopParameters.st_units.value, n_algo_param_dict, indicator.type)
            st_eval_p = self.__extract_variable__(SuddenStopParameters.st_eval_p.value, n_algo_param_dict, indicator.type)
            st_blackout_p = self.__extract_variable__(SuddenStopParameters.st_blackout_p.value, n_algo_param_dict, indicator.type)

            self.sudden_stop_ind[key]=SuddenStopIndicator(indicator.indicator,st_units,st_eval_p,st_blackout_p)


        sudden_stop_ind=self.sudden_stop_ind[key]
        self.logger.do_log(
            f"Processing predictions for {date} for indicator {indicator.indicator} (units={sudden_stop_ind.units} months)",
            MessageType.INFO)

        sudden_stop = sudden_stop_ind.eval_sudden_stop(date,indicators_series_df)

        return not sudden_stop #If not sudden Stop --> Indicator active


    def __process_arima_indicator__(self,indicators_series_df,row,indicator,n_algo_param_dict, inv_ind=False):
        arima_Analyzer = ARIMAModelsAnalyzer(self.logger)

        date=row["date"]
        filtered_df = indicators_series_df[indicators_series_df['date'] <= date]

        self.logger.do_log(f"Processing predictions for {date} for indicator {indicator.indicator}",MessageType.INFO)
        p = self.__extract_variable__(ArimaParameters.p.value, n_algo_param_dict, indicator.type)
        d = self.__extract_variable__(ArimaParameters.d.value, n_algo_param_dict, indicator.type)
        q = self.__extract_variable__(ArimaParameters.q.value, n_algo_param_dict, indicator.type)
        period = self.__extract_variable__(ArimaParameters.period.value, n_algo_param_dict, indicator.type,
                                           optional=True, def_val=None)
        min_units_to_pred = self.__extract_variable__(ArimaParameters.min_units_to_pred.value, n_algo_param_dict, indicator.type,
                                           optional=True, def_val=24)

        step = self.__extract_variable__(ArimaParameters.step.value, n_algo_param_dict, indicator.type)
        inv_steps = self.__extract_variable__(ArimaParameters.inv_steps.value, n_algo_param_dict,
                                              indicator.type)


        if len(filtered_df)<min_units_to_pred:
            return OnOffIndicatorValue.OFF_NUMERIC.value
        else:
            preds = arima_Analyzer.build_and__predict_ARIMA_model(filtered_df,
                                                                  f"{ColumnsPrefix.CLOSE_PREFIX.value}{indicator.indicator}",
                                                                  p, d, q, period, step)
            all_on_ind= arima_Analyzer.eval_still_on_indicator(preds, inv_steps,inv_steps )
            active_ind= all_on_ind if not inv_ind else not all_on_ind
            self.logger.do_log(f"Predictions for {date} for indicator {indicator.indicator} successfully processed: ON --> {active_ind}", MessageType.INFO)

            return  active_ind


    def __process_sarima_indicator__(self,indicators_series_df,row,indicator,n_algo_param_dict, inv_ind=False):
        arima_Analyzer = ARIMAModelsAnalyzer(self.logger)

        date=row["date"]
        filtered_df = indicators_series_df[indicators_series_df['date'] <= date]

        self.logger.do_log(f"Processing predictions for {date} for indicator {indicator.indicator}",MessageType.INFO)
        p = self.__extract_variable__(ArimaParameters.p.value, n_algo_param_dict, indicator.type)
        d = self.__extract_variable__(ArimaParameters.d.value, n_algo_param_dict, indicator.type)
        q = self.__extract_variable__(ArimaParameters.q.value, n_algo_param_dict, indicator.type)
        s= self.__extract_variable__(ArimaParameters.s.value, n_algo_param_dict, indicator.type)
        period = self.__extract_variable__(ArimaParameters.period.value, n_algo_param_dict, indicator.type,
                                           optional=True, def_val=None)
        min_units_to_pred = self.__extract_variable__(ArimaParameters.min_units_to_pred.value, n_algo_param_dict, indicator.type,
                                           optional=True, def_val=24)

        step = self.__extract_variable__(ArimaParameters.step.value, n_algo_param_dict, indicator.type)
        inv_steps = self.__extract_variable__(ArimaParameters.inv_steps.value, n_algo_param_dict,
                                              indicator.type)


        if len(filtered_df)<min_units_to_pred:
            return OnOffIndicatorValue.OFF_NUMERIC.value
        else:
            preds = arima_Analyzer.build_and_predict_SARIMA_model(filtered_df,
                                                                  f"{ColumnsPrefix.CLOSE_PREFIX.value}{indicator.indicator}",
                                                                  p, d, q,p,d,q,s, period, step)
            all_on_ind= arima_Analyzer.eval_still_on_indicator(preds, inv_steps,inv_steps )
            active_ind= all_on_ind if not inv_ind else not all_on_ind
            self.logger.do_log(f"Predictions for {date} for indicator {indicator.indicator} successfully processed: ON --> {active_ind}", MessageType.INFO)

            return  active_ind

    def build_sinthetic_indicator(self,indicators_series_df,indicator_type_arr,n_algo_param_dict):

        for indicator in indicator_type_arr:
            if (indicator.type==IndicatorType.DIRECT_SLOPE.value
                    or indicator.type==IndicatorType.INV_SLOPE.value
                   ):
                indicators_series_df=SlopeCalculator.calculate_indicator_slope(indicators_series_df,
                                                                    int(n_algo_param_dict[InvSlopeBacktester._SLOPE_UNITS_COL]),
                                                                    f"{ColumnsPrefix.CLOSE_PREFIX.value}{indicator.indicator}")

        for index,row in indicators_series_df.iterrows():
            all_on_ind_arr=[]
            for indicator in indicator_type_arr:
                all_on_ind=True
                if indicator.type==IndicatorType.DIRECT_SLOPE.value:
                    all_on_ind=self.__proces__direct_slope__indicator__(row,indicator)
                elif indicator.type==IndicatorType.INV_SLOPE.value:
                    all_on_ind = self.__proces__inv_slope__indicator__(row, indicator)
                elif indicator.type == IndicatorType.ARIMA.value:
                    all_on_ind=self.__process_arima_indicator__(indicators_series_df, row, indicator,
                                                                n_algo_param_dict,inv_ind=False)
                elif indicator.type == IndicatorType.POS_THRESHOLD.value:
                    all_on_ind= self.__process_threshold_indicator__(indicators_series_df, row, indicator,
                                                                     n_algo_param_dict,inv_ind=False)
                elif indicator.type == IndicatorType.SARIMA.value:
                    all_on_ind=self.__process_sarima_indicator__(indicators_series_df, row, indicator,
                                                                n_algo_param_dict,inv_ind=False)

                elif indicator.type == IndicatorType.INV_ARIMA.value:
                    all_on_ind=self.__process_arima_indicator__(indicators_series_df, row, indicator,
                                                                n_algo_param_dict,inv_ind=True)
                elif indicator.type == IndicatorType.INV_SARIMA.value:
                    all_on_ind=self.__process_sarima_indicator__(indicators_series_df, row, indicator,
                                                                n_algo_param_dict,inv_ind=True)
                elif indicator.type == IndicatorType.SUDDEN_STOP.value:
                    all_on_ind=self.__process_sudden_top__(indicators_series_df, row, indicator,
                                                           n_algo_param_dict,inv_ind=True)
                else:
                    raise Exception(f"Could not recognize indicator type {indicator.type}")

                all_on_ind_arr.append(all_on_ind)


            all_ind_active= all(all_on_ind_arr)

            indicators_series_df.at[index, 'signal'] = OnOffIndicatorValue.ON_NUMERIC.value if all_ind_active \
                                                                else OnOffIndicatorValue.OFF_NUMERIC.value


        return indicators_series_df