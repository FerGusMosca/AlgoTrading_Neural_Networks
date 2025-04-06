import asyncio
import traceback
from datetime import timedelta, datetime

from dateutil.relativedelta import relativedelta

from business_entities.porft_summary import PortfSummary
from common.dto.indicator_type_dto import IndicatorTypeDTO
from common.enums.columns_prefix import ColumnsPrefix
from common.enums.grouping_criterias import GroupCriteria as gc
from common.enums.intervals import Intervals
from common.enums.parameters_keys import ParametersKeys
from common.enums.sliding_window_strategy import SlidingWindowStrategy as sws, SlidingWindowStrategy
from common.enums.trading_algo_strategy import TradingAlgoStrategy as tas, TradingAlgoStrategy
from common.util.dataframe_filler import DataframeFiller
from common.util.dataframe_printer import DataframePrinter
from common.util.date_handler import DateHandler
from common.util.economic_value_handler import EconomicValueHandler
from common.util.etf_logic_manager import ETFLogicManager
from common.util.graph_builder import GraphBuilder
from common.util.image_handler import ImageHandler
from common.util.light_logger import LightLogger
from common.util.portfolio_summary_analyzer import PortfolioSummaryAnalyzer
from common.util.random_walk_generator import RandomWalkGenerator
from data_access_layer.date_range_classification_manager import DateRangeClassificationManager
from data_access_layer.economic_series_manager import EconomicSeriesManager
from data_access_layer.timestamp_classification_manager import TimestampClassificationManager

from framework.common.logger.message_type import MessageType
from logic_layer.ARIMA_models_analyzer import ARIMAModelsAnalyzer
from logic_layer.convolutional_neural_netowrk import ConvolutionalNeuralNetwork
from logic_layer.indicator_algos.sintethic_indicator_creator import SintheticIndicatorCreator
from logic_layer.trading_algos.base_class_daily_trading_backtester import BaseClassDailyTradingBacktester
from logic_layer.trading_algos.buy_and_hold_backtester import BuyAndHoldBacktester
from logic_layer.trading_algos.direct_slope_backtester import DirectSlopeBacktester
from logic_layer.trading_algos.inv_slope_backtester import InvSlopeBacktester
from logic_layer.trading_algos.n_min_buffer_w_flip_daily_trading_backtester import NMinBufferWFlipDailyTradingBacktester
from logic_layer.trading_algos.on_off_backtester import OnOffBacktester
from logic_layer.trading_algos.only_signal_n_min_plus_mov_avg import OnlySignalNMinMovAvgBacktester
from logic_layer.trading_algos.raw_algo_daily_trading_backtester import RawAlgoDailyTradingBacktester
from logic_layer.data_set_builder import DataSetBuilder
from logic_layer.deep_neural_network import DeepNeuralNetwork
from logic_layer.indicator_based_trading_backtester import IndicatorBasedTradingBacktester
from logic_layer.ml_models_analyzer import MLModelAnalyzer
import pandas as pd

from logic_layer.neural_network_models_trainer import NeuralNetworkModelTrainer
from logic_layer.daytrading_RNN_model_creator import DayTradingRNNModelCreator

class AlgosOrchestationLogic:
    def __init__(self,hist_data_conn_str,ml_reports_conn_str,p_classification_map_key,logger):

        self.logger=logger

        self.data_set_builder=DataSetBuilder(hist_data_conn_str,ml_reports_conn_str,p_classification_map_key,logger)

        self.date_range_classif_mgr = DateRangeClassificationManager(ml_reports_conn_str)

        self.timestamp_range_classif_mgr=TimestampClassificationManager(ml_reports_conn_str)


    def __classify_group__(self,classifications, grouping_classif_criteria):
        unique_classes = classifications.unique()

        # If there's only one unique classification in the group, return it
        if len(unique_classes) == 1:
            return unique_classes[0]

        # Apply different criteria for multiple classifications
        if gc(grouping_classif_criteria) == gc.GO_FIRST:
            # Return the first classification in the group
            return classifications.iloc[0]
        elif gc(grouping_classif_criteria) == gc.GO_HIGHEST_COUNT:
            # Return the most frequent classification
            return classifications.mode().iloc[0]
        elif gc(grouping_classif_criteria) == gc.GO_FLAT_ON_DIFF:
            # Mark the entire group as "FLAT"
            return "FLAT"
        else:
            raise ValueError("Unknown grouping_classif_criteria")

    def __group_dataframe__(self, training_series_df, grouping_unit, variables_csv, grouping_classif_criteria=None,
                            classif_key=None):
        """
        Groups the DataFrame by a specified time unit, applies OHLC (open, high, low, close) calculations
        on the specified variables, and optionally adds a classification column based on a criterion.

        :param training_series_df: The input DataFrame containing time series data.
        :param grouping_unit: The time unit to group by (e.g., '10min', '30min').
        :param variables_csv: Comma-separated string of variable names (e.g., 'SPY,VIX').
        :param grouping_classif_criteria: Criteria for adding a classification column (optional).
        :param classif_key: Column to be classified based on the grouping criteria (optional).
        :return: DataFrame with grouped OHLC data for the specified variables.
        """
        # Convert the 'date' column to datetime and set it as the DataFrame index
        training_series_df['date'] = pd.to_datetime(training_series_df['date'])
        training_series_df = training_series_df.set_index('date')

        # List to store grouped data for each variable
        grouped_data = []

        # Iterate over the variables (symbols) listed in variables_csv
        for var in variables_csv.split(','):
            # Ensure the column for the variable exists in the DataFrame
            if var not in training_series_df.columns:
                continue

            # Resample the data based on the grouping unit and calculate OHLC (open, high, low, close)
            resampled_df = training_series_df[[var]].resample(f'{grouping_unit}min').agg({
                var: ['first', 'max', 'min', 'last']
            })

            # Rename the columns to reflect 'open', 'high', 'low', 'close' for the current variable
            resampled_df.columns = [f'open_{var}', f'high_{var}', f'low_{var}', f'close_{var}']

            # Append the resampled and renamed DataFrame to the grouped data list
            grouped_data.append(resampled_df)

        # Concatenate all grouped DataFrames along the columns
        final_grouped_df = pd.concat(grouped_data, axis=1)

        # Optionally add a classification column based on the specified criteria
        if grouping_classif_criteria is not None and classif_key is not None:
            # Resample the classification key and apply the classification criteria
            classification_df = training_series_df.resample(f'{grouping_unit}min')[classif_key].apply(
                lambda x: self.__classify_group__(x, grouping_classif_criteria)
            )
            # Add the classification data to the final grouped DataFrame
            final_grouped_df[classif_key] = classification_df

        # Reset the index to bring 'date' back as a column
        final_grouped_df = final_grouped_df.reset_index()

        # Drop any rows with NaN values that might have been generated during resampling
        final_grouped_df = final_grouped_df.dropna()

        return final_grouped_df

    def __log_day_trading_results__(self,day,daily_net_profit,total_positions,trading_summary_df):
        self.logger.do_log(
            f"Results for day {day}: Net_Profit=${daily_net_profit:.2f} (total positions={total_positions})",
            MessageType.INFO)
        self.logger.do_log("---Summarizing trades---", MessageType.INFO)
        for index, row in trading_summary_df[trading_summary_df['total_net_profit'].notnull()].iterrows():
            self.logger.do_log(
                f" Pos: {row['side']} --> open_time={row['open']} close_time={row['close']} open_price=${row['price_open']:.2f} close_price=${row['price_close']:.2f} --> net_profit=${row['total_net_profit']}",
                MessageType.INFO)
        self.logger.do_log("--------------------", MessageType.INFO)


    def __log_scalping_trading_results__(self,summary_dto):

        max_cum_drawdown_percentage = summary_dto.max_cum_drawdown * 100
        self.logger.do_log(
            f"Results for algo {summary_dto.algo}: Net_Profit=${summary_dto.daily_net_profit:,.2f} (total positions={summary_dto.total_positions} Max Drawdown ={max_cum_drawdown_percentage:.2f}%)",
            MessageType.INFO)
        self.logger.do_log("---Summarizing trades---", MessageType.INFO)
        for index, row in summary_dto.trading_summary_df[summary_dto.trading_summary_df['total_net_profit'].notnull()].iterrows():
            self.logger.do_log(
                f" Pos: {row['side']} --> open_time={row['open']} close_time={row['close']} "
                f"open_price=${row['price_open']:,.2f} close_price=${row['price_close']:,.2f} --> "
                f"net_profit=${row['total_net_profit']:,.2f} ==> Final Portfolio = ${row['end_portfolio']:,.2f}"
                f" ( Pos. Profit=${row['pct_profit']}  Max. Cum. Drawdown=${row['max_drawdown']})",
                MessageType.INFO)

        self.logger.do_log("--------------------", MessageType.INFO)

    def __sliding_window__(self,df, timesteps):
        """
        Genera ventanas deslizantes de tamaño `timesteps + 1` en un DataFrame.

        Args:
        - df: DataFrame de entrada ordenado por tiempo.
        - timesteps: Número de pasos de tiempo a considerar (ventana de tamaño timesteps + 1).

        Yields:
        - Un DataFrame para cada ventana deslizante.
        """
        for start in range(len(df) - timesteps):
            yield df.iloc[start: start + timesteps + 1]

    # def __backtest_scalping_strategy__(self,rnn_predictions_df,portf_size, trade_comm,trading_algo,n_algo_params=[]):
    #
    #     #print(f"{rnn_predictions_df.head()}")
    #
    #     if tas(trading_algo)==tas.RAW_ALGO:
    #         daily_trading_backtester = RawAlgoDailyTradingBacktester()
    #         daily_net_profit, total_positions, max_daily_cum_drawdown, trading_summary_df = daily_trading_backtester.backtest_daily_predictions(
    #                                                                                         rnn_predictions_df, portf_size, trade_comm,n_algo_params)
    #
    #         return daily_net_profit, total_positions, max_daily_cum_drawdown, trading_summary_df
    #     elif tas(trading_algo)==tas.N_MIN_BUFFER_W_FLIP:
    #         raise Exception(f'Not implemented algo')
    #     elif tas(trading_algo)==tas.ONLY_SIGNAL_N_MIN_PLUS_MOV_AVG:
    #         raise Exception(f'Not implemented algo')
    #     else:
    #         raise Exception(f"NOT RECOGNIZED trading algo {trading_algo}")
    def __backtest_daily_strategy__(self,rnn_predictions_df,portf_summary):


        #print(f"{rnn_predictions_df.head()}")

        if tas(portf_summary.trading_algo)==tas.RAW_ALGO:
            daily_trading_backtester = RawAlgoDailyTradingBacktester()
            summary_dto, trading_summary_df = daily_trading_backtester.backtest_daily_predictions(rnn_predictions_df, portf_summary.portf_pos_size, portf_summary.trade_comm,portf_summary.n_algo_params)

            return summary_dto, trading_summary_df
        elif tas(portf_summary.trading_algo)==tas.N_MIN_BUFFER_W_FLIP:


            daily_trading_backtester = NMinBufferWFlipDailyTradingBacktester()
            summary_dto, trading_summary_df = daily_trading_backtester.backtest_daily_predictions(rnn_predictions_df, portf_summary.portf_pos_size, portf_summary.trade_comm,portf_summary.n_algo_params)

            return summary_dto, trading_summary_df
        elif tas(portf_summary.trading_algo)==tas.ONLY_SIGNAL_N_MIN_PLUS_MOV_AVG:
            daily_trading_backtester = OnlySignalNMinMovAvgBacktester()
            summary_dto, trading_summary_df = daily_trading_backtester.backtest_daily_predictions(rnn_predictions_df, portf_summary.portf_pos_size, portf_summary.trade_comm,portf_summary.n_algo_params)

            return summary_dto, trading_summary_df

        #InvSlopeBacktester

        else:
            raise Exception(f"NOT RECOGNIZED trading algo {portf_summary.trading_algo}")

    def __backtest_strategy__(self, series_df,indicator,portf_size,
                                    n_algo_params,portf_summary,
                                    etf_comp_dto_arr=None):

        if tas(portf_summary.trading_algo) == tas.RAW_INV_SLOPE:
            backtester=InvSlopeBacktester()
            return backtester.backtest(series_df, indicator, portf_size, n_algo_params,
                                       etf_comp_dto_arr=etf_comp_dto_arr)

        elif tas(portf_summary.trading_algo) == tas.RAW_DIRECT_SLOPE:
            backtester=DirectSlopeBacktester()
            return backtester.backtest(series_df, indicator, portf_size, n_algo_params,
                                       etf_comp_dto_arr=etf_comp_dto_arr)

        elif tas(portf_summary.trading_algo) == tas.RAW_ON_OFF_VALUE:
            backtester=OnOffBacktester()
            return backtester.backtest(series_df, indicator, portf_size, n_algo_params,
                                       etf_comp_dto_arr=etf_comp_dto_arr)

        elif tas(portf_summary.trading_algo) == tas.BUY_AND_HOLD:
            backtester=BuyAndHoldBacktester()
            return backtester.backtest(series_df, indicator, portf_size, n_algo_params,
                                       etf_comp_dto_arr=etf_comp_dto_arr)


        else:
            raise Exception(f"NOT RECOGNIZED trading algo {portf_summary.trading_algo}")

    # Returns prices of Symbol, starting on day, for N timesteps of X interval
    # Ex: If inteval is Day and start day 10/10/2023  and timesteps is 10
    # it returns daily prices from 10/10/2023 to 10/20/2023
    def __build_symbol_series__(self,symbol,day,timesteps,interval):
        start_day_timestamp = day
        start_period = day + pd.offsets.BDay(-1 * (timesteps + 2))  # we go back <timesteps> business days in time
        start_period_all_ind = start_period - timedelta(days=60)
        end_period = start_day_timestamp + timedelta(hours=23, minutes=59, seconds=59)

        symbol_int_series_df = self.data_set_builder.build_interval_series(symbol, start_period_all_ind,
                                                                           end_period, interval=interval,
                                                                           output_col=["symbol", "date", "open",
                                                                                       "high", "low", "close"])

        return symbol_int_series_df,start_period_all_ind,start_period,end_period


    def __get_business_days_in_range__(self,d_from,d_to):
        # Generate a date range between d_from and d_to
        all_days = pd.date_range(start=d_from, end=d_to)

        # Filter out weekends (Saturday = 5, Sunday = 6)
        business_days = [day for day in all_days if day.weekday() < 5]

        return  business_days

    def __eval_df_grouping__(self,test_series_df,grouping_unit,variables_csv):

        if grouping_unit is not None:
            test_series_df = self.__group_dataframe__(test_series_df, grouping_unit, variables_csv)
            self.logger.do_log(test_series_df.head(),MessageType.INFO)

        return test_series_df

    def __backtest_scalping__(self,algo,symbol,rnn_predictions_df,symbol_int_series_df):

        max_cum_drawdowns = []
        daily_profits = []
        total_net_profit = 0
        accum_positions = 0

        # TODO Run backtests and evaluate performance
        max_daily_drawdown = 0
        max_total_drawdown = 0

        ml_analyzer = MLModelAnalyzer(self.logger)

        #n-We pre-process the predictions --> chg action->Prediction
        rnn_predictions_df.rename(columns={'action': 'Prediction'}, inplace=True)
        predictions_dic = {}
        predictions_dic[algo] = rnn_predictions_df

        #n-We add the colum <symbol> with the price to use : the 'open' price
        symbol_df = symbol_int_series_df  # TODO see how to best apply only open numbers
        symbol_df[symbol] = symbol_df['open'] #

        #n-We run the backtest
        #TODO - Properly implmeent the backetsting algo
        raise Exception("Backtesting algo not implemented")

        self.logger.do_log(f"---Summarizing PORTFOLIO PERFORMANCE---", MessageType.INFO)
        self.logger.do_log(f" Total Net_Profit=${total_net_profit:.2f} Accum. Positions={accum_positions} Max. Daily Drawdown=${max_daily_drawdown:.2f} Max. Period Drawdown=${max_total_drawdown:.2f}",MessageType.INFO)

        return None


    def train_algos(self,series_csv,d_from,d_to,p_classif_key,algos_arr=None):

        try:
            series_df= self.data_set_builder.build_daily_series_classification(series_csv, d_from, d_to)
            mlAnalyzer=MLModelAnalyzer(self.logger)
            model_base_name,variables_def =MLModelAnalyzer.__build_model_base_name__(series_csv,d_from,d_to,p_classif_key)
            comp_df= mlAnalyzer.fit_and_evaluate(series_df, DataSetBuilder._CLASSIFICATION_COL,model_base_name,variables_def,
                                                 algos_arr=algos_arr)
            return comp_df

        except Exception as e:
            msg="CRITICAL ERROR processing model @train_algos:{}".format(str(e))
            self.logger.do_log(msg,MessageType.ERROR)
            raise Exception(msg)


    def backtest_neural_network_algo(self,symbol, variables_csv,d_from,d_to,model_to_use):
        try:

            #symbol_df = self.data_set_builder.build_series(symbol, d_from, d_to)
            test_series_df = self.data_set_builder.build_daily_series_classification(variables_csv, d_from, d_to)
            nn_trainer = NeuralNetworkModelTrainer(self.logger)

            nn_trainer.run_predictions(test_series_df,DataSetBuilder._CLASSIFICATION_COL,model_to_use)

            return None

        except Exception as e:
            msg = "CRITICAL ERROR processing model @backtest_neural_network_algo:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def train_neural_network(self,symbol, variables_csv,d_from,d_to,depth,learning_rate,epochs,model_output):
        try:
            series_df = self.data_set_builder.build_daily_series_classification(variables_csv, d_from, d_to)
            nn_trainer = NeuralNetworkModelTrainer(self.logger)
            nn_trainer.train_neural_network(series_df,variables_csv,DataSetBuilder._CLASSIFICATION_COL,depth,learning_rate,epochs,model_output)
            return None

        except Exception as e:
            msg = "CRITICAL ERROR processing model @train_neural_network:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def sliding_train_and_evaluate_ml_performance(self, symbol, series_csv, d_from, d_to, bias, last_trading_dict=None,
                                                  n_algo_param_dict={}):
        """
        Train and evaluate ML performance using a sliding window approach.
        Asynchronously invoke calculate_portfolio_metrics after each window to update results incrementally.

        Args:
            symbol (str): Symbol for trading.
            series_csv (str): Path to the CSV file with series data.
            d_from (datetime): Start date.
            d_to (datetime): End date.
            bias (float): Bias parameter.
            last_trading_dict (dict, optional): Last trading dictionary.
            n_algo_param_dict (dict): Parameters for the algorithm.

        Returns:
            list: List of summary dictionaries for each window.
        """
        summary_dict_arr = []
        summary_dict = None
        sliding_window_years = int(n_algo_param_dict["sliding_window_years"])
        sliding_window_months = int(n_algo_param_dict["sliding_window_months"])
        classif_key = n_algo_param_dict["classif_key"]
        init_portf_size = n_algo_param_dict["init_portf_size"]

        bias_df=None
        if bias is None or bias !=SlidingWindowStrategy.NONE.value:
            bias_df = self.data_set_builder.get_classification_df(bias,d_from,d_to)

        # Validate inputs
        if d_from > d_to:
            raise ValueError("d_from must be earlier than d_to.")
        if sliding_window_years < 0 or sliding_window_months <= 0:
            raise ValueError("sliding_window_years must be non-negative and sliding_window_months must be positive.")

        # Generate a single timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize the list to store the windows
        windows = []

        algos_arr=n_algo_param_dict["algos"].split(",") if n_algo_param_dict["algos"] is not None else None

        # Start with the first window
        curr_d_from = d_from
        # Calculate the end of the first window by adding the window duration
        curr_d_to = curr_d_from + relativedelta(years=sliding_window_years) - relativedelta(days=1)

        # Ensure the first window's end does not exceed d_to
        if curr_d_to > d_to:
            curr_d_to = d_to

        # Loop until the window's start date exceeds d_to
        while curr_d_from <= d_to:
            # Add the current window to the list
            windows.append((curr_d_from, curr_d_to))

            # 1- We train the algos in the train window
            self.train_algos(series_csv, curr_d_from, curr_d_to, classif_key,algos_arr=algos_arr)

            eval_d_from = curr_d_to + relativedelta(days=1)
            eval_d_to = eval_d_from + relativedelta(months=sliding_window_months) - relativedelta(days=1)

            if bias_df is not None:
                bias = self.data_set_builder.get_classification_in_classif_df(bias_df,eval_d_from)
                bias= bias  if bias is not None else SlidingWindowStrategy.NONE.value

            summary_dict = self.evaluate_trading_performance(
                symbol, series_csv, eval_d_from, eval_d_to,
                last_trading_dict=summary_dict,
                bias=bias,
                n_algo_param_dict=n_algo_param_dict,
                algos_arr=algos_arr
            )
            summary_dict_arr.append(summary_dict)

            # Asynchronously invoke calculate_portfolio_metrics after each window
            PortfolioSummaryAnalyzer.calculate_portfolio_metrics(summary_dict_arr, init_portf_size, eval_d_from, eval_d_to, timestamp)

            # Slide the window forward by sliding_window_months
            curr_d_from = curr_d_from + relativedelta(months=sliding_window_months)
            # Recalculate the end date for the new window
            curr_d_to = curr_d_from + relativedelta(years=sliding_window_years) - relativedelta(days=1)

            # Ensure the end date does not exceed d_to
            if curr_d_to > d_to:
                curr_d_to = d_to

            # Break if the window's end date is not advancing (i.e., we're at the end)
            if curr_d_to <= windows[-1][1]:
                break

        return summary_dict_arr




    def evaluate_trading_performance(self,symbol,series_csv,d_from,d_to,bias,last_trading_dict=None,
                                     n_algo_param_dict={},algos_arr=None):

        try:
            symbol_df = self.data_set_builder.build_daily_series_classification(symbol, d_from, d_to,add_classif_col=False)
            series_df = self.data_set_builder.build_daily_series_classification(series_csv, d_from, d_to,add_classif_col=False)

            # Given different frequencies of the data, we will the missing <interval> values with the last available data
            series_df = DataframeFiller.fill_missing_values(series_df)  # We fill missing values with the last one

            mlAnalyzer = MLModelAnalyzer(self.logger)
            portf_pos_dict = mlAnalyzer.evaluate_trading_performance_last_model(symbol_df,symbol,
                                                                                series_df, bias,
                                                                                last_trading_dict,
                                                                                n_algo_param_dict,
                                                                                algos_arr=algos_arr)

            backtester=IndicatorBasedTradingBacktester()

            summary_dict={}
            for algo in portf_pos_dict.keys():
                port_positions_arr=portf_pos_dict[algo]
                if len(port_positions_arr)>0:
                    summary= backtester.calculate_portfolio_performance_summary_extended(symbol,port_positions_arr,ref_date=d_from)
                    summary_dict[algo]=summary
                else:
                    summary=PortfSummary(symbol=symbol,p_portf_position_size=n_algo_param_dict["init_portf_size"],
                                         p_trade_comm=0,p_trading_algo=algo,
                                         p_algo_params=n_algo_param_dict["init_portf_size"],
                                         p_period=DateHandler.get_two_month_period_from_date(d_from),
                                         p_year=d_from.year)
                    summary_dict[algo] = summary

            return summary_dict

        except Exception as e:
            msg = "CRITICAL ERROR processing model @evaluate_trading_performance:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def run_predictions_last_model(self,series_csv,d_from,d_to):

        try:
            series_df = self.data_set_builder.build_daily_series_classification(series_csv, d_from, d_to,
                                                                                add_classif_col=False)
            mlAnalyzer = MLModelAnalyzer(self.logger)
            pred_dict = mlAnalyzer.run_predictions_last_model(series_df)
            return pred_dict

        except Exception as e:
            msg = "CRITICAL ERROR processing model @run_predictions_last_model:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def build_ARIMA(self,symbol, d_from, d_to, period=None):
        try:
            series_df = self.data_set_builder.build_daily_series_classification(symbol, d_from, d_to,
                                                                                add_classif_col=False)
            arima_Analyzer = ARIMAModelsAnalyzer(self.logger)
            dickey_fuller_test_dict=arima_Analyzer.build_ARIMA_model(series_df,symbol,period,True)
            return dickey_fuller_test_dict

        except Exception as e:
            msg = "CRITICAL ERROR processing model @build_ARIMA:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def eval_singe_indicator_algo(self,symbol,indicator, inv, d_from, d_to):
        try:
            series_df = self.data_set_builder.build_daily_series_classification(symbol, d_from, d_to,
                                                                                add_classif_col=False)

            indic_classif_list = self.date_range_classif_mgr.get_date_range_classification_values(indicator,d_from,d_to)
            indic_classif_df = pd.DataFrame([vars(classif) for classif in indic_classif_list])

            backtester = IndicatorBasedTradingBacktester()
            return backtester.backtest_indicator_based_strategy(symbol,series_df,indic_classif_df,inv)


        except Exception as e:
            # Obtiene la pila de llamadas
            tb = traceback.extract_tb(e.__traceback__)
            # Obtiene la última línea de la pila de llamadas
            file_name, line_number, func_name, line_code = tb[-1]
            msg = "CRITICAL ERROR processing model @eval_singe_indicator_algo:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def eval_ml_biased_algo(self,symbol, indicator,seriesCSV,d_from,d_to,inverted):
        try:
            series_df = self.data_set_builder.build_daily_series_classification(seriesCSV, d_from, d_to,
                                                                                add_classif_col=False)

            indic_classif_list = self.date_range_classif_mgr.get_date_range_classification_values(indicator, d_from,
                                                                                                  d_to)
            indic_classif_df = pd.DataFrame([vars(classif) for classif in indic_classif_list])

            mlAnalyzer = MLModelAnalyzer(self.logger)

            pred_dict = mlAnalyzer.run_predictions_last_model(series_df)

            backtester = IndicatorBasedTradingBacktester()

            return backtester.backtest_ML_indicator_biased_strategy(symbol,series_df,indic_classif_df,inverted,pred_dict)

        except Exception as e:
            # Obtiene la pila de llamadas
            tb = traceback.extract_tb(e.__traceback__)
            # Obtiene la última línea de la pila de llamadas
            file_name, line_number, func_name, line_code = tb[-1]
            msg = "CRITICAL ERROR processing model @eval_ml_biased_algo:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def predict_ARIMA(self,symbol, p,d,q,d_from,d_to, steps,period=None):
        try:
            series_df = self.data_set_builder.build_daily_series_classification(symbol, d_from, d_to,
                                                                                add_classif_col=False)
            arima_Analyzer = ARIMAModelsAnalyzer(self.logger)
            preds=arima_Analyzer.build_and__predict_ARIMA_model(series_df,symbol,p,d,q,period,steps)
            return preds

        except Exception as e:
            msg = "CRITICAL ERROR processing model @predict_ARIMA:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)


    def train_convolutional_neural_network(self,train_true_path,train_false_path,test_true_path,test_false_path,true_label,arch_file,padding,stride,iterations):
        try:
            LightLogger.do_log("Extracting images from true and false paths")

            handler = ImageHandler()

            LightLogger.do_log("Fetching images from true path {} and false path {} ".format(train_true_path, train_false_path))

            train_x, train_y, image_idx_train = handler.create_non_vect_sets(train_true_path, train_false_path, true_label, ".jpg")

            test_x, test_y, image_idx_test = handler.create_non_vect_sets(test_true_path, test_false_path, true_label, ".jpg")

            cnn= ConvolutionalNeuralNetwork()

            cnn.train_model(train_x, train_y,test_x, test_y,arch_file,padding,stride,iterations)


            #TODO finish everything
            #raise ("NOT FINISHED")

        except Exception as e:
            # Obtiene la pila de llamadas
            tb = traceback.extract_tb(e.__traceback__)
            # Obtiene la última línea de la pila de llamadas
            file_name, line_number, func_name, line_code = tb[-1]
            msg = "CRITICAL ERROR @train_convolutional_neural_network:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)


    def train_deep_neural_network(self,true_path,false_path,true_label,learning_rate=0.075,num_iterations=2500,
                                  arch_file=None, activ_file=None,output_file=None,step_size=200,lambd=0,use_He_init=False):


        try:

            LightLogger.do_log("Extracting images from true and false paths")

            handler = ImageHandler()

            offset=0
            end=False

            neural_network = DeepNeuralNetwork()

            parameters=None
            activations = neural_network.build_activations(activ_file)
            index=0

            while not end  : #we prepare the batches

                LightLogger.do_log("Fetching images for step {}".format(index))

                train_x,train_y,image_idx=handler.create_sets(true_path,false_path,true_label,".jpg",offset,step_size)

                if(len(train_x)< (step_size*2) ): #we have less than en enough rows for the available step --> we got to the end
                    end=True #this is the last run

                if len(train_x)>0:

                    LightLogger.do_log("Extracted {} train examples".format(len(image_idx)))

                    layers_dims=neural_network.build_layers_dims(len(train_x),arch_file) # len(train_x)--> ints in flattened vectors--> ex: 122880

                    LightLogger.do_log("Training Network with Learning Rate={} and num_iterations={}".format(learning_rate,num_iterations))
                    parameters, costs =neural_network.L_layer_model_train(train_x,train_y,layers_dims,activations,learning_rate=learning_rate,
                                                                          num_iterations=num_iterations,print_cost=True,parameters=parameters,loop=index,
                                                                          lambd=lambd,use_He_init=use_He_init)

                offset+=1
                index+=1


            #We persist everything

            if output_file is not None:
                neural_network.persist_parameters(parameters,activations,output_file)#parameters es el modelo!
                LightLogger.do_log("Model successfully persisted at {}".format(output_file))

                #test retreive parameters
                paramters2= neural_network.retrieve_parameters(output_file)
                LightLogger.do_log("Successfully retreived {} for testing".format(output_file))



        except Exception as e:
            # Obtiene la pila de llamadas
            tb = traceback.extract_tb(e.__traceback__)
            # Obtiene la última línea de la pila de llamadas
            file_name, line_number, func_name, line_code = tb[-1]
            msg = "CRITICAL ERROR @train_deep_neural_network:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)


    def test_deep_neural_network_model(self,true_path,false_path,true_label,output_file):
        try:
            LightLogger.do_log("Extracting model from file {}".format(output_file))

            neural_network = DeepNeuralNetwork()

            parameters = neural_network.retrieve_parameters(output_file)
            LightLogger.do_log("Successfully retrieved {} for testing".format(output_file))

            LightLogger.do_log("Extracting tests sets from true path={} and false paths={}".format(true_path,false_path))
            handler = ImageHandler()
            test_x, test_y, image_idx = handler.create_sets(true_path, false_path, true_label, ".jpg")

            accuracy=neural_network.L_layer_model_test(test_x,test_y,image_idx,parameters,parameters["activations"])

            LightLogger.do_log("Found an accuracy of {} for {} test instances".format(accuracy,test_x.shape[1]))

            return  accuracy


        except Exception as e:
            # Obtiene la pila de llamadas
            tb = traceback.extract_tb(e.__traceback__)
            # Obtiene la última línea de la pila de llamadas
            file_name, line_number, func_name, line_code = tb[-1]
            msg = "CRITICAL ERROR @test_deep_neural_network_model:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def process_test_scalping_LSTM(self,symbol,variables_csv, model_to_use, d_from,d_to,timesteps,portf_size, trade_comm,
                                trading_algo,interval=None,grouping_unit=None,n_algo_params=[],
                                   make_stationary=True,classif_threshold=0.5):
        try:

            self.logger.do_log(f"Initializing backest for symbol {symbol} from {d_from} to {d_to} (porft_size={portf_size} comm={trade_comm} )", MessageType.INFO)

            rnn_model_processer = DayTradingRNNModelCreator()
            rnn_predictions_df=None
            states = None
            for day in self.__get_business_days_in_range__(d_from,d_to):
                self.logger.do_log(f"Processing day {day}  )",MessageType.INFO)
                #1- We get the date range of prices to use to predict day <day> AND the symbol prices of that period
                symbol_int_series_df,start_period_all_ind,start_period,end_period= self.__build_symbol_series__(symbol,day,timesteps,interval)

                if symbol_int_series_df is None:
                    self.logger.do_log(f"Skipping day {day} because missing values (probable holiday!)",MessageType.WARNING)
                    continue

                #2- We get the variables (features) of that period
                variables_int_series_df = self.data_set_builder.build_interval_series(variables_csv,
                                                                                      start_period_all_ind, end_period,
                                                                                      interval=interval,
                                                                                      output_col=["symbol", "date",
                                                                                                  "open",
                                                                                                  "high", "low",
                                                                                                  "close"])
                #3- We merge everything in one single dataframe
                test_series_df = self.data_set_builder.merge_series(symbol_int_series_df, variables_int_series_df,
                                                                    "symbol", "date", symbol)

                #4- Given different frequencies of the data, we will the missing <interval> values with the last available data
                test_series_df = DataframeFiller.fill_missing_values(test_series_df)#We fill missing values with the last one

                #5- We evaluate if grouping must be applied
                test_series_df=self.__eval_df_grouping__(test_series_df,grouping_unit,variables_csv)

                #we filter the unecessary in all record to fetch all the indicators
                test_series_df = test_series_df[test_series_df['date'] >= start_period ]

                if(test_series_df["trading_symbol"].isna().any()):
                    continue # must be a holiday

                rnn_predictions_df_today,states = rnn_model_processer.test_LSTM_scalping(symbol, test_series_df,
                                                                                         model_to_use, timesteps,
                                                                                         prev_states=states,
                                                                                         make_stationary=make_stationary,
                                                                                         variables_csv=variables_csv,
                                                                                         threshold=classif_threshold)
                if rnn_predictions_df is None:
                    rnn_predictions_df = pd.DataFrame(columns=rnn_predictions_df_today.columns).astype(rnn_predictions_df_today.dtypes)
                rnn_predictions_df_today = rnn_predictions_df_today[rnn_predictions_df_today['date'] == day]
                rnn_predictions_df = pd.concat([rnn_predictions_df, rnn_predictions_df_today], ignore_index=True)

            #6- We have the predictions --> Lest run the backtest
            self.__backtest_scalping__("LSTM_SCALPING",symbol,rnn_predictions_df,symbol_int_series_df)


        except Exception as e:
            msg = "CRITICAL ERROR processing model @process_test_daily_LSTM:{}".format(str(e))
            traceback.print_exc()
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def __process_LSTM_day_single_run__(self,rnn_model_processer,model_to_use,test_series_df,timesteps,day,symbol,
                                        variables_csv=None):

        rnn_predictions_today_df, states = rnn_model_processer.test_LSTM_daily(symbol=symbol,
                                                                               test_series_df=test_series_df,
                                                                               model_to_use=model_to_use,
                                                                               timesteps=timesteps,
                                                                               price_to_use="close",
                                                                               variables_csv=variables_csv)
        self.logger.do_log(f"Predicting ALL TRADES for day {day} and symbol {symbol}",MessageType.INFO)
        return rnn_predictions_today_df


    def __process_LSTM_day_as_sliding_window__(self,rnn_model_processer,model_to_use,test_series_df,timesteps,day,symbol):
        rnn_predictions_today_df = None
        #preloaded_model = rnn_model_processer.preload_model(model_to_use=model_to_use)
        states=None
        for i, window in enumerate(self.__sliding_window__(test_series_df, timesteps)):
            test_series_curr_window_df = window.copy()
            min_timestamp = test_series_curr_window_df["date"].min()
            max_timestamp = test_series_curr_window_df["date"].max()
            self.logger.do_log(f"Processing Window from {min_timestamp} to {max_timestamp} for day {day} for symbol {symbol}",MessageType.INFO)

            if rnn_predictions_today_df is None:  # we initialize the summarization df
                rnn_predictions_today_df = pd.DataFrame(columns=test_series_curr_window_df.columns).astype(
                    test_series_curr_window_df.dtypes)

            rnn_predictions_curr_window_df,states = rnn_model_processer.test_LSTM_daily(symbol,
                                                                                           test_series_curr_window_df,
                                                                                           model_to_use=model_to_use,
                                                                                           timesteps=timesteps,
                                                                                           price_to_use="close",
                                                                                           prev_states=states)

            pred_action = rnn_predictions_curr_window_df["action"].iloc[0]
            curr_mkt_price = rnn_predictions_curr_window_df["trading_symbol_price"].iloc[0]
            end_of_timestamp = rnn_predictions_curr_window_df["date"].iloc[0]
            self.logger.do_log(
                f"Predicting at {pred_action} at end of {end_of_timestamp} at current mkt price={curr_mkt_price} for symbol {symbol}",
                MessageType.INFO)

            rnn_predictions_today_df = pd.concat([rnn_predictions_today_df, rnn_predictions_curr_window_df],ignore_index=True)
        return rnn_predictions_today_df


    #
    def __process_LSTM_day_as_fill_fake_data__(self,rnn_model_processer,model_to_use,test_series_df,timesteps,
                                               day,symbol,variables_csv):
        rnn_predictions_today_df = None
        #preloaded_model = rnn_model_processer.preload_model(model_to_use=model_to_use)
        states=None
        for i, window in enumerate(self.__sliding_window__(test_series_df, timesteps)):
            test_series_curr_window_df = window.copy()
            min_window_timestamp = test_series_curr_window_df["date"].min()
            max_window_timestamp = test_series_curr_window_df["date"].max()
            max_df_timestamp=test_series_df["date"].max()

            curr_min_df= test_series_df[test_series_df['date'] <= max_window_timestamp]
            self.logger.do_log(f"Processing Window from {min_window_timestamp} to {max_window_timestamp} for day {day} for symbol {symbol}", MessageType.INFO)

            # We fill with random values
            curr_min_df = RandomWalkGenerator.__fill_with_random_walk_values__(symbol, curr_min_df,
                                                                               max_window_timestamp,
                                                                               max_df_timestamp,
                                                                               variables_csv)
            if rnn_predictions_today_df is None:  # we initialize the summarization df
                rnn_predictions_today_df =  pd.DataFrame(columns=['trading_symbol', 'date', 'formatted_date', 'action'])

            rnn_predictions_curr_window_df,_ = rnn_model_processer.test_LSTM_daily(symbol, curr_min_df,
                                                                                      model_to_use=model_to_use,
                                                                                      timesteps=timesteps,
                                                                                      price_to_use="close",
                                                                                      prev_states=states,
                                                                                      variables_csv=variables_csv)

            pred_action = rnn_predictions_curr_window_df["action"].iloc[-1]
            curr_mkt_price = rnn_predictions_curr_window_df["trading_symbol_price"].iloc[-1]
            end_of_timestamp = rnn_predictions_curr_window_df["date"].iloc[-1]
            self.logger.do_log(f"{datetime.now()}-Predicting at {pred_action} at end of {end_of_timestamp} at current mkt price={curr_mkt_price} for symbol {symbol}",MessageType.INFO)

            rnn_predictions_today_df = pd.concat([rnn_predictions_today_df, rnn_predictions_curr_window_df.iloc[-1].to_frame().T],ignore_index=True)
        return rnn_predictions_today_df


    def __process_LSTM_day_as_cum_sliding_window__(self,rnn_model_processer,model_to_use,test_series_df,timesteps,
                                                   day,symbol,variables_csv):
        rnn_predictions_today_df = None
        #preloaded_model = rnn_model_processer.preload_model(model_to_use=model_to_use)
        states=None
        for i, window in enumerate(self.__sliding_window__(test_series_df, timesteps)):
            test_series_curr_window_df = window.copy()
            min_timestamp = test_series_curr_window_df["date"].min()
            max_timestamp = test_series_curr_window_df["date"].max()

            curr_min_df= test_series_df[test_series_df['date'] <= max_timestamp]
            self.logger.do_log(f"Processing Window from {min_timestamp} to {max_timestamp} for day {day} for symbol {symbol}",MessageType.INFO)



            if rnn_predictions_today_df is None:  # we initialize the summarization df
                rnn_predictions_today_df =  pd.DataFrame(columns=['trading_symbol', 'date', 'formatted_date', 'action'])

            rnn_predictions_curr_window_df = rnn_model_processer.test_stateful_LSTM(symbol=symbol,
                                                                                    test_series_df=curr_min_df,
                                                                                    model_to_use=model_to_use,
                                                                                    timesteps=timesteps,
                                                                                    price_to_use="close",
                                                                                    variables_csv=variables_csv)



            # rnn_predictions_curr_window_df,states = rnn_model_processer.test_daytrading_LSTM(symbol=symbol,
            #                                                                                  test_series_df=curr_min_df,
            #                                                                                  model_to_use=model_to_use,
            #                                                                                  timesteps=timesteps,
            #                                                                                  price_to_use="close",
            #                                                                                  prev_states=states)

            pred_action = rnn_predictions_curr_window_df["action"].iloc[-1]
            curr_mkt_price = rnn_predictions_curr_window_df["trading_symbol_price"].iloc[-1]
            end_of_timestamp = rnn_predictions_curr_window_df["date"].iloc[-1]
            self.logger.do_log(f"Predicting at {pred_action} at end of {end_of_timestamp} at current mkt price={curr_mkt_price} for symbol {symbol}",MessageType.INFO)

            rnn_predictions_today_df = pd.concat([rnn_predictions_today_df, rnn_predictions_curr_window_df.iloc[-1].to_frame().T],ignore_index=True)
        return rnn_predictions_today_df


    def __run_LSTM_daily_backtest__(self,day,rnn_predictions_today_df,portf_summary):
        self.logger.do_log(f"Backtesting all day {day} for symbol {portf_summary.symbol}", MessageType.INFO)
        summary_dto, trading_summary_df = self.__backtest_daily_strategy__(
                                                                                        rnn_predictions_today_df, portf_summary)

        #portf_summary.max_cum_drawdowns.append(max_daily_cum_drawdown)
        portf_summary.daily_profits.append(summary_dto.daily_net_profit)
        portf_summary.accum_positions += summary_dto.total_positions
        self.__log_day_trading_results__(day, summary_dto.daily_net_profit, summary_dto.total_positions, trading_summary_df)
        self.logger.do_log(f"Moving to the next business day from {day}", MessageType.INFO)
        return  summary_dto.daily_net_profit

    def process_test_daily_LSTM(self,symbol,variables_csv, model_to_use, d_from,d_to,timesteps,portf_size, trade_comm,
                                trading_algo,interval=None,grouping_unit=None,n_algo_params=[],
                                use_sliding_window=None,make_stationary=True):
        try:

            self.logger.do_log(f"Initializing backest for symbol {symbol} from {d_from} to {d_to} (porft_size={portf_size} comm={trade_comm} )", MessageType.INFO)
            # Generate a date range between d_from and d_to
            all_days = pd.date_range(start=d_from, end=d_to)

            # Filter out weekends (Saturday = 5, Sunday = 6)
            business_days = [day for day in all_days if day.weekday() < 5]
            rnn_model_processer = DayTradingRNNModelCreator()

            portf_summary = PortfSummary(symbol,portf_size,p_trade_comm= trade_comm,
                                         p_trading_algo= trading_algo,p_algo_params=n_algo_params)
            for day in business_days:
                self.logger.do_log(f"Processing day {day}  )",MessageType.INFO)

                start_timestamp=day if (d_from.hour == 0 and d_from.minute == 0 and d_to.second == 0) else d_from
                end_timestamp= (start_timestamp + timedelta(hours=23, minutes=59, seconds=59)) if (d_to.hour == 0 and d_to.minute == 0 and d_to.second == 0) else d_to

                symbol_int_series_df = self.data_set_builder.build_interval_series(portf_summary.symbol, start_timestamp,
                                                                                   end_timestamp, interval=interval,
                                                                                   output_col=["symbol", "date", "open",
                                                                                               "high", "low", "close"])

                if symbol_int_series_df is None or symbol_int_series_df.shape[0] <= 1:
                    self.logger.do_log(f"Skipping day {day} because missing values (probable holiday!)",MessageType.WARNING)
                    continue


                variables_int_series_df = self.data_set_builder.build_interval_series(variables_csv, start_timestamp,
                                                                                      end_timestamp, interval=interval,
                                                                                      output_col=["symbol", "date",
                                                                                                  "open",
                                                                                                  "high", "low",
                                                                                                  "close"])

                test_series_df = self.data_set_builder.merge_series(symbol_int_series_df, variables_int_series_df,
                                                                    "symbol", "date", portf_summary.symbol)

                if grouping_unit is not None:
                    test_series_df = self.__group_dataframe__(test_series_df, grouping_unit,variables_csv)
                    print((test_series_df.head()))

                #preprocess before the predictions
                test_series_df = test_series_df.dropna(subset=[portf_summary.symbol])

                if sws(use_sliding_window)==sws.CUT_INPUT_DF :#slower, but we only pass every <timestemps> records to make sure of the accuracy of the prediction
                    #rnn_predictions_today_df=self.__process_LSTM_day_as_sliding_window__(rnn_model_processer,model_to_use,test_series_df,timesteps,day,symbol)
                    rnn_predictions_today_df = self.__process_LSTM_day_as_cum_sliding_window__(rnn_model_processer,model_to_use, test_series_df,timesteps, day, portf_summary.symbol,variables_csv=variables_csv)
                elif sws(use_sliding_window)==sws.NONE:#faster, but we pass the dataframe with ALL the daily records, which might create some look ahead bias
                    rnn_predictions_today_df=self.__process_LSTM_day_single_run__(rnn_model_processer,model_to_use,test_series_df,timesteps,day,portf_summary.symbol,variables_csv=variables_csv)
                elif sws(use_sliding_window)==sws.GET_FAKE_DATA:
                    rnn_predictions_today_df = self.__process_LSTM_day_as_fill_fake_data__(rnn_model_processer,model_to_use,test_series_df,timesteps, day, portf_summary.symbol,variables_csv= variables_csv)
                else:
                    raise  Exception(f"Could not find use_sliding_window param {use_sliding_window}")

                portf_summary.total_net_profit+=self.__run_LSTM_daily_backtest__(day, rnn_predictions_today_df,
                                                                                portf_summary)

                new_position_summary=portf_summary.calculate_last_portf_position_summary(day)#We append the day summary

                self.logger.do_log(f"PERFORMANCE--> day {new_position_summary.day}: Profit:{new_position_summary.day_nom_profit:2f} ( drawdown={new_position_summary.day_drawdown:2f})", MessageType.INFO)



            portf_summary.update_max_drawdown()
            self.logger.do_log(f"---Re Displaying Daily PERFORMANCE---", MessageType.INFO)
            for pos_summary in portf_summary.portf_pos_summary:
                self.logger.do_log(f" day {pos_summary.day}: Profit:{pos_summary.day_nom_profit:2f} ( drawdown={pos_summary.day_drawdown:2f})", MessageType.INFO)
            self.logger.do_log(f"---------------------------------------", MessageType.INFO)
            self.logger.do_log(f"---Summarizing PORTFOLIO PERFORMANCE---", MessageType.INFO)
            self.logger.do_log(
                f" Total Net_Profit=${portf_summary.total_net_profit:.2f} Accum. Positions={portf_summary.accum_positions} Max. Daily Drawdown=${portf_summary.max_daily_drawdown:.2f} Max. Period Drawdown=${portf_summary.max_drawdown:.2f}",
                MessageType.INFO)




        except Exception as e:
            msg = "CRITICAL ERROR processing model @process_test_daily_LSTM:{}".format(str(e))
            traceback.print_exc()
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def process_train_LSTM(self,symbol,variables_csv,d_from,d_to,model_output,classif_key,
                           epochs,timestamps,n_neurons,learning_rate,reg_rate, dropout_rate,
                           interval=DataSetBuilder._1_MIN_INTERVAL,clipping_rate=None,threshold_stop=None,grouping_unit=None,
                           grouping_classif_criteria=None,
                           group_as_mov_avg=False,grouping_mov_avg_unit=20,
                           batch_size=1,inner_activation=None, make_stationary=False):
        try:

            range_clasifs=None
            if interval==DataSetBuilder._1_MIN_INTERVAL:
                range_clasifs=self.timestamp_range_classif_mgr.get_timestamp_range_classification_values(classif_key, d_from, d_to)
            elif interval==DataSetBuilder._1_DAY_INTERVAL:
                range_clasifs = self.date_range_classif_mgr.get_date_range_classification_values(
                    classif_key, d_from, d_to)
            else:
                raise Exception(f'Invalid interval at process_train_LSTM:{interval}')


            symbol_min_series_df= self.data_set_builder.build_interval_series(symbol, d_from, d_to, interval=interval,
                                                                              output_col=["symbol", "date", "open",
                                                                                          "high", "low", "close"])
            symbol_min_series_df = self.data_set_builder.build_minute_series_classification(range_clasifs,
                                                                                            symbol_min_series_df,
                                                                                            classif_col_name=classif_key,
                                                                                            not_found_clasif="FLAT")

            variables_min_series_df= self.data_set_builder.build_interval_series(variables_csv, d_from, d_to,
                                                                                 interval=interval,
                                                                                 output_col=["symbol", "date", "open",
                                                                                             "high", "low", "close"])

            training_series_df= self.data_set_builder.merge_series(symbol_min_series_df, variables_min_series_df,
                                                                   "symbol", "date", symbol)

            if group_as_mov_avg:
                training_series_df=self.data_set_builder.group_as_mov_avgs(training_series_df,variables_csv,grouping_mov_avg_unit)


            DataframePrinter.print_dataframe_head_values_w_time(variables_min_series_df, "symbol", variables_csv, 10,"date","10:30:00")
            if grouping_unit is not None:

                training_series_df= self.__group_dataframe__(training_series_df, grouping_unit,
                                                             variables_csv,grouping_classif_criteria, classif_key)
                DataframePrinter.print_data_farme_head(training_series_df,10)



            rnn_model_trainer= DayTradingRNNModelCreator()

            if interval==DataSetBuilder._1_DAY_INTERVAL:
                rnn_model_trainer.train_LSTM_scalping(training_series_df, model_output, symbol_min_series_df, classif_key,
                                                      epochs, timestamps, n_neurons, learning_rate, reg_rate, dropout_rate,
                                                      variables_csv, clipping_rate, threshold_stop,
                                                      make_stationary=make_stationary, inner_activation=inner_activation,
                                                      batch_size=batch_size)
            elif interval==DataSetBuilder._1_MIN_INTERVAL:
                rnn_model_trainer.train_LSTM_daily_stateful(training_series_df= training_series_df,
                                                           model_output= model_output,
                                                           symbol=symbol,
                                                           variables_csv=variables_csv,
                                                           classif_key=classif_key,
                                                           epochs= epochs,
                                                           timestamps= timestamps,
                                                           n_neurons= n_neurons,
                                                           learning_rate= learning_rate,
                                                           reg_rate= reg_rate,
                                                           dropout_rate= dropout_rate,
                                                           clipping_rate= clipping_rate,
                                                           inner_activation=inner_activation,
                                                           batch_size=batch_size)

            #pd.set_option('display.max_columns', None)
            #print(training_series_df.head())
            #pd.reset_option('display.max_columns')

            return None

        except Exception as e:
            msg = "CRITICAL ERROR processing model @train_neural_network:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)


    def process_daily_candles_graph(self,symbol, date, interval,mov_avg_unit):
        start_of_day = datetime(date.year, date.month, date.day)
        end_of_day = start_of_day + timedelta(hours=23, minutes=59, seconds=59)

        prices_df= self.data_set_builder.build_interval_series(symbol, start_of_day, end_of_day,
                                                               interval=DataSetBuilder._1_MIN_INTERVAL)

        GraphBuilder.build_candles_graph(prices_df,mov_avg_unit =mov_avg_unit)

        return  None


    def process_indicator_candles_graph(self,symbol, d_from,d_to, interval,mov_avg_unit):
        start_of_day = datetime(d_from.year, d_from.month, d_from.day)
        end_of_day = datetime(d_to.year, d_to.month, d_to.day) + timedelta(hours=23, minutes=59, seconds=59)

        prices_df= self.data_set_builder.build_interval_series(symbol, start_of_day, end_of_day,
                                                               interval=interval)

        GraphBuilder.build_candles_graph(prices_df,mov_avg_unit =mov_avg_unit)

        return  None


    def process_backtest_slope_model(self,symbol,model_candle,d_from,d_to,portf_size,trading_algo,
                                     n_algo_params):
        start_of_day = datetime(d_from.year, d_from.month, d_from.day)
        end_of_day = d_to + timedelta(hours=23, minutes=59, seconds=59)

        #1- The trading symbol DF
        symbol_series_df = self.data_set_builder.build_interval_series(symbol, start_of_day, end_of_day,
                                                                       interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                       output_col=["symbol", "date", "open",
                                                                                       "high", "low", "close"])

        #2- The explanatory variables DF
        variables_series_df = self.data_set_builder.build_interval_series(model_candle, d_from, d_to,
                                                                          interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                          output_col=["symbol", "date", "open",
                                                                                          "high", "low", "close"])

        #3- We merge them
        training_series_df = self.data_set_builder.merge_series(symbol_series_df, variables_series_df, "symbol", "date",
                                                                symbol)

        #4- We drop weekends and holidays
        training_series_df = training_series_df.dropna(subset=[f"close_{symbol}"])

        portf_summary = PortfSummary(symbol, portf_size, p_trade_comm=0,
                                     p_trading_algo=trading_algo, p_algo_params=n_algo_params)

        summary_dto,portf_positions= self.__backtest_strategy__(training_series_df, model_candle, portf_size, n_algo_params,
                                                portf_summary=portf_summary)


        self.__log_scalping_trading_results__(summary_dto)
        return summary_dto


    def process_ARIMA_predictions(self, series_key,d_from,d_to,algo_params):
        start_of_day = datetime(d_from.year, d_from.month, d_from.day)
        end_of_day = d_to + timedelta(hours=23, minutes=59, seconds=59)

        # 1- The trading symbols DF
        series_df = self.data_set_builder.build_interval_series(series_key, start_of_day, end_of_day,
                                                                           interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                           output_col=["symbol", "date", "open",
                                                                                       "high", "low", "close"])

        #2 - Predict ARIMA
        arima_Analyzer = ARIMAModelsAnalyzer(self.logger)
        p=algo_params["p"]
        d=algo_params["d"]
        q=algo_params["q"]
        s = algo_params["s"]
        period=algo_params["period"]
        step=algo_params["step"]

        preds=None
        if s is  None: #ARIMA
            preds = arima_Analyzer.build_and__predict_ARIMA_model(series_df,f"close",
                                                                 p, d, q, period, step)
        else:
            preds = arima_Analyzer.build_and_predict_SARIMA_model(series_df, f"close",
                                                                  p, d, q,p,d,q,s,period, step)

        return preds


    def process_create_sinthetic_indicator_logic(self,comp_path,model_candle,d_from,d_to,algo_params):
        start_of_day = datetime(d_from.year, d_from.month, d_from.day)
        end_of_day = d_to + timedelta(hours=23, minutes=59, seconds=59)

        #0- We extract the ETF composition info
        indicators_csv = self.data_set_builder.extract_series_csv_from_etf_file(comp_path, 0)
        indicator_types = self.data_set_builder.extract_series_csv_from_etf_file(comp_path, 1)

        #1- The trading symbols DF
        indicators_series_df = self.data_set_builder.build_interval_series(indicators_csv, start_of_day, end_of_day,
                                                                        interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                        output_col=["symbol", "date", "open",
                                                                                       "high", "low", "close"])

        #2- We have one dataframe per indicator
        indicators_series_df=self.data_set_builder.privot_and_merge_dataframes(indicators_series_df)
        indicators_series_df["indicator"]=model_candle

        #3- Dto with indicator-type arrays
        indicator_type_arr= IndicatorTypeDTO.load_indicator_type_data(indicators_csv,indicator_types)

        #4- Build and create the indicator
        ind_creator = SintheticIndicatorCreator(self.logger)
        indicators_series_df=ind_creator.build_sinthetic_indicator(indicators_series_df,indicator_type_arr,algo_params)

        #4- We persist the newly created indicator
        self.data_set_builder.persist_sinthetic_indicator(indicators_series_df)

        self.logger.do_log(f"Successfully created {len(indicators_series_df)} records for indicator {model_candle}",MessageType.INFO)

        return  indicators_series_df


    def process_backtest_slope_model_on_custom_etf(self,etf_path,model_candle,d_from,d_to,portf_size,trading_algo,
                                     n_algo_params):
        start_of_day = datetime(d_from.year, d_from.month, d_from.day)
        end_of_day = d_to + timedelta(hours=23, minutes=59, seconds=59)

        #0- We extract the ETF composition info
        symbols_csv=self.data_set_builder.extract_series_csv_from_etf_file(etf_path,1)
        etf_comp_dto_arr = ETFLogicManager.__extract_etf_composition__(etf_path, 1, 0)

        #1- The trading symbols DF
        symbols_series_df = self.data_set_builder.build_interval_series(symbols_csv, start_of_day, end_of_day,
                                                                        interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                        output_col=["symbol", "date", "open",
                                                                                       "high", "low", "close"])

        #2- We have one datafraem per symbol in the ETF
        symbols_series_df_arr=self.data_set_builder.split_dataframe_by_symbol(symbols_series_df,"symbol")


        #3- The explanatory variables DF
        variables_series_df = self.data_set_builder.build_interval_series(model_candle, d_from, d_to,
                                                                          interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                          output_col=["symbol", "date", "open",
                                                                                          "high", "low", "close"])
        variables_series_df=self.data_set_builder.shift_dates(variables_series_df,n_algo_params)

        #4- We merge them
        training_series_df_arr={}
        for symbol in  symbols_series_df_arr:
            symbols_series_df= symbols_series_df_arr[symbol]
            training_series_df = self.data_set_builder.merge_series(symbols_series_df,
                                                                    variables_series_df,
                                                                    "symbol","date",
                                                                    symbol)
            training_series_df_arr[symbol]=training_series_df


        #5-We merge all the training_series_df
        merged_training_series_df=None
        for symbol in   training_series_df_arr:
            training_series_df=training_series_df_arr[symbol]
            if merged_training_series_df is None:
                merged_training_series_df=training_series_df
            else:
                merged_training_series_df=self.data_set_builder.merge_dataframes(training_series_df,merged_training_series_df,
                                                                          "date")
        merged_training_series_df = merged_training_series_df.drop(columns=['trading_symbol'], errors='ignore')


        #6- We run the backtest
        portf_summary = PortfSummary(symbols_csv, portf_size, p_trade_comm=0,p_trading_algo=trading_algo, p_algo_params=n_algo_params)
        summ_dto,portf_positions= self.__backtest_strategy__(merged_training_series_df, model_candle, portf_size, n_algo_params,
                                             portf_summary=portf_summary, etf_comp_dto_arr=etf_comp_dto_arr)


        self.__log_scalping_trading_results__(summ_dto)
        return summ_dto,portf_positions


    def model_custom_etf(self,weights_csv,symbols_csv,start_of_day, end_of_day):

        # 0- We extract the ETF composition info
        etf_comp_dto_arr=ETFLogicManager.__extract_etf_composition_from_csv__(weights_csv,symbols_csv)

        # 1- The trading symbols DF
        symbols_series_df = self.data_set_builder.build_interval_series(symbols_csv, start_of_day, end_of_day,
                                                                        interval=DataSetBuilder._1_DAY_INTERVAL,
                                                                        output_col=["symbol", "date", "open",
                                                                                    "high", "low", "close"])

        # 2- We have one datafraem per symbol in the ETF
        symbols_series_df_arr = self.data_set_builder.split_dataframe_by_symbol(symbols_series_df, "symbol")


        # 3-We merge all the training_series_df
        merged_training_series_df=self.data_set_builder.merge_multiple_series(symbols_series_df_arr.values(),date_col="date")

        #6- We run the backtest
        portf_size=ParametersKeys.STANDARD_PORTFOLIO.value
        n_algo_param_dict={}
        n_algo_param_dict[ParametersKeys.TRADE_COMM_NOM_KEY.value] = 0  # No comissions
        portf_summary = PortfSummary(symbols_csv, portf_size, p_trade_comm=0,
                                     p_trading_algo=TradingAlgoStrategy.BUY_AND_HOLD.name, p_algo_params={})
        summary_dto,portf_pos= self.__backtest_strategy__(merged_training_series_df, TradingAlgoStrategy.BUY_AND_HOLD.name,
                                             portf_size,n_algo_params= n_algo_param_dict,
                                             portf_summary=portf_summary,
                                             etf_comp_dto_arr=etf_comp_dto_arr)


        #4- We bould the custom ETF series
        if len(portf_pos)<=0:
            raise Exception("CRITICAL ERROR building MTMs portfolio! ")
        return portf_pos[0].detailed_MTMS


