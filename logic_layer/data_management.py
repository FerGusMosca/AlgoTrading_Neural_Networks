import sys
import traceback
from datetime import timedelta, datetime

from common.util.dataframe_printer import DataframePrinter
from common.util.graph_builder import GraphBuilder
from common.util.image_handler import ImageHandler
from common.util.light_logger import LightLogger
from data_access_layer.date_range_classification_manager import DateRangeClassificationManager
from data_access_layer.economic_series_manager import EconomicSeriesManager
from data_access_layer.timestamp_classification_manager import TimestampClassificationManager

from framework.common.logger.message_type import MessageType
from logic_layer.ARIMA_models_analyzer import ARIMAModelsAnalyzer
from logic_layer.convolutional_neural_netowrk import ConvolutionalNeuralNetwork
from logic_layer.n_min_buffer_w_flip_daily_trading_backtester import NMinBufferWFlipDailyTradingBacktester
from logic_layer.raw_algo_daily_trading_backtester import RawAlgoDailyTradingBacktester
from logic_layer.data_set_builder import DataSetBuilder
from logic_layer.deep_neural_network import DeepNeuralNetwork
from logic_layer.indicator_based_trading_backtester import IndicatorBasedTradingBacktester
from logic_layer.ml_models_analyzer import MLModelAnalyzer
import pandas as pd
import numpy as np

from logic_layer.neural_network_models_trainer import NeuralNetworkModelTrainer
from logic_layer.daytrading_RNN_model_creator import DayTradingRNNModelCreator

# Global variables
GO_FIRST = "GO_FIRST"
GO_HIGHEST_COUNT = "GO_HIGHEST_COUNT"
GO_FLAT_ON_DIFF = "GO_FLAT_ON_DIFF"

_TRADING_ALGO_RAW_ALGO="RAW_ALGO"
_TRADING_ALGO_N_MIN_BUFFER_W_FLIP="N_MIN_BUFFER_W_FLIP"
_TRADING_ALGO_ONLY_SIGNAL_N_MIN_PLUS_MOV_AVG="ONLY_SIGNAL_N_MIN_+_MOV_AVG"

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
        if grouping_classif_criteria == GO_FIRST:
            # Return the first classification in the group
            return classifications.iloc[0]
        elif grouping_classif_criteria == GO_HIGHEST_COUNT:
            # Return the most frequent classification
            return classifications.mode().iloc[0]
        elif grouping_classif_criteria == GO_FLAT_ON_DIFF:
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


    def __backtest_strategy__(self,rnn_predictions_df,portf_size, trade_comm,trading_algo):


        #print(f"{rnn_predictions_df.head()}")

        if trading_algo==_TRADING_ALGO_RAW_ALGO:
            daily_trading_backtester = RawAlgoDailyTradingBacktester()
            daily_net_profit, total_positions, max_daily_cum_drawdown, trading_summary_df = daily_trading_backtester.backtest_daily_predictions(
                                                                                            rnn_predictions_df, portf_size, trade_comm)

            return daily_net_profit, total_positions, max_daily_cum_drawdown, trading_summary_df
        elif trading_algo==_TRADING_ALGO_N_MIN_BUFFER_W_FLIP:
            daily_trading_backtester = NMinBufferWFlipDailyTradingBacktester()
            daily_net_profit, total_positions, max_daily_cum_drawdown, trading_summary_df = daily_trading_backtester.backtest_daily_predictions(
                rnn_predictions_df, portf_size, trade_comm)

            return daily_net_profit, total_positions, max_daily_cum_drawdown, trading_summary_df
        elif trading_algo==_TRADING_ALGO_ONLY_SIGNAL_N_MIN_PLUS_MOV_AVG:
            raise Exception(f"Implement trading algo: {_TRADING_ALGO_ONLY_SIGNAL_N_MIN_PLUS_MOV_AVG}")
        else:
            raise Exception(f"NOT RECOGNIZED trading algo {trading_algo}")


    def __calculate_max_total_drawdown__(self,daily_profits):
        daily_trading_backtester = RawAlgoDailyTradingBacktester()
        return daily_trading_backtester.calculate_max_total_drawdown(daily_profits)

    def train_algos(self,series_csv,d_from,d_to):

        try:
            series_df= self.data_set_builder.build_daily_series_classification(series_csv, d_from, d_to)
            mlAnalyzer=MLModelAnalyzer(self.logger)
            comp_df= mlAnalyzer.fit_and_evaluate(series_df, DataSetBuilder._CLASSIFICATION_COL)
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

    def evaluate_trading_performance(self,symbol,series_csv,d_from,d_to,bias,last_trading_dict=None):

        try:
            symbol_df = self.data_set_builder.build_daily_series_classification(symbol, d_from, d_to)
            series_df = self.data_set_builder.build_daily_series_classification(series_csv, d_from, d_to)
            mlAnalyzer = MLModelAnalyzer(self.logger)
            portf_pos_dict = mlAnalyzer.evaluate_trading_performance_last_model(symbol_df,symbol,series_df, bias,last_trading_dict)

            backtester=IndicatorBasedTradingBacktester()

            summary_dict={}
            for algo in portf_pos_dict.keys():
                port_positions_arr=portf_pos_dict[algo]
                summary= backtester.calculate_portfolio_performance(symbol,port_positions_arr)
                summary_dict[algo]=summary

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

    def build_ARIMA(self,symbol, period, d_from, d_to):
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

    def predict_ARIMA(self,symbol, p,d,q,d_from,d_to,period, steps):
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



    def process_test_daily_LSTM(self,symbol,variables_csv, model_to_use, d_from,d_to,timesteps,portf_size, trade_comm,
                                trading_algo,grouping_unit=None):
        try:

            self.logger.do_log(f"Initializing backest for symbol {symbol} from {d_from} to {d_to} (porft_size={portf_size} comm={trade_comm} )", MessageType.INFO)
            # Generate a date range between d_from and d_to
            all_days = pd.date_range(start=d_from, end=d_to)

            # Filter out weekends (Saturday = 5, Sunday = 6)
            business_days = [day for day in all_days if day.weekday() < 5]
            rnn_model_processer = DayTradingRNNModelCreator()

            max_cum_drawdowns=[]
            daily_profits=[]
            total_net_profit=0
            accum_positions=0

            for day in business_days:
                self.logger.do_log(
                    f"Processing day {day}  )",
                    MessageType.INFO)

                start_timestamp=day
                end_timestamp= start_timestamp + timedelta(hours=23, minutes=59, seconds=59)

                symbol_min_series_df = self.data_set_builder.build_minute_series(symbol,
                                                                                 start_timestamp, end_timestamp,
                                                                                 output_col=["symbol", "date", "open",
                                                                                             "high", "low", "close"])

                if symbol_min_series_df is None:
                    self.logger.do_log(f"Skipping day {day} because missing values (probable holiday!)",MessageType.WARNING)
                    continue


                variables_min_series_df = self.data_set_builder.build_minute_series(variables_csv, start_timestamp, end_timestamp,
                                                                                    output_col=["symbol", "date", "open",
                                                                                                "high", "low", "close"])

                test_series_df = self.data_set_builder.merge_minute_series(symbol_min_series_df,
                                                                           variables_min_series_df, "symbol", "date",
                                                                           symbol)

                if grouping_unit is not None:
                    test_series_df = self.__group_dataframe__(test_series_df, grouping_unit,variables_csv)
                    print((test_series_df.head()))


                rnn_predictions_df=rnn_model_processer.test_daytrading_LSTM(symbol, test_series_df, model_to_use, timesteps)

                daily_net_profit, total_positions, max_daily_cum_drawdown,trading_summary_df= self.__backtest_strategy__(rnn_predictions_df, portf_size, trade_comm,trading_algo)
                max_cum_drawdowns.append(max_daily_cum_drawdown)
                daily_profits.append(daily_net_profit)
                total_net_profit+=daily_net_profit
                accum_positions+=total_positions

                self.logger.do_log(f"Results for day {day}: Net_Profit=${daily_net_profit:.2f} (total positions={total_positions})", MessageType.INFO)
                self.logger.do_log("---Summarizing trades---",MessageType.INFO)
                for index, row in trading_summary_df.iterrows():
                    self.logger.do_log(f" Pos: {row['side']} --> open_time={row['open']} close_time={row['close']} open_price=${row['price_open']:.2f} close_price=${row['price_close']:.2f} --> net_profit=${row['total_net_profit']}",MessageType.INFO)
                self.logger.do_log("--------------------",MessageType.INFO)


            max_daily_drawdown=min(max_cum_drawdowns) if len(max_cum_drawdowns) is None else 0
            max_total_drawdown= self.__calculate_max_total_drawdown__(daily_profits)
            self.logger.do_log(f"---Summarizing PORTFOLIO PERFORMANCE---",MessageType.INFO)
            self.logger.do_log(f" Total Net_Profit=${total_net_profit:.2f} Accum. Positions={accum_positions} Max. Daily Drawdown=${max_daily_drawdown:.2f} Max. Period Drawdown=${max_total_drawdown:.2f}", MessageType.INFO)

        except Exception as e:
            msg = "CRITICAL ERROR processing model @process_test_daily_LSTM:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)

    def process_train_LSTM(self,symbol,variables_csv,d_from,d_to,model_output,classif_key,
                           epochs,timestamps,n_neurons,learning_rate,reg_rate, dropout_rate,
                           clipping_rate=None,accuracy_stop=None,grouping_unit=None,
                           grouping_classif_criteria=None,
                           group_as_mov_avg=False,grouping_mov_avg_unit=20):
        try:
            timestamp_range_clasifs=self.timestamp_range_classif_mgr.get_timestamp_range_classification_values(classif_key,d_from,d_to)

            symbol_min_series_df= self.data_set_builder.build_minute_series(symbol, d_from, d_to, output_col=["symbol", "date", "open", "high", "low", "close"])
            symbol_min_series_df = self.data_set_builder.build_minute_series_classification(timestamp_range_clasifs,
                                                                                            symbol_min_series_df,
                                                                                            classif_col_name=classif_key,
                                                                                            not_found_clasif="FLAT")

            variables_min_series_df=self.data_set_builder.build_minute_series(variables_csv, d_from, d_to, output_col=["symbol", "date", "open", "high", "low", "close"])

            training_series_df= self.data_set_builder.merge_minute_series(symbol_min_series_df,variables_min_series_df,"symbol","date",symbol)

            if group_as_mov_avg:
                training_series_df=self.data_set_builder.group_as_mov_avgs(training_series_df,variables_csv,grouping_mov_avg_unit)


            DataframePrinter.print_dataframe_head_values_w_time(variables_min_series_df, "symbol", variables_csv, 10,"date","10:30:00")
            if grouping_unit is not None:

                training_series_df= self.__group_dataframe__(training_series_df, grouping_unit,
                                                             variables_csv,grouping_classif_criteria, classif_key)
                DataframePrinter.print_data_farme_head(training_series_df,10)



            rnn_model_trainer= DayTradingRNNModelCreator()

            rnn_model_trainer.train_daytrading_LSTM(training_series_df, model_output, symbol_min_series_df, classif_key,
                                                    epochs, timestamps, n_neurons, learning_rate, reg_rate,
                                                    dropout_rate,clipping_rate,accuracy_stop)

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

        prices_df= self.data_set_builder.build_minute_series(symbol,start_of_day,end_of_day)

        GraphBuilder.build_candles_graph(prices_df,mov_avg_unit =mov_avg_unit)

        return  None


