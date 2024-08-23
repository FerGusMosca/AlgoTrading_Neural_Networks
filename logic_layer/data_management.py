import sys
import traceback
from datetime import timedelta, datetime

from common.util.graph_builder import GraphBuilder
from common.util.image_handler import ImageHandler
from common.util.light_logger import LightLogger
from data_access_layer.date_range_classification_manager import DateRangeClassificationManager
from data_access_layer.economic_series_manager import EconomicSeriesManager
from data_access_layer.timestamp_classification_manager import TimestampClassificationManager

from framework.common.logger.message_type import MessageType
from logic_layer.ARIMA_models_analyzer import ARIMAModelsAnalyzer
from logic_layer.convolutional_neural_netowrk import ConvolutionalNeuralNetwork
from logic_layer.data_set_builder import DataSetBuilder
from logic_layer.deep_neural_network import DeepNeuralNetwork
from logic_layer.indicator_based_trading_backtester import IndicatorBasedTradingBacktester
from logic_layer.ml_models_analyzer import MLModelAnalyzer
import pandas as pd
import numpy as np

from logic_layer.neural_network_models_trainer import NeuralNetworkModelTrainer
from logic_layer.rnn_models_analyzer import RNNModelsAnalyzer


class AlgosOrchestationLogic:

    def __init__(self,hist_data_conn_str,ml_reports_conn_str,p_classification_map_key,logger):

        self.logger=logger

        self.data_set_builder=DataSetBuilder(hist_data_conn_str,ml_reports_conn_str,p_classification_map_key,logger)

        self.date_range_classif_mgr = DateRangeClassificationManager(ml_reports_conn_str)

        self.timestamp_range_classif_mgr=TimestampClassificationManager(ml_reports_conn_str)


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


    def process_train_LSTM(self,symbol,variables_csv,d_from,d_to,model_output,classif_key):
        try:
            timestamp_range_clasifs=self.timestamp_range_classif_mgr.get_timestamp_range_classification_values(classif_key,d_from,d_to)

            symbol_min_series_df= self.data_set_builder.build_minute_series(symbol, d_from, d_to, output_col=["symbol", "date", "open", "high", "low", "close"])
            symbol_min_series_df = self.data_set_builder.build_minute_series_classification(timestamp_range_clasifs,
                                                                                            symbol_min_series_df,
                                                                                            classif_col_name=classif_key,
                                                                                            not_found_clasif="FLAT")

            variables_min_series_df=self.data_set_builder.build_minute_series(variables_csv, d_from, d_to, output_col=["symbol", "date", "open", "high", "low", "close"])

            training_series_df= self.data_set_builder.merge_minute_series(symbol_min_series_df,variables_min_series_df,"symbol","date",symbol)

            rnn_model_trainer= RNNModelsAnalyzer()

            rnn_model_trainer.build_LSTM(training_series_df,model_output,symbol_min_series_df,classif_key,30)

            #pd.set_option('display.max_columns', None)
            #print(training_series_df.head())
            #pd.reset_option('display.max_columns')

            return None

        except Exception as e:
            msg = "CRITICAL ERROR processing model @train_neural_network:{}".format(str(e))
            self.logger.do_log(msg, MessageType.ERROR)
            raise Exception(msg)


    def process_daily_candles_graph(self,symbol, date, interval):
        start_of_day = datetime(date.year, date.month, date.day)
        end_of_day = start_of_day + timedelta(hours=23, minutes=59, seconds=59)

        prices_df= self.data_set_builder.build_minute_series(symbol,start_of_day,end_of_day)

        GraphBuilder.build_candles_graph(prices_df)

        return  None


