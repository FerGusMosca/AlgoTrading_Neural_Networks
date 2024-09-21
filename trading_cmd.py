from common.util.date_handler import DateHandler
from common.util.logger import Logger
from common.util.ml_settings_loader import MLSettingsLoader
from framework.common.logger.message_type import MessageType
from logic_layer.data_management import AlgosOrchestationLogic
from IPython.display import display
import pandas as pd

_DATE_FORMAT = "%m/%d/%Y"

last_trading_dict = None


def show_commands():
    print("#0-DailyCandlesGraph  [Symbol] [date] [interval] [mmov_avg]")
    print("#1-TrainMLAlgos  [SeriesCSV] [from] [to] [classif_key]")
    print("#2-RunPredictionsLastModel [SeriesCSV] [from] [to] [classif_key]")
    print("#3-EvalBiasedTradingAlgo [Symbol] [SeriesCSV] [from] [to] [Bias] [classif_key]")
    print("#4-EvaluateARIMA [Symbol] [Period] [from] [to]")
    print("#5-PredictARIMA [Symbol] [p] [d] [q] [from] [to] [Period] [Step]")
    print("#6-EvalSingleIndicatorAlgo [Symbol] [indicator] [from] [to] [inverted] [classif_key]")
    print("#7-EvalMLBiasedAlgo [Symbol] [indicator] [SeriesCSV] [from] [to] [inverted] [classif_key]")
    print("#8-TrainNeuralNetworkAlgo [symbol] [variables_csv] [from] [to] [depth] [learning_rate] [iterations] [model_output] [classif_key]")
    print("#9-BacktestNeuralNetworkAlgo [symbol] [variables_csv] [from] [to] [model_to_use] [classif_key]")
    print("#10-TrainLSTM [symbol] [variables_csv] [from] [to] [model_output] [classif_key] [epochs] [timestamps] [# neurons] [learning_rate] [reg_rate] [dropout_rate] [clipping_rate] [acc_stop]")
    print("#11-TrainLSTMWithGrouping [symbol] [variables_csv] [from] [to] [model_output] [classif_key] [epochs] [timestamps] [# neurons] [learning_rate] [reg_rate] [dropout_rate] [clipping_rate] [acc_stop] [grouping_unit] [grouping_classif_criteria]")

    print("#12-TestDailyLSTM [symbol] [variables_csv] [from] [to] [timestemps] [model_to_use] [portf_size] [trade_comm] [trading_algo] [algo_params*]")
    print("#13-TestDailyLSTMWithGrouping [symbol] [variables_csv] [from] [to] [timestemps] [model_to_use] [portf_size] [trade_comm] [trading_algo] [grouping_unit] [algo_params*]")

    #TrainNeuralNetworkAlgo
    print("#n-Exit")


def count_params( param_list, exp_len):
    return len(param_list) == exp_len

def params_validation(cmd, param_list, exp_len):
    if (len(param_list) != exp_len):
        raise Exception("Command {} expects {} parameters".format(cmd, exp_len))


def __get_testing_params__(trading_algo,cmd_param_list,base_length=10):
    n_params = []
    if (trading_algo == AlgosOrchestationLogic._TRADING_ALGO_RAW_ALGO):
        params_validation("TestDailyLSTM", cmd_param_list, base_length)
    elif (trading_algo == AlgosOrchestationLogic._TRADING_ALGO_N_MIN_BUFFER_W_FLIP):
        params_validation("TestDailyLSTM", cmd_param_list, base_length+1)
        n_params.append(int(cmd_param_list[base_length]))
    elif (trading_algo == AlgosOrchestationLogic._TRADING_ALGO_ONLY_SIGNAL_N_MIN_PLUS_MOV_AVG):
        params_validation("TestDailyLSTM", cmd_param_list, base_length+2)
        n_params.append(int(cmd_param_list[base_length]))
        n_params.append(int(cmd_param_list[base_length+1]))

    return  n_params


def process_traing_LSTM_cmd(cmd_param_list):
    if count_params(cmd_param_list, 15):

        params_validation("TrainLSTM", cmd_param_list, 15)
        process_train_LSTM(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                           cmd_param_list[5], cmd_param_list[6], cmd_param_list[7],
                           cmd_param_list[8], cmd_param_list[9], cmd_param_list[10]
                           , cmd_param_list[11], cmd_param_list[12], cmd_param_list[13], cmd_param_list[14])
    elif count_params(cmd_param_list, 19):  # group_as_mov_avg + grouping_mov_avg_unit
        params_validation("TrainLSTM", cmd_param_list, 19)
        process_train_LSTM(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                           cmd_param_list[5], cmd_param_list[6], cmd_param_list[7],
                           cmd_param_list[8], cmd_param_list[9], cmd_param_list[10]
                           , cmd_param_list[11], cmd_param_list[12], cmd_param_list[13],
                           cmd_param_list[14],
                           grouping_unit=cmd_param_list[15], grouping_classif_criteria=cmd_param_list[16],
                           group_as_mov_avg=cmd_param_list[17] == "True", grouping_mov_avg_unit=cmd_param_list[18])

    elif count_params(cmd_param_list, 17):  # group_as_mov_avg + grouping_mov_avg_unit
        params_validation("TrainLSTM", cmd_param_list, 17)
        process_train_LSTM(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                           cmd_param_list[5], cmd_param_list[6], cmd_param_list[7],
                           cmd_param_list[8], cmd_param_list[9], cmd_param_list[10]
                           , cmd_param_list[11], cmd_param_list[12], cmd_param_list[13],
                           cmd_param_list[14],
                           group_as_mov_avg=cmd_param_list[15] == "True", grouping_mov_avg_unit=cmd_param_list[16])


def process_train_ml_algos(cmd_param_list, str_from, str_to, classification_key=None):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Initializing dataframe creation for series : {}".format(cmd_param_list[1]), MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         classification_key if classification_key is not None else config_settings[
                                             "classification_map_key"], logger)
        dataMgm.train_algos(cmd_param_list, DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                            DateHandler.convert_str_date(str_to, _DATE_FORMAT))

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)


def process_biased_trading_algo(symbol, cmd_series_csv, str_from, str_to, bias, classification_key):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        global last_trading_dict
        logger.print("Evaluating trading performance for symbol from last model from {} to {}".format(str_from, str_to),
                     MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         config_settings[
                                             "classification_map_key"] if classification_key is None else classification_key,
                                         logger)
        summary_dict = dataMgm.evaluate_trading_performance(symbol, cmd_series_csv,
                                                            DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                                            DateHandler.convert_str_date(str_to, _DATE_FORMAT), bias,
                                                            last_trading_dict)

        last_trading_dict = summary_dict

        print("Displaying all the different models predictions for the different alogs:")

        for key in summary_dict.keys():
            print("============{}============ for {}".format(key, symbol))
            summary = summary_dict[key]
            print("From={} To={}".format(str_from, str_to))
            print("Nom. Profit={}".format(summary.calculate_th_nom_profit()))
            print("Pos. Size={}".format(summary.portf_pos_size))
            print("Est. Max Drawdown={}".format(summary.max_drawdown))


    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)


def process_run_predictions_last_model(cmd_param_list, str_from, str_to, classification_key=None):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Running predictions fro last model from {} to {}".format(str_from, str_to), MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         config_settings[
                                             "classification_map_key"] if classification_key is None else classification_key,
                                         logger)
        pred_dict = dataMgm.run_predictions_last_model(cmd_param_list,
                                                       DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                                       DateHandler.convert_str_date(str_to, _DATE_FORMAT))
        print("Displaying all the different models predictions for the different alogs:")
        pd.set_option('display.max_rows', None)
        for key in pred_dict.keys():
            print("============{}============".format(key))
            display(pred_dict[key])
            print("")
            print("")
        pd.reset_option('display.max_rows')

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)


def process_eval_ARIMA(symbol, period, str_from, str_to):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Building ARIMA model for {} (period {}) from {} to {}".format(symbol, period, str_from, str_to),
                     MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         config_settings["classification_map_key"], logger)
        pred_dict = dataMgm.build_ARIMA(symbol, period,
                                        DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                        DateHandler.convert_str_date(str_to, _DATE_FORMAT))

        print("======= Showing Dickey Fuller Test after building ARIMA======= ")
        for key in pred_dict.keys():
            print("{}={}".format(key, pred_dict[key]))

        pass  #brkpnt to see the graph!

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)

        # [Symbol] [indicator] [from] [to] [inverted]


def process_eval_single_indicator_algo(symbol, indicator, str_from, str_to, inverted, classification_key):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Evaluating Single Indicator Algo for {} from {} to {}".format(symbol, str_from, str_to),
                     MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         config_settings["classification_map_key"] if classification_key is None else
                                         config_settings["classification_map_key"],
                                         logger)
        portf_summary = dataMgm.eval_singe_indicator_algo(symbol, indicator, inverted == "True",
                                                          DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                                          DateHandler.convert_str_date(str_to, _DATE_FORMAT))

        print("============= Displaying porftolio status for symbol {} =============".format(symbol))
        print("From={} To={}".format(str_from, str_to))
        print("Nom. Profit={}".format(portf_summary.calculate_th_nom_profit()))
        print("Pos. Size={}".format(portf_summary.portf_pos_size))
        pass

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)


def process_eval_ml_biased_algo(symbol, indicator, seriesCSV, str_from, str_to, inverted, classification_key):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Evaluating ML biasde algo for {} from {} to {}".format(symbol, str_from, str_to),
                     MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         config_settings[
                                             "classification_map_key"] if classification_key is None else classification_key,
                                         logger)
        portf_summary_dict = dataMgm.eval_ml_biased_algo(symbol, indicator, seriesCSV,
                                                         DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                                         DateHandler.convert_str_date(str_to, _DATE_FORMAT),
                                                         inverted == "True")

        for algo in portf_summary_dict.keys():
            portf_summary = portf_summary_dict[algo]
            print(
                "============= Displaying porftolio status for symbol {} w/Algo {} =============".format(symbol, algo))
            print("From={} To={}".format(str_from, str_to))
            print("Nom. Profit={}".format(portf_summary.calculate_th_nom_profit()))
            print("Pos. Size={}".format(portf_summary.portf_pos_size))
            print("============= =============")

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)


def process_predict_ARIMA(symbol, p, d, q, str_from, str_to, period, step):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print(
            "Predicting w/last built ARIMA model for {} (period {}) from {} to {}".format(symbol, period, str_from,
                                                                                          str_to), MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         config_settings["classification_map_key"], logger)
        preds_list = dataMgm.predict_ARIMA(symbol, int(p), int(d), int(q),
                                           DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                           DateHandler.convert_str_date(str_to, _DATE_FORMAT),
                                           period, int(step))
        print("==== Displaying Predictions for following periods ==== ")
        i = 1
        for pred in preds_list:
            print("{} --> {} %".format(period + str(i), "{:.2f}".format(pred * 100)))
            i += 1

        pass  #brkpnt to see the graph!

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)


def process_backtest_neural_network_algo(symbol, variables_csv, str_from, str_to, model_to_use,
                                         classification_key):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print("Initializing dataframe creation for series : {}".format(variables_csv), MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         config_settings[
                                             "classification_map_key"] if classification_key is None else classification_key,
                                         logger)

        dataMgm.backtest_neural_network_algo(symbol, variables_csv,
                                             DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                             DateHandler.convert_str_date(str_to, _DATE_FORMAT), model_to_use)

        #TODO ---> print backtesting output
        logger.print("Model successfully trained for symbol {} and variables {}".format(symbol, variables_csv),
                     MessageType.INFO)

    except Exception as e:
        logger.print("CRITICAL ERROR running proces_train_neural_network_algo:{}".format(str(e)), MessageType.ERROR)


def process_train_neural_network_algo(symbol, variables_csv, str_from, str_to, depth, learning_rate, epochs,
                                      model_output,
                                      classification_key):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print("Initializing dataframe creation for series : {}".format(variables_csv), MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         config_settings[
                                             "classification_map_key"] if classification_key is None else classification_key,
                                         logger)

        dataMgm.train_neural_network(symbol, variables_csv, DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                     DateHandler.convert_str_date(str_to, _DATE_FORMAT), depth, learning_rate,
                                     epochs, model_output)

        logger.print("Model successfully trained for symbol {} and variables {}".format(symbol, variables_csv),
                     MessageType.INFO)

    except Exception as e:
        logger.print("CRITICAL ERROR running proces_train_neural_network_algo:{}".format(str(e)), MessageType.ERROR)


def process_train_LSTM(symbol, variables_csv, str_from, str_to, model_output, classification_key,
                       epochs, timestamps, n_neurons, learning_rate, reg_rate, dropout_rate,clipping_rate,
                       accuracy_stop,grouping_unit=None,grouping_classif_criteria=None,
                       group_as_mov_avg=False,grouping_mov_avg_unit=100):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print("Initializing dataframe creation for series : {}".format(variables_csv), MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         config_settings[
                                             "classification_map_key"] if classification_key is None else classification_key,
                                         logger)

        dataMgm.process_train_LSTM(symbol, variables_csv,
                                   DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                   DateHandler.convert_str_date(str_to, _DATE_FORMAT),
                                   model_output.replace('"', ""),
                                   classification_key, int(epochs), int(timestamps),
                                   int(n_neurons), float(learning_rate),
                                   float(reg_rate), float(dropout_rate),float(clipping_rate),
                                   float(accuracy_stop),
                                   int(grouping_unit) if grouping_unit is not None else None,
                                   grouping_classif_criteria,
                                   bool(group_as_mov_avg),int(grouping_mov_avg_unit))

        # TODO ---> print backtesting output
        logger.print("Model successfully trained for symbol {} and variables {}".format(symbol, variables_csv),
                     MessageType.INFO)

    except Exception as e:
        logger.print("CRITICAL ERROR running process_train_LSTM:{}".format(str(e)), MessageType.ERROR)


def process_test_daily_LSTM(symbol, variables_csv, str_from,str_to, timesteps, model_to_use, portf_size, trade_comm,
                            trading_algo,grouping_unit=None,n_params=[]):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print("Initializing model testing for symbol {} and model {} on {}".format(symbol, model_to_use, str_from),
                     MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         None,
                                         logger)

        dataMgm.process_test_daily_LSTM(symbol=symbol, variables_csv=variables_csv,
                                        model_to_use=model_to_use.replace('"', ""),
                                        d_from=DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                        d_to=DateHandler.convert_str_date(str_to, _DATE_FORMAT),
                                        timesteps=int(timesteps),
                                        portf_size=float(portf_size),
                                        trade_comm=float(trade_comm),
                                        trading_algo=trading_algo,
                                        grouping_unit=int(grouping_unit) if grouping_unit is not None else None,
                                        n_algo_params=n_params

                                        )

        logger.print(
            "Displaying predictions for LSTM model: symbol {} and model {} on {}".format(symbol, model_to_use, str_from),
            MessageType.INFO)
    except Exception as e:
        logger.print("CRITICAL ERROR running process_test_daily_LSTM:{}".format(str(e)), MessageType.ERROR)


def process_daily_candles_graph(symbol, date, interval,mov_avg_unit):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print("Initializing daily candle graph creation for symbol {} on date {}".format(symbol, date),
                     MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         None, logger)

        dataMgm.process_daily_candles_graph(symbol, DateHandler.convert_str_date(date, _DATE_FORMAT),
                                            interval.replace('_', ""),int(mov_avg_unit))

        logger.print("Daily Graph successfully shown for symbol {} on date {}".format(symbol, date),
                     MessageType.INFO)

    except Exception as e:
        logger.print("CRITICAL ERROR running process_daily_candles_graph:{}".format(str(e)), MessageType.ERROR)


def process_commands(cmd):
    cmd_param_list = cmd.split(" ")

    if cmd_param_list[0] == "TrainMLAlgos":
        params_validation("TrainMLAlgos", cmd_param_list, 5)
        process_train_ml_algos(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4])

    elif cmd_param_list[0] == "RunPredictionsLastModel":
        params_validation("RunPredictionsLastModel", cmd_param_list, 5)
        process_run_predictions_last_model(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4])
    elif cmd_param_list[0] == "EvalBiasedTradingAlgo":
        params_validation("EvalBiasedTradingAlgo", cmd_param_list, 7)
        process_biased_trading_algo(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                    cmd_param_list[5], cmd_param_list[6])
    elif cmd_param_list[0] == "EvaluateARIMA":
        params_validation("EvaluateARIMA", cmd_param_list, 5)
        process_eval_ARIMA(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4])
    elif cmd_param_list[0] == "PredictARIMA":
        params_validation("PredictARIMA", cmd_param_list, 9)
        process_predict_ARIMA(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                              cmd_param_list[5], cmd_param_list[6], cmd_param_list[7], cmd_param_list[8])
    elif cmd_param_list[0] == "EvalSingleIndicatorAlgo":
        params_validation("EvalSingleIndicatorAlgo", cmd_param_list, 7)
        process_eval_single_indicator_algo(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                           cmd_param_list[5], cmd_param_list[6])
    elif cmd_param_list[0] == "EvalMLBiasedAlgo":
        params_validation("EvalMLBiasedAlgo", cmd_param_list, 8)
        process_eval_ml_biased_algo(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                    cmd_param_list[5], cmd_param_list[6], cmd_param_list[7])
    elif cmd_param_list[0] == "TrainNeuralNetworkAlgo":
        params_validation("TrainNeuralNetworkAlgo", cmd_param_list, 10)
        process_train_neural_network_algo(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                          int(cmd_param_list[5]), float(cmd_param_list[6]), int(cmd_param_list[7]),
                                          cmd_param_list[8], cmd_param_list[9])
    elif cmd_param_list[0] == "BacktestNeuralNetworkAlgo":
        params_validation("BacktestNeuralNetworkAlgo", cmd_param_list, 7)
        process_backtest_neural_network_algo(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                             cmd_param_list[5], cmd_param_list[6])

    elif cmd_param_list[0] == "TrainLSTM":
        process_traing_LSTM_cmd(cmd_param_list)

    elif cmd_param_list[0] == "TrainLSTMWithGrouping":
        params_validation("TrainLSTMWithGrouping", cmd_param_list, 17)
        process_train_LSTM(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                           cmd_param_list[5], cmd_param_list[6], cmd_param_list[7],
                           cmd_param_list[8], cmd_param_list[9], cmd_param_list[10]
                           , cmd_param_list[11], cmd_param_list[12], cmd_param_list[13], cmd_param_list[14]
                           , cmd_param_list[15] , cmd_param_list[16])
    #
    elif cmd_param_list[0] == "DailyCandlesGraph":

        params_validation("DailyCandlesGraph", cmd_param_list, 5)
        process_daily_candles_graph(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3],
                                    cmd_param_list[4])
    elif cmd_param_list[0] == "TestDailyLSTM":

        trading_algo = cmd_param_list[9]
        n_params=__get_testing_params__(trading_algo,cmd_param_list,10)
        process_test_daily_LSTM(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                cmd_param_list[5], cmd_param_list[6], cmd_param_list[7], cmd_param_list[8]
                                , cmd_param_list[9],grouping_unit=None,n_params= n_params)
    elif cmd_param_list[0] == "TestDailyLSTMWithGrouping":
        trading_algo = cmd_param_list[9]
        n_params = __get_testing_params__(trading_algo, cmd_param_list, 11)
        process_test_daily_LSTM(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                cmd_param_list[5], cmd_param_list[6], cmd_param_list[7], cmd_param_list[8],
                                cmd_param_list[9],grouping_unit=cmd_param_list[10],n_params=n_params)

    #


    #TestDailyLSTM

    #TrainNeuralNetworkAlgo

    else:
        print("Not recognized command {}".format(cmd_param_list[0]))


if __name__ == '__main__':

    while True:

        show_commands()
        cmd = input("Enter a command:")
        try:
            process_commands(cmd)
            if (cmd == "Exit"):
                break
        except Exception as e:
            print("Could not process command:{}".format(str(e)))

    print("Exit")
