import traceback

from business_entities.portf_position import PortfolioPosition
from common.util.date_handler import DateHandler
from common.util.logger import Logger
from common.util.ml_settings_loader import MLSettingsLoader
from controllers.main_dashboard_controller import MainDashboardController
from framework.common.logger.message_type import MessageType
from logic_layer.algos_orchestation_logic import AlgosOrchestationLogic
from IPython.display import display
import pandas as pd

from logic_layer.data_set_builder import DataSetBuilder
from controllers.routing_dashboard_controller import RoutingDashboardController

_DATE_FORMAT = "%m/%d/%Y"
_TIMESTAMP_FORMAT='%m/%d/%Yt%H:%M:%S'

last_trading_dict = None


def show_commands():
    print("#0.1-DailyCandlesGraph [Symbol] [date] [interval] [mmov_avg]")
    print("#0.2-IndicatorCandlesGraph [Symbol] [from] [to] [interval] [mmov_avg]")
    print("#1-TrainMLAlgos  [SeriesCSV] [from] [to] [classif_key]")
    print("#2-RunPredictionsLastModel [SeriesCSV] [from] [to] [classif_key]")
    print("#3-EvalBiasedTradingAlgo [Symbol] [SeriesCSV] [from] [to] [Bias] [classif_key]")
    print("#4-EvaluateARIMA [Symbol] [Period] [from] [to]")
    print("#5-PredictARIMA [Symbol] [p] [d] [q] [from] [to] [Period] [Step]")
    print("#6-EvalSingleIndicatorAlgo [Symbol] [indicator] [from] [to] [inverted] [classif_key]")
    print("#7-EvalMLBiasedAlgo [Symbol] [indicator] [SeriesCSV] [from] [to] [inverted] [classif_key]")
    print("#8-TrainNeuralNetworkAlgo [symbol] [variables_csv] [from] [to] [depth] [learning_rate] [iterations] [model_output] [classif_key]")
    print("#9-BacktestNeuralNetworkAlgo [symbol] [variables_csv] [from] [to] [model_to_use] [classif_key]")
    print("#10-TrainLSTM [symbol] [variables_csv] [from] [to] [model_output] [classif_key] [epochs] [timestamps] [# neurons] [learning_rate] [reg_rate] [dropout_rate] [clipping_rate] [acc_stop] [batch_size*] [inner_activation*] [make_stationary*]")
    print("#11-TrainLSTMWithGrouping [symbol] [variables_csv] [from] [to] [model_output] [classif_key] [epochs] [timestamps] [# neurons] [learning_rate] [reg_rate] [dropout_rate] [clipping_rate] [acc_stop] [grouping_unit] [grouping_classif_criteria] [batch_size*] [inner_activation*]")

    print("#12-TestDailyLSTM [symbol] [variables_csv] [from] [to] [timestemps] [model_to_use] [portf_size] [trade_comm] [trading_algo] [algo_params*]")
    print("#13-TestDailyLSTMWithGrouping [symbol] [variables_csv] [from] [to] [timestemps] [model_to_use] [portf_size] [trade_comm] [trading_algo] [grouping_unit] [algo_params*]")
    print("#14-BacktestSlopeModel [symbol] [model_candle] [from] [to] [portf_size] [trade_comm] [trading_algo] [algo_params*]")
    print("#15-BacktestSlopeModelOnCustomETF [ETF_path] [model_candle] [from] [to] [portf_size] [trade_comm] [trading_algo] [algo_params*]")
    print("#16-CreateSintheticIndicator [comp_path] [model_candle] [from] [to] [slope_units]")
    print("======================== UI ========================")
    print("#30-BiasMainLandingPage")
    print("#31-DisplayOrderRoutingScreen")
    #TrainNeuralNetworkAlgo
    print("#n-Exit")


def count_params( param_list, exp_len):
    return len(param_list) == exp_len

def params_validation(cmd, param_list, exp_len):
    if (len(param_list) != exp_len):
        raise Exception("Command {} expects {} parameters".format(cmd, exp_len))


def __get_value_after_equals__(command, key,optional=False):
    # Separamos el comando en partes por espacios
    parts = command.split(" ")

    # Recorremos las partes buscando las que coincidan con la clave
    for part in parts:
        # Verificamos si la parte contiene el key seguido de "="
        if part.startswith(key + "="):
            # Extraemos el valor después del "="
            value = part.split("=")[1].strip()  # Tomamos lo que está después del '='

            # Si el valor está entre comillas, eliminamos las comillas
            if value.startswith('"') or value.startswith("'"):
                value = value[1:-1]  # Eliminamos las comillas

            return value

    if not optional:
    # Si no se encontró la clave, lanzamos una excepción
        raise KeyError(f"Key {key} not found.")
    else:
        return None

def __get_bool_param__(command, key, optional=False,def_value=None):
    str_val = __get_param__(command,key,optional,def_value)

    if str_val=="True" or str_val=="False":
        return str_val=="True"
    else:
        return def_value

def __get_param__(command, key, optional=False,def_value=None):
    value = __get_value_after_equals__(command, key,optional)

    if value==None and optional:
        return  def_value

    # Intentar convertir a fecha en formato MM/dd/yyyy
    try:
        return  DateHandler.convert_str_date(value, _DATE_FORMAT)
    except Exception:
        pass  # No es una fecha válida, continuamos con las siguientes verificaciones

    # Intentar convertir a fecha en formato MM/dd/yyyy HH:mm:ss
    try:
        return  DateHandler.convert_str_date(value, _TIMESTAMP_FORMAT)
    except Exception:
        pass  # No es una fecha válida, continuamos con las siguientes verificaciones

    # Intentar convertir a entero
    try:
        return int(value)
    except ValueError:
        pass  # No es un entero válido, continuamos con las siguientes verificaciones

    # Intentar convertir a float con dos decimales
    try:
        return float(value)
    except ValueError:
        pass  # No es un float válido, continuamos con el último caso

    # Si no es ninguno de los anteriores, lo devolvemos como string
    return value


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

def process_backtest_slope_model(cmd):
    symbol = __get_param__(cmd, "symbol")
    model_candle = __get_param__(cmd, "model_candle")
    d_from = __get_param__(cmd, "from")
    d_to = __get_param__(cmd, "to")
    portf_size = __get_param__(cmd, "portf_size",optional=True,def_value=100000)

    trading_algo = __get_param__(cmd, "trading_algo")
    candle_slope = __get_param__(cmd, "candle_slope", True, None)
    slope_units = __get_param__(cmd, "slope_units", True, None)

    trade_comm = __get_param__(cmd, "trade_comm", optional=True, def_value=0)
    trade_comm_pct = __get_param__(cmd, "trade_comm_pct", optional=True, def_value=0)

    cmd_param_dict = {}
    if candle_slope is not None:
        cmd_param_dict["candle_slope"]=candle_slope

    if slope_units is not None:
        cmd_param_dict["slope_units"]=slope_units

    if trade_comm is not None:
        cmd_param_dict["trade_comm"]=trade_comm

    if trade_comm_pct is not None:
        cmd_param_dict["trade_comm_pct"]=trade_comm_pct

    process_backtest_slope_model_logic(symbol,model_candle=model_candle,d_from=d_from,d_to=d_to,
                                        portf_size=portf_size,
                                        trading_algo=trading_algo,algo_params=cmd_param_dict)

    print(f"Test Backtest Slope Model finished...")


def process_bias_main_landing_page(cmd):
    loader = MLSettingsLoader()
    logger = Logger()

    # loader user to load settings
    config_settings = loader.load_settings("./configs/commands_mgr.ini")

    main_dash_contr = MainDashboardController(logger, config_settings)
    main_dash_contr.display()
    print(f"Main Dashboard successfully shown...")


def process_display_order_routing_screen(cmd):
    loader = MLSettingsLoader()
    logger = Logger()

    #loader user to load settings
    config_settings = loader.load_settings("./configs/commands_mgr.ini")

    ib_prod_ws=config_settings["IB_PROD_WS"]
    primary_prod_ws = config_settings["PRIMARY_PROD_WS"]
    ib_dev_ws = config_settings["IB_DEV_WS"]


    wmc= RoutingDashboardController(logger, ib_prod_ws, primary_prod_ws, ib_dev_ws)

    wmc.display_order_routing_screen()
    print(f"Web Manager Logic successfully shown...")


def process_create_sinthetic_indicator(cmd):
    comp_apth = __get_param__(cmd, "comp_path")
    model_candle = __get_param__(cmd, "model_candle")
    d_from = __get_param__(cmd, "from")
    d_to = __get_param__(cmd, "to")

    #DIRECT/INV SLOPE
    slope_units = __get_param__(cmd, "slope_units", True, None)

    #ARIMA
    p = __get_param__(cmd, "p", True, None)
    d = __get_param__(cmd, "d", True, None)
    q = __get_param__(cmd, "q", True, None)
    step = __get_param__(cmd, "step", True, None)
    inv_steps = __get_param__(cmd, "inv_steps", True, None)
    min_units_to_pred = __get_param__(cmd, "min_units_to_pred", True, None)

    cmd_param_dict = {}

    if slope_units is not None:
        cmd_param_dict["slope_units"] = slope_units

    if p is not None:
        cmd_param_dict["p"] = p

    if d is not None:
        cmd_param_dict["d"] = d

    if q is not None:
        cmd_param_dict["q"] = q

    if step is not None:
        cmd_param_dict["step"] = step

    if inv_steps is not None:
        cmd_param_dict["inv_steps"] = inv_steps

    if min_units_to_pred is not None:
        cmd_param_dict["min_units_to_pred"] = min_units_to_pred

    process_create_sinthetic_indicator_logic(comp_apth, model_candle=model_candle, d_from=d_from, d_to=d_to,
                                            algo_params=cmd_param_dict)

    print(f"Create sinthetic indicator finished...")
def process_backtest_slope_model_on_custom_etf(cmd):
    etf_path = __get_param__(cmd, "ETF_path")
    model_candle = __get_param__(cmd, "model_candle")
    d_from = __get_param__(cmd, "from")
    d_to = __get_param__(cmd, "to")
    portf_size = __get_param__(cmd, "portf_size",optional=True,def_value=100000)

    trading_algo = __get_param__(cmd, "trading_algo")
    candle_slope = __get_param__(cmd, "candle_slope", True, None)
    slope_units = __get_param__(cmd, "slope_units", True, None)

    trade_comm = __get_param__(cmd, "trade_comm", optional=True, def_value=0)
    trade_comm_pct = __get_param__(cmd, "trade_comm_pct", optional=True, def_value=0)

    days_to_add_to_date = __get_param__(cmd, "days_to_add_to_date", optional=True, def_value=None)

    cmd_param_dict = {}
    if candle_slope is not None:
        cmd_param_dict["candle_slope"]=candle_slope

    if slope_units is not None:
        cmd_param_dict["slope_units"]=slope_units

    if trade_comm is not None:
        cmd_param_dict["trade_comm"]=trade_comm

    if trade_comm_pct is not None:
        cmd_param_dict["trade_comm_pct"]=trade_comm_pct

    if days_to_add_to_date is not None:
        cmd_param_dict["days_to_add_to_date"]=days_to_add_to_date

    process_backtest_slope_model_on_custom_etf_logic(etf_path,model_candle=model_candle,d_from=d_from,d_to=d_to,
                                                    portf_size=portf_size,
                                                    trading_algo=trading_algo,algo_params=cmd_param_dict)

    print(f"Test Backtest Slope Model finished...")


def process_test_LSTM_cmd(cmd):
    symbol = __get_param__(cmd, "symbol")
    variables_csv = __get_param__(cmd, "variables_csv")
    d_from = __get_param__(cmd, "from")
    d_to = __get_param__(cmd, "to")
    timesteps=__get_param__(cmd,"timesteps")
    model_to_use = __get_param__(cmd, "model_to_use")
    portf_size=__get_param__(cmd,"portf_size")
    comm=__get_param__(cmd,"comm")
    interval=__get_param__(cmd,"interval",True,None)
    trading_algo=__get_param__(cmd,"trading_algo")
    grouping_unit=__get_param__(cmd,"grouping_unit",True,None)
    n_buffer=__get_param__(cmd,"n_buffer",True,None)
    mov_avg=__get_param__(cmd,"mov_avg",True,None)
    use_sliding_window = __get_param__(cmd, "use_sliding_window", True,def_value="None")#NONE,CUT_INPUT_DF,GET_FAKE_DATA
    make_stationary = __get_bool_param__(cmd, "make_stationary", True, False)

    cmd_param_list=[]
    if n_buffer is not None:
        cmd_param_list.append(n_buffer)

    if mov_avg is not None:
        cmd_param_list.append(mov_avg)


    process_test_daily_LSTM(symbol=symbol, variables_csv=variables_csv, d_from=d_from, d_to=d_to,
                            timesteps=timesteps,model_to_use=model_to_use, portf_size=portf_size,
                            trade_comm=comm, trading_algo=trading_algo,interval=interval,
                            grouping_unit=grouping_unit,n_params=cmd_param_list,
                            use_sliding_window=use_sliding_window,make_stationary=make_stationary)

    print(f"Test LSTM successfully finished...")



def process_traing_LSTM_cmd(cmd,cmd_param_list):
    symbol = __get_param__(cmd, "symbol")
    variables_csv = __get_param__(cmd, "variables_csv")
    d_from = __get_param__(cmd, "from")
    d_to = __get_param__(cmd, "to")
    model_output = __get_param__(cmd, "model_output")
    classif_key = __get_param__(cmd, "classif_key")
    epochs = __get_param__(cmd, "epochs")
    n_neurons = __get_param__(cmd, "n_neurons")
    timesteps = __get_param__(cmd, "timesteps")
    learning_rate = __get_param__(cmd, "learning_rate")
    dropout_rate = __get_param__(cmd, "dropout_rate")
    clipping_rate = __get_param__(cmd, "clipping_rate")
    reg_rate = __get_param__(cmd, "reg_rate")
    acc_stop = __get_param__(cmd, "acc_stop")
    interval = __get_param__(cmd, "interval",True,None)
    grouping_unit=__get_param__(cmd,"grouping_unit",True)
    grouping_classif_criteria=__get_param__(cmd,"grouping_classif_criteria",True)
    group_as_mov_avg=__get_bool_param__(cmd,"grouping_classif_criteria",True,def_value=False)
    grouping_mov_avg_unit=__get_param__(cmd,"grouping_mov_avg_unit",True,def_value=100)
    batch_size = __get_param__(cmd, "batch_size", True, def_value=1)
    inner_activation = __get_param__(cmd, "inner_activation", True, def_value=None)
    make_stationary = __get_bool_param__(cmd, "make_stationary", True,False)


    process_train_LSTM(symbol=symbol, variables_csv=variables_csv, d_from=d_from, d_to=d_to, model_output=model_output,
                       classification_key=classif_key, epochs=epochs, timestamps=timesteps, n_neurons=n_neurons,
                       learning_rate=learning_rate, reg_rate=reg_rate, dropout_rate=dropout_rate,
                       clipping_rate=clipping_rate, accuracy_stop=acc_stop, interval=interval,
                       grouping_unit=grouping_unit,grouping_classif_criteria=grouping_classif_criteria,
                       group_as_mov_avg=group_as_mov_avg,grouping_mov_avg_unit=grouping_mov_avg_unit,
                       batch_size=batch_size,inner_activation=inner_activation,
                       make_stationary=make_stationary)  # will use default

    print(f"Train LSTM successfully finished...")




def process_train_ml_algos(cmd_param_list, d_from, d_to, classification_key=None):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Initializing dataframe creation for series : {}".format(cmd_param_list[1]), MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        classif_key=classification_key if classification_key is not None else config_settings["classification_map_key"]

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         classif_key, logger)
        dataMgm.train_algos(cmd_param_list,d_from,d_to,classif_key)

    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)


def run_train_ml_algo(cmd):
    series_csv = __get_param__(cmd, "series_csv")
    d_from = __get_param__(cmd, "from")
    d_to = __get_param__(cmd, "to")
    classif_key = __get_param__(cmd, "classif_key")
    process_train_ml_algos(series_csv, d_from, d_to, classif_key)

def run_biased_trading_algo(cmd):
    symbol = __get_param__(cmd, "symbol")
    series_csv = __get_param__(cmd, "series_csv")
    d_from = __get_param__(cmd, "from")
    d_to = __get_param__(cmd, "to")
    bias = __get_param__(cmd, "bias")

    trade_comm = __get_param__(cmd, "trade_comm", optional=True, def_value=5)
    init_portf_size=__get_param__(cmd, "init_portf_size",optional=True,def_value=PortfolioPosition._DEF_PORTF_AMT)

    n_algo_param_dict = {}
    n_algo_param_dict["trade_comm"]=trade_comm
    n_algo_param_dict["init_portf_size"]=init_portf_size
    process_biased_trading_algo(symbol,series_csv, d_from, d_to,bias,n_algo_param_dict)

def process_biased_trading_algo(symbol, cmd_series_csv, d_from, d_to, bias,n_algo_param_dict):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        global last_trading_dict
        logger.print("Evaluating trading performance for symbol from last model from {} to {}".format(d_from, d_to),
                     MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         None ,logger)
        summary_dict = dataMgm.evaluate_trading_performance(symbol, cmd_series_csv,
                                                            d_from,d_to, bias,last_trading_dict,
                                                            n_algo_param_dict=n_algo_param_dict)

        last_trading_dict = summary_dict

        print("Displaying all the different models predictions for the different alogs:")

        for key in summary_dict.keys():
            print("============{}============ for {}".format(key, symbol))
            summary = summary_dict[key]
            print("From={} To={}".format(d_from, d_to))
            print(f"Init Portfolio={round(summary.portf_init_MTM, 2)} $")
            print(f"Final Portfolio={round(summary.portf_final_MTM,2)} $")
            print(f"Pct Profit. Size={summary.total_net_profit}")
            print(f"Est. Max Drawdown={summary.max_drawdown_on_MTM}")
            print("     =========== Portf Positions=========== ")
            for portf_pos in summary.portf_pos_summary:

                print(f"    --Side={portf_pos.side} Open Date={portf_pos.date_open} Close Date={portf_pos.date_close} Open Price={portf_pos.price_open} Close Price={portf_pos.price_close} --> Pct. Profit={portf_pos.calculate_pct_profit()}%")


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


def process_eval_ARIMA_cmd(cmd):
    symbol = __get_param__(cmd, "symbol")
    d_from = __get_param__(cmd, "from")
    d_to = __get_param__(cmd, "to")
    period = __get_param__(cmd, "period",optional=True,def_value=None)

    process_eval_ARIMA(symbol,d_from,d_to,period)

def process_eval_ARIMA(symbol, d_from, d_to, period=None):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print("Building ARIMA model for {} (period {}) from {} to {}".format(symbol, period, d_from, d_to),
                     MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         config_settings["classification_map_key"], logger)
        pred_dict = dataMgm.build_ARIMA(symbol,d_from,d_to, period)

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


def process_predict_ARIMA_cmd(cmd):
    symbol = __get_param__(cmd, "symbol")
    d_from = __get_param__(cmd, "from")
    d_to = __get_param__(cmd, "to")
    p = __get_param__(cmd, "p")
    d = __get_param__(cmd, "d")
    q = __get_param__(cmd, "q")
    step = __get_param__(cmd, "step")
    period = __get_param__(cmd, "period", optional=True, def_value=None)


    process_predict_ARIMA(symbol,p,d,q,d_from,d_to , step,period)

def process_predict_ARIMA(symbol, p, d, q, d_from, d_to, step,period=None):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        logger.print(
            "Predicting w/last built ARIMA model for {} (period {}) from {} to {}".format(symbol, period, d_from,
                                                                                          d_to), MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         config_settings["classification_map_key"], logger)
        preds_list = dataMgm.predict_ARIMA(symbol, int(p), int(d), int(q),
                                           d_from,d_to, int(step),period)
        # Print ARIMA forecasted log returns in a clean table format
        print("\n📊 ARIMA Forecasted Log Returns\n")
        print("Period | Forecast (%)")
        print("-" * 30)

        for i, pred in enumerate(preds_list, start=1):
            print(f"{i:^8} | {pred * 100:.2f} %")

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


def process_train_LSTM(symbol, variables_csv, d_from, d_to, model_output, classification_key,
                       epochs, timestamps, n_neurons, learning_rate, reg_rate, dropout_rate,clipping_rate,
                       accuracy_stop,grouping_unit=None,grouping_classif_criteria=None,
                       group_as_mov_avg=False,grouping_mov_avg_unit=100,interval=None,
                       batch_size=1,inner_activation=None,make_stationary=False):
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
                                   #DateHandler.convert_str_date(str_from, _DATE_FORMAT),
                                   #DateHandler.convert_str_date(str_to, _DATE_FORMAT),
                                   d_from, d_to,
                                   model_output.replace('"', ""),
                                   classification_key, int(epochs), int(timestamps),
                                   int(n_neurons), float(learning_rate),
                                   float(reg_rate), float(dropout_rate),
                                   interval=interval.replace('_', " ") if interval is not None else None,
                                   clipping_rate= float(clipping_rate),
                                   accuracy_stop= float(accuracy_stop),

                                   grouping_unit= int(grouping_unit) if grouping_unit is not None else None,
                                   grouping_classif_criteria= grouping_classif_criteria,
                                   group_as_mov_avg= bool(group_as_mov_avg),
                                   grouping_mov_avg_unit= int(grouping_mov_avg_unit),
                                   batch_size=batch_size,inner_activation=inner_activation,
                                   make_stationary=make_stationary
                                   )

        # TODO ---> print backtesting output
        logger.print("Model successfully trained for symbol {} and variables {}".format(symbol, variables_csv),
                     MessageType.INFO)

    except Exception as e:
        logger.print("CRITICAL ERROR running process_train_LSTM:{}".format(str(e)), MessageType.ERROR)


def process_test_daily_LSTM(symbol, variables_csv, d_from,d_to, timesteps, model_to_use, portf_size, trade_comm,
                            trading_algo,grouping_unit=None,n_params=[],interval=None,use_sliding_window=None,
                            make_stationary=True):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print("Initializing model testing for symbol {} and model {} on {}".format(symbol, model_to_use, d_from),
                     MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         None,
                                         logger)
        interval=interval.replace('_', " ")
        if (interval== DataSetBuilder._1_MIN_INTERVAL or interval is None):
            dataMgm.process_test_daily_LSTM(symbol=symbol, variables_csv=variables_csv,
                                            model_to_use=model_to_use.replace('"', ""),
                                            d_from=d_from,
                                            d_to=d_to,
                                            timesteps=int(timesteps),
                                            portf_size=float(portf_size),
                                            trade_comm=float(trade_comm),
                                            trading_algo=trading_algo,
                                            grouping_unit=int(grouping_unit) if grouping_unit is not None else None,
                                            n_algo_params=n_params,
                                            interval=interval if interval is not None else None,
                                            use_sliding_window=use_sliding_window,
                                            make_stationary=make_stationary
                                            )
        elif interval==DataSetBuilder._1_DAY_INTERVAL:
            dataMgm.process_test_scalping_LSTM(symbol=symbol, variables_csv=variables_csv,
                                               model_to_use=model_to_use.replace('"', ""),
                                               d_from=d_from,
                                               d_to=d_to,
                                               timesteps=int(timesteps),
                                               portf_size=float(portf_size),
                                               trade_comm=float(trade_comm),
                                               trading_algo=trading_algo,
                                               grouping_unit=int(grouping_unit) if grouping_unit is not None else None,
                                               n_algo_params=n_params,
                                               interval=interval if interval is not None else None,
                                               make_stationary=make_stationary
                                               )
        else:
            raise Exception(f"Unknown interval! : {interval}")


        logger.print(
            "Displaying predictions for LSTM model: symbol {} and model {} on {}".format(symbol, model_to_use, d_from),
            MessageType.INFO)
    except Exception as e:
        logger.print("CRITICAL ERROR running process_test_daily_LSTM:{}".format(str(e)), MessageType.ERROR)



def process_indicator_candles_graph(symbol, d_from,d_to, interval,mov_avg_unit):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(f"Initializing indicator candle graph creation for symbol {symbol} from {d_from} to {d_to}",
                     MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         None, logger)

        dataMgm.process_indicator_candles_graph(symbol,
                                            DateHandler.convert_str_date(d_from, _DATE_FORMAT),
                                            DateHandler.convert_str_date(d_to, _DATE_FORMAT),
                                            interval.replace('_', " "),int(mov_avg_unit))

        logger.print(f"Indicator Graph successfully shown for symbol {symbol} from {d_from} to {d_to}",MessageType.INFO)

    except Exception as e:
        logger.print("CRITICAL ERROR running process_indicator_candles_graph:{}".format(str(e)), MessageType.ERROR)


def process_backtest_slope_model_logic(symbol,model_candle,d_from,d_to,portf_size,trading_algo,algo_params):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(f"Initializing model slope backtest for symbol {symbol} from {d_from} to {d_to}",MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        trd_algos = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         None, logger)

        trd_algos.process_backtest_slope_model(symbol,model_candle,d_from,d_to,portf_size,trading_algo,algo_params)

        logger.print(f"Slope Model backtest successfully shown for symbol {symbol} from {d_from} to {d_to}",
                     MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print("CRITICAL ERROR running process_backtest_slope_model:{}".format(str(e)), MessageType.ERROR)


def process_create_sinthetic_indicator_logic(comp_path,model_candle,d_from,d_to,algo_params):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(f"Initializing indicators file {comp_path} from {d_from} to {d_to}", MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        trd_algos = AlgosOrchestationLogic(config_settings["hist_data_conn_str"],
                                           config_settings["ml_reports_conn_str"],
                                           None, logger)

        trd_algos.process_create_sinthetic_indicator_logic(comp_path, model_candle, d_from, d_to, algo_params)

        logger.print(f"Sinthetic Indicator {model_candle} successfully created from ETF file {comp_path} from {d_from} to {d_to}",
                     MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print("CRITICAL ERROR running process_create_sinthetic_indicator_logic:{}".format(str(e)),MessageType.ERROR)


def process_backtest_slope_model_on_custom_etf_logic(etf_path,model_candle,d_from,d_to,portf_size,trading_algo,algo_params):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(f"Initializing model slope backtest for ETF file {etf_path} from {d_from} to {d_to}",MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        trd_algos = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         None, logger)

        trd_algos.process_backtest_slope_model_on_custom_etf(etf_path,model_candle,d_from,d_to,portf_size,trading_algo,algo_params)

        logger.print(f"Slope Model backtest successfully shown for ETF file {etf_path} from {d_from} to {d_to}",
                     MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print("CRITICAL ERROR running process_backtest_slope_model_on_custom_etf_logic:{}".format(str(e)), MessageType.ERROR)



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
        run_train_ml_algo(cmd)
    elif cmd_param_list[0] == "RunPredictionsLastModel":
        process_run_predictions_last_model(cmd_param_list,cmd_param_list[1],cmd_param_list[2],cmd_param_list[3])
    elif cmd_param_list[0] == "EvalBiasedTradingAlgo":
        run_biased_trading_algo(cmd)
    elif cmd_param_list[0] == "EvaluateARIMA":
        #params_validation("EvaluateARIMA", cmd_param_list, 5)
        process_eval_ARIMA_cmd(cmd)
    elif cmd_param_list[0] == "PredictARIMA":
        process_predict_ARIMA_cmd(cmd)
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
        process_traing_LSTM_cmd(cmd,cmd_param_list)

    elif cmd_param_list[0] == "TrainLSTMWithGrouping":
        process_traing_LSTM_cmd(cmd,cmd_param_list)
    elif cmd_param_list[0] == "DailyCandlesGraph":

        params_validation("DailyCandlesGraph", cmd_param_list, 5)
        process_daily_candles_graph(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3],
                                    cmd_param_list[4])
    elif cmd_param_list[0] == "IndicatorCandlesGraph":

        params_validation("IndicatorCandlesGraph", cmd_param_list, 6)
        process_indicator_candles_graph(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3],
                                    cmd_param_list[4],cmd_param_list[5])
    elif cmd_param_list[0] == "TestDailyLSTM":
        process_test_LSTM_cmd(cmd)
    elif cmd_param_list[0] == "BacktestSlopeModel":
        process_backtest_slope_model(cmd)

    elif cmd_param_list[0] == "BacktestSlopeModelOnCustomETF":
        process_backtest_slope_model_on_custom_etf(cmd)
    elif cmd_param_list[0] == "TestDailyLSTMWithGrouping":
        process_test_LSTM_cmd(cmd)

    elif cmd_param_list[0] == "CreateSintheticIndicator":
        process_create_sinthetic_indicator(cmd)
    elif cmd_param_list[0] == "DisplayOrderRoutingScreen":
        process_display_order_routing_screen(cmd)
    elif cmd_param_list[0] == "BiasMainLandingPage":
        process_bias_main_landing_page(cmd)

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
