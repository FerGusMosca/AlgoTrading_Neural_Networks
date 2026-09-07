import traceback

from business_entities.portf_position import PortfolioPosition
from common.enums.folders import Folders
from common.enums.information_vendors import InformationVendors
from common.enums.market_regimes import MarketRegimes
from common.util.financial_calculations.date_handler import DateHandler
from common.util.logging.logger import Logger
from common.util.std_in_out.ml_settings_loader import MLSettingsLoader
from common.util.std_in_out.param_reader import ParamReader
from controllers.main_dashboard_controller import MainDashboardController
from framework.common.logger.message_type import MessageType
from logic_layer.algos_orchestation_logic import AlgosOrchestationLogic
from logic_layer.sec_metadata_orchestation_logic import SECMetadataOrchestationLogic
from IPython.display import display
import pandas as pd

from logic_layer.data_set_builder import DataSetBuilder
from controllers.routing_dashboard_controller import RoutingDashboardController

_DATE_FORMAT = "%m/%d/%Y"
_TIMESTAMP_FORMAT='%m/%d/%Yt%H:%M:%S'

last_trading_dict = None

Folders.load_from_config()
def show_commands():
    print("======================== Financial Algos ========================")
    print("#1-EvalBiasedTradingAlgo [Symbol] [SeriesCSV] [from] [to] [Bias] [classif_key]")
    print("#2-EvalSlidingBiasedTradingAlgo [Symbol] [SeriesCSV] [from] [to] [init_portf_size] [trade_comm] [sliding_window_years] [sliding_window_months] [classif_key] ")
    print("#3-EvaluateARIMA [Symbol] [Period] [from] [to]")
    print("#4-PredictARIMA [Symbol] [p] [d] [q] [from] [to] [Period] [Step]")
    print("#5-EvalSingleIndicatorAlgo [Symbol] [indicator] [from] [to] [inverted] [classif_key]")
    print("#6-BacktestSlopeModel [symbol] [model_candle] [from] [to] [portf_size] [trade_comm] [trading_algo] [algo_params*]")
    print("#7-BacktestSlopeModelOnCustomETF [ETF_path] [model_candle] [from] [to] [portf_size] [trade_comm] [trading_algo] [algo_params*]")
    print("#8-CreateSintheticIndicator [comp_path] [model_candle] [from] [to] [slope_units]")
    print("#9-CustomRegimeSwitchDetector [variables] [from] [to] [regime_switch_filter] [regime_candle] [regime_window]")
    print("==================================================================")
    print("======================== Machine Learning ========================")
    print("#20-TrainLSTM [symbol] [variables_csv] [from] [to] [model_output] [classif_key] [epochs] [timestamps] [# neurons] [learning_rate] [reg_rate] [dropout_rate] [clipping_rate] [threshold_stop] [batch_size*] [inner_activation*] [make_stationary*]")
    print("#21-TrainLSTMWithGrouping [symbol] [variables_csv] [from] [to] [model_output] [classif_key] [epochs] [timestamps] [# neurons] [learning_rate] [reg_rate] [dropout_rate] [clipping_rate] [threshold_stop] [grouping_unit] [grouping_classif_criteria] [batch_size*] [inner_activation*]")
    print("#22-TestDailyLSTM [symbol] [variables_csv] [from] [to] [timestemps] [model_to_use] [portf_size] [trade_comm] [trading_algo] [classif_threshold] [algo_params*]")
    print("#23-TestDailyLSTMWithGrouping [symbol] [variables_csv] [from] [to] [timestemps] [model_to_use] [portf_size] [trade_comm] [trading_algo] [grouping_unit] [algo_params*]")
    print("#24-TrainXGBoost [symbol] [variables_csv] [from] [to] [model_output] [classif_key] [n_estimators] [max_depth] [learning_rate] [subsample] [colsample_bytree] [batch_size*] [grouping_unit*] [grouping_classif_criteria*] [group_as_mov_avg*] [grouping_mov_avg_unit*] [class_weight*] [make_stationary*] [interval*]")
    print("#25-TestXGBoost [symbol] [variables_csv] [from] [to] [model_to_use] [price_to_use*] [classif_key*] [normalize*] [make_stationary*] [threshold*]")
    print("#26-EvalMLBiasedAlgo [Symbol] [indicator] [SeriesCSV] [from] [to] [inverted] [classif_key]")
    print("#27-TrainNeuralNetworkAlgo [symbol] [variables_csv] [from] [to] [depth] [learning_rate] [iterations] [model_output] [classif_key]")
    print("#28-BacktestNeuralNetworkAlgo [symbol] [variables_csv] [from] [to] [model_to_use] [classif_key]")
    print("#29-TrainMLAlgos  [SeriesCSV] [from] [to] [classif_key]")
    print("#30-RunPredictionsLastModel [SeriesCSV] [from] [to] [classif_key]")
    print("#31-TrainRF [symbol] [variables_csv] [from] [to] [model_output] [classif_key] [n_estimators] [max_depth] [min_samples_split] [criterion] [batch_size*] [grouping_unit*] [grouping_classif_criteria*] [group_as_mov_avg*] [grouping_mov_avg_unit*] [class_weight*] [make_stationary*] [interval*]")
    print("#32-TestDailyRF [symbol] [series_csv] [from] [to] [model_to_use] [portf_size] [trade_comm] [trading_algo] [classif_threshold] [algo_params*]")
    print("#33-EvalSlidingRandomForest [symbol] [series_csv] [from] [to] [classif_key] [init_portf_size] [trade_comm] [classif_threshold] [sliding_window_years] [sliding_window_months]")

    print("==================================================================")
    print("======================== UI ========================")
    print("#40-BiasMainLandingPage")
    print("#41-DisplayOrderRoutingScreen")
    print("#42-DailyCandlesGraph [Symbol] [date] [interval] [mmov_avg]")
    print("#43-IndicatorCandlesGraph [Symbol] [from] [to] [interval] [mmov_avg]")
    print("==================================================================")
    print("======================== Financial Reports ========================")
    print("#60-DownloadFinancialData [symbol] [from*] [to*] [vendor_params*]")
    print("#61-DownloadFinancialDataBulk [symbols] [from*] [to*] [vendor_params*]")
    print("#62-CreateLightweightIndicator [csv_indicators] [from*] [to*] [benchmark*] [plot_result*]")
    print("#63-CreateSpreadVariable [diff_indicators] [from*] [to*] [output_symbol]")
    print("#64-CreateSpreadVariableBulk [diff_indicators*] [output_symbols*] [from*]")
    print("#65-DownloadSECSecurities")
    print("#71-DownloadSECSecuritiesMetadata [top*] [retry_errors*]")
    print("#72-DownloadSECSecurityMetadata [symbol]")
    print("#73-TagSecuritiesFromCSV [tag] [csv_path] [tag_group*]")
    #print("#66-RunReport [report*] [year*]")
    print("#67-DownloadBCRAInterestRates [from*] [to*]")
    print("#68-DownloadBYMAInterestRates [from*] [to*]")
    print("#69-CreateRatioVariable [numerator] [denominator] [multiplier] [from*] [to*] [output_symbol]")
    print("#70-CreateAverageVariable [symbol] [output_symbol] [from*] [to*]")

    print("==================================================================")
    #TrainNeuralNetworkAlgo
    print("#n-Exit")


def process_backtest_slope_model(cmd):
    symbol = ParamReader.get_param(cmd, "symbol")
    model_candle = ParamReader.get_param(cmd, "model_candle")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    portf_size = ParamReader.get_param(cmd, "portf_size",optional=True,def_value=100000)

    trading_algo = ParamReader.get_param(cmd, "trading_algo")
    candle_slope = ParamReader.get_param(cmd, "candle_slope", True, None)
    slope_units = ParamReader.get_param(cmd, "slope_units", True, None)

    trade_comm = ParamReader.get_param(cmd, "trade_comm", optional=True, def_value=0)
    trade_comm_pct = ParamReader.get_param(cmd, "trade_comm_pct", optional=True, def_value=0)

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

    fund_mgmt_dashboard_cs = config_settings["fund_mgmt_dashboard_cs"]


    wmc= RoutingDashboardController(logger, ib_prod_ws, primary_prod_ws, ib_dev_ws,fund_mgmt_dashboard_cs)

    wmc.display_order_routing_screen()
    print(f"Web Manager Logic successfully shown...")

def process_download_financial_data_bulk(cmd):
    # Required parameters
    symbols_str = ParamReader.get_param(cmd, "symbol")  # NOT 'symbols', para respetar el formato que vos diste
    d_from = ParamReader.get_param(cmd, "from", True, None)
    d_to = ParamReader.get_param(cmd, "to", True, None)

    raw_symbols = [s.strip() for s in symbols_str.split(",") if s.strip() != ""]

    for full_symbol in raw_symbols:
        parts = full_symbol.split(".")

        if len(parts) == 2:
            # Format: SYMBOL.VENDOR (FRED)
            symbol = parts[0]
            vendor = parts[1].upper()
            exchange = None
        elif len(parts) == 3:
            # Format: SYMBOL.EXCHANGE.VENDOR (TRADINGVIEW)
            symbol = parts[0]
            exchange = parts[1]
            vendor = parts[2].upper()
        else:
            raise Exception(f"[BULK] Invalid symbol format: '{full_symbol}'")

        # Build vendor_params dynamically
        vendor_params = {}

        if vendor == InformationVendors.FRED.value:
            pass  # no additional params needed

        elif vendor == InformationVendors.TRADINGVIEW.value:
            for key in ["session", "token", "username", "password", "interval"]:
                val = ParamReader.get_param(cmd, key, True, None)
                if val is not None:
                    vendor_params[key] = val
            if exchange is not None:
                vendor_params["exchange"] = exchange.replace("_", " ")
        else:
            raise Exception(f"[BULK] Unsupported vendor: '{vendor}'")

        cmd_param_dict = {
            "symbol": symbol,
            "vendor": vendor,
            "vendor_params": vendor_params
        }

        process_download_financial_data_logic_bulk(symbol, d_from, d_to, cmd_param_dict)



def process_download_financial_data(cmd):
    # Required parameters
    symbol = ParamReader.get_param(cmd, "symbol")
    output_symbol = ParamReader.get_param(cmd, "output_symbol",True,None)
    d_from = ParamReader.get_param(cmd, "from", True, None)
    d_to = ParamReader.get_param(cmd, "to", True, None)

    # Required vendor
    vendor = ParamReader.get_param(cmd, "vendor")

    # Build vendor_params dict from known optional parameters
    vendor_params = {}
    if output_symbol is not None:
        vendor_params ["output_symbol"]=output_symbol

    if vendor == InformationVendors.FRED.value:
        # No additional inline parameters expected for now
        pass

    elif vendor == InformationVendors.TRADINGVIEW.value:

        for key in ["session", "token", "username", "password", "interval", "exchange"]:
            val = ParamReader.get_param(cmd, key, True, None)
            if key == "exchange" and val is not None:
                val=val.replace("_"," ")

            if val is not None:
                vendor_params[key] = val
    else:
        raise Exception(f"Unsupported vendor: {vendor}")

    # Compose param dictionary to pass to logic
    cmd_param_dict = {
        "symbol": symbol,
        "vendor": vendor,
        "vendor_params": vendor_params
    }

    # Call core logic method
    process_download_financial_data_logic(symbol=symbol,
                                          d_from=d_from,
                                          d_to=d_to,
                                          algo_params=cmd_param_dict)


#
def process_create_spread_variable(cmd):
    # Required
    diff_indicators = ParamReader.get_param(cmd, "diff_indicators")
    output_symbol = ParamReader.get_param(cmd, "output_symbol", True, "LIGHTWEIGHT_INDICATOR")

    # Optional
    d_from = ParamReader.get_param(cmd, "from", True, None)
    d_to = ParamReader.get_param(cmd, "to", True, None)


    # Run core logic
    process_create_spread_variable_logic(diff_indicators=diff_indicators, d_from=d_from, d_to=d_to,output_symbol=output_symbol)

def process_download_sec_securities(cmd):
    # No parameters required, always download all securities
    process_download_sec_securities_logic()


def process_download_sec_securities_metadata(cmd):
    top = ParamReader.get_param(cmd, "top", True, None)
    retry_errors = ParamReader.get_param(cmd, "retry_errors", True, False)
    process_download_sec_securities_metadata_logic(top, retry_errors)


def process_download_sec_security_metadata(cmd):
    symbol = ParamReader.get_param(cmd, "symbol")
    process_download_sec_security_metadata_logic(symbol)


def process_tag_securities_from_csv(cmd):
    tag = ParamReader.get_param(cmd, "tag")
    csv_path = ParamReader.get_param(cmd, "csv_path")
    tag_group = ParamReader.get_param(cmd, "tag_group", True, "CUSTOM")
    process_tag_securities_from_csv_logic(tag, csv_path, tag_group)



def process_create_spread_variable_bulk(cmd):
    # Required parameters
    diff_str = ParamReader.get_param(cmd, "diff_indicators")
    output_str = ParamReader.get_param(cmd, "output_symbols")
    d_from = ParamReader.get_param(cmd, "from", True, None)

    diff_indicators = [d.strip() for d in diff_str.split(",") if d.strip() != ""]
    output_symbols = [o.strip() for o in output_str.split(",") if o.strip() != ""]

    if len(diff_indicators) != len(output_symbols):
        raise Exception("[BULK][SPREAD] ❌ Mismatch between diff_indicators and output_symbols count")

    # Paso a lógica como en el comando anterior
    process_create_spread_variable_bulk_logic(diff_indicators, output_symbols, d_from)


def process_create_ratio_variable(cmd):
    # Required
    numerator = ParamReader.get_param(cmd, "numerator")
    denominator = ParamReader.get_param(cmd, "denominator")
    multiplier = float(ParamReader.get_param(cmd, "multiplier", True, 1))
    output_symbol = ParamReader.get_param(cmd, "output_symbol", True, "LIGHTWEIGHT_INDICATOR")

    # Optional
    d_from = ParamReader.get_param(cmd, "from", True, None)
    d_to = ParamReader.get_param(cmd, "to", True, None)

    # Run core logic
    process_create_ratio_variable_logic(
        numerator=numerator,
        denominator=denominator,
        multiplier=multiplier,
        d_from=d_from,
        d_to=d_to,
        output_symbol=output_symbol
    )

def process_create_average_variable(cmd):
    symbols = ParamReader.get_param(cmd, "symbol")               # CSV
    output_symbol = ParamReader.get_param(cmd, "output_symbol")  # required

    d_from = ParamReader.get_param(cmd, "from", True, None)
    d_to   = ParamReader.get_param(cmd, "to", True, None)

    process_create_average_variable_logic(
        symbols=symbols,
        output_symbol=output_symbol,
        d_from=d_from,
        d_to=d_to
    )


def process_create_lightweight_indicator_logic(csv_indicators, d_from, d_to,output_symbol, benchmark=None, plot_result=True):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(f"🧪 Starting lightweight indicator creation with {len(csv_indicators.split(','))} input variables", MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        trd_algos = AlgosOrchestationLogic(
            config_settings["hist_data_conn_str"],
            config_settings["ml_reports_conn_str"],
            None,
            logger
        )

        trd_algos.process_create_lightweight_indicator(csv_indicators=csv_indicators, d_from=d_from, d_to=d_to,
                                                       output_symbol=output_symbol,
                                                       benchmark=benchmark,plot_result=plot_result
                                                       )

        logger.print("✅ Lightweight indicator successfully created and persisted", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"CRITICAL ERROR running process_create_lightweight_indicator_logic: {str(e)}", MessageType.ERROR)


def process_create_lightweight_indicator(cmd):
    # Required
    csv_indicators = ParamReader.get_param(cmd, "csv_indicators")


    # Optional
    plot_result = ParamReader.get_param(cmd, "plot_result", True, False)
    benchmark = ParamReader.get_param(cmd, "benchmark", True, None)
    d_from = ParamReader.get_param(cmd, "from", True, None)
    d_to = ParamReader.get_param(cmd, "to", True, None)
    output_symbol = ParamReader.get_param(cmd, "output_symbol", True, "LIGHTWEIGHT_INDICATOR")

    # Run core logic
    process_create_lightweight_indicator_logic(csv_indicators=csv_indicators, d_from=d_from, d_to=d_to,
                                               benchmark=benchmark, plot_result=plot_result,output_symbol=output_symbol)


def process_create_sinthetic_indicator(cmd):
    comp_apth = ParamReader.get_param(cmd, "comp_path")
    model_candle = ParamReader.get_param(cmd, "model_candle")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")

    #DIRECT/INV SLOPE
    slope_units = ParamReader.get_param(cmd, "slope_units", True, None)

    #ARIMA
    p = ParamReader.get_param(cmd, "p", True, None)
    d = ParamReader.get_param(cmd, "d", True, None)
    q = ParamReader.get_param(cmd, "q", True, None)
    step = ParamReader.get_param(cmd, "step", True, None)
    inv_steps = ParamReader.get_param(cmd, "inv_steps", True, None)
    min_units_to_pred = ParamReader.get_param(cmd, "min_units_to_pred", True, None)

    #SARIMA
    s = ParamReader.get_param(cmd, "s", True, None)

    #POS_THRESHOLDS ind
    pos_threshold = ParamReader.get_param(cmd, "pos_threshold", True, None)

    #SUDDEN_STOP
    st_units = ParamReader.get_param(cmd, "st_units", True, None)
    st_eval_p = ParamReader.get_param(cmd, "st_eval_p", True, None)
    st_blackout_p = ParamReader.get_param(cmd, "st_blackout_p", True, None)

    cmd_param_dict = {}

    if slope_units is not None:
        cmd_param_dict["slope_units"] = slope_units

    if p is not None:
        cmd_param_dict["p"] = p

    if d is not None:
        cmd_param_dict["d"] = d

    if q is not None:
        cmd_param_dict["q"] = q

    if s is not None:
        cmd_param_dict["s"] = s

    if step is not None:
        cmd_param_dict["step"] = step

    if inv_steps is not None:
        cmd_param_dict["inv_steps"] = inv_steps

    if min_units_to_pred is not None:
        cmd_param_dict["min_units_to_pred"] = min_units_to_pred

    if pos_threshold is not None:
        cmd_param_dict["pos_threshold"] = pos_threshold

    if st_units is not None:
        cmd_param_dict["st_units"] = st_units

    if st_eval_p is not None:
        cmd_param_dict["st_eval_p"] = st_eval_p

    if st_blackout_p is not None:
        cmd_param_dict["st_blackout_p"] = st_blackout_p

    process_create_sinthetic_indicator_logic(comp_apth, model_candle=model_candle, d_from=d_from, d_to=d_to,
                                            algo_params=cmd_param_dict)

    print(f"Create sinthetic indicator finished...")
def process_backtest_slope_model_on_custom_etf(cmd):
    etf_path = ParamReader.get_param(cmd, "ETF_path")
    model_candle = ParamReader.get_param(cmd, "model_candle")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    portf_size = ParamReader.get_param(cmd, "portf_size",optional=True,def_value=100000)

    trading_algo = ParamReader.get_param(cmd, "trading_algo")
    candle_slope = ParamReader.get_param(cmd, "candle_slope", True, None)
    slope_units = ParamReader.get_param(cmd, "slope_units", True, None)

    trade_comm = ParamReader.get_param(cmd, "trade_comm", optional=True, def_value=0)
    trade_comm_pct = ParamReader.get_param(cmd, "trade_comm_pct", optional=True, def_value=0)

    days_to_add_to_date = ParamReader.get_param(cmd, "days_to_add_to_date", optional=True, def_value=None)

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

def process_test_XGBoost_cmd(cmd):
    symbol = ParamReader.get_param(cmd, "symbol")
    series_csv = ParamReader.get_param(cmd, "series_csv")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    model_to_use = ParamReader.get_param(cmd, "model_to_use")

    # Optional parameters
    interval = ParamReader.get_param(cmd, "interval", True, def_value=DataSetBuilder._1_DAY_INTERVAL)
    init_portf_size = float(ParamReader.get_param(cmd, "init_portf_size", True, def_value=100000))
    trade_comm = float(ParamReader.get_param(cmd, "trade_comm", True, def_value=0.0))
    draw_predictions = ParamReader.get_bool_param(cmd, "draw_predictions", True, def_value=False)
    grouping_unit = ParamReader.get_param(cmd, "grouping_unit", True)
    grouping_classif_criteria = ParamReader.get_param(cmd, "grouping_classif_criteria", True, def_value=None)
    group_as_mov_avg = ParamReader.get_bool_param(cmd, "group_as_mov_avg", True, def_value=False)
    grouping_mov_avg_unit = ParamReader.get_param(cmd, "grouping_mov_avg_unit", True, def_value=100)
    lower_percentile_limit = float(ParamReader.get_param(cmd, "lower_percentile_limit", True, def_value=0.5))
    lower_prob_limit_raw = ParamReader.get_param(cmd, "lower_prob_limit", True, def_value=None)
    lower_prob_limit = float(lower_prob_limit_raw) if lower_prob_limit_raw is not None else None
    make_stationary = ParamReader.get_bool_param(cmd, "make_stationary", True, def_value=False)
    n_flip = int(ParamReader.get_param(cmd, "n_flip", True, def_value=3))
    bias = ParamReader.get_param(cmd, "bias", True, def_value=None)
    pos_regime_filters_csv = ParamReader.get_param(cmd, "pos_regime_filters_csv", True, def_value=None)
    neg_regime_filters_csv = ParamReader.get_param(cmd, "neg_regime_filters_csv", True, def_value=None)

    # Compose param dictionaryy
    n_algo_param_dict = {
        "interval": interval.replace("_", " "),
        "init_portf_size": init_portf_size,
        "series_csv": series_csv,
        "trade_comm": trade_comm,
        "grouping_unit": int(grouping_unit) if grouping_unit is not None else None,
        "grouping_classif_criteria": grouping_classif_criteria,
        "group_as_mov_avg": group_as_mov_avg,
        "grouping_mov_avg_unit": int(grouping_mov_avg_unit) if grouping_mov_avg_unit is not None else None,
        "make_stationary": make_stationary,
        "n_flip": n_flip,
        "lower_percentile_limit": lower_percentile_limit,
        "lower_prob_limit": lower_prob_limit,
        "bias": bias,
        "draw_predictions": draw_predictions,
        "pos_regime_filters_csv": pos_regime_filters_csv,
        "neg_regime_filters_csv": neg_regime_filters_csv
    }

    # Run the test
    process_test_daily_XGBoost(
        symbol=symbol,
        series_csv=series_csv,
        d_from=d_from,
        d_to=d_to,
        model_to_use=model_to_use,
        n_algo_param_dict=n_algo_param_dict
    )

    print("Test XGBoost successfully finished...")

def process_test_RF_cmd(cmd):
    symbol = ParamReader.get_param(cmd, "symbol")
    series_csv = ParamReader.get_param(cmd, "series_csv")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    model_to_use = ParamReader.get_param(cmd, "model_to_use")

    # Optional parameters
    interval = ParamReader.get_param(cmd, "interval", True, DataSetBuilder._1_DAY_INTERVAL)
    init_portf_size = float(ParamReader.get_param(cmd, "init_portf_size"))
    trade_comm = float(ParamReader.get_param(cmd, "trade_comm"))
    draw_predictions = ParamReader.get_param(cmd, "draw_predictions", optional=True, def_value=False)
    grouping_unit = ParamReader.get_param(cmd, "grouping_unit", True)
    grouping_classif_criteria = ParamReader.get_param(cmd, "grouping_classif_criteria", True, def_value=None)
    group_as_mov_avg = ParamReader.get_bool_param(cmd, "group_as_mov_avg", True, def_value=False)
    grouping_mov_avg_unit = ParamReader.get_param(cmd, "grouping_mov_avg_unit", True, def_value=100)
    classif_threshold = ParamReader.get_param(cmd, "classif_threshold", True, def_value=0.5)
    make_stationary = ParamReader.get_bool_param(cmd, "make_stationary", True, False)
    n_flip = int(ParamReader.get_param(cmd, "n_flip", True, 3))
    bias = ParamReader.get_param(cmd, "bias", True, None)
    pos_regime_filters_csv = ParamReader.get_param(cmd, "pos_regime_filters_csv", True, None)
    neg_regime_filters_csv = ParamReader.get_param(cmd, "neg_regime_filters_csv", True, None)

    # Create parameter dictionary to be passed to test logic
    n_algo_param_dict = {
        "interval": interval.replace("_", " "),
        "init_portf_size": init_portf_size,
        "series_csv": series_csv,
        "trade_comm": trade_comm,
        "grouping_unit": int(grouping_unit) if grouping_unit is not None else None,
        "grouping_classif_criteria": grouping_classif_criteria,
        "group_as_mov_avg":group_as_mov_avg,
        "grouping_mov_avg_unit": int(grouping_mov_avg_unit) if grouping_mov_avg_unit is not None else None,
        "make_stationary": make_stationary,
        "n_flip": n_flip,
        "classif_threshold": classif_threshold,
        "bias": bias,
        "draw_predictions": draw_predictions,
        "pos_regime_filters_csv": pos_regime_filters_csv,
        "neg_regime_filters_csv": neg_regime_filters_csv
    }

    # Run the daily RF test process
    process_test_daily_RF(
        symbol=symbol,
        series_csv=series_csv,
        d_from=d_from,
        d_to=d_to,
        model_to_use=model_to_use,
        n_algo_param_dict=n_algo_param_dict
    )

    print("Test RF successfully finished...")




def process_test_LSTM_cmd(cmd):
    symbol = ParamReader.get_param(cmd, "symbol")
    variables_csv = ParamReader.get_param(cmd, "variables_csv")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    timesteps=ParamReader.get_param(cmd,"timesteps")
    model_to_use = ParamReader.get_param(cmd, "model_to_use")
    portf_size=ParamReader.get_param(cmd,"portf_size")
    comm=ParamReader.get_param(cmd,"comm")
    interval=ParamReader.get_param(cmd,"interval",True,None)
    trading_algo=ParamReader.get_param(cmd,"trading_algo")
    grouping_unit=ParamReader.get_param(cmd,"grouping_unit",True,None)
    n_buffer=ParamReader.get_param(cmd,"n_buffer",True,None)
    mov_avg=ParamReader.get_param(cmd,"mov_avg",True,None)
    use_sliding_window = ParamReader.get_param(cmd, "use_sliding_window", True,def_value="None")#NONE,CUT_INPUT_DF,GET_FAKE_DATA
    make_stationary = ParamReader.get_bool_param(cmd, "make_stationary", True, False)
    classif_threshold = ParamReader.get_param(cmd, "classif_threshold", True, 0.5)

    cmd_param_list=[]
    if n_buffer is not None:
        cmd_param_list.append(n_buffer)

    if mov_avg is not None:
        cmd_param_list.append(mov_avg)


    process_test_daily_LSTM(symbol=symbol, variables_csv=variables_csv, d_from=d_from, d_to=d_to,
                            timesteps=timesteps,model_to_use=model_to_use, portf_size=portf_size,
                            trade_comm=comm, trading_algo=trading_algo,interval=interval,
                            grouping_unit=grouping_unit,n_params=cmd_param_list,
                            use_sliding_window=use_sliding_window,make_stationary=make_stationary,
                            classif_threshold=classif_threshold)

    print(f"Test LSTM successfully finished...")


def process_train_XGBoost(symbol, series_csv, d_from, d_to, model_output, classif_key,
                          n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
                          grouping_unit=None, grouping_classif_criteria=None,
                          group_as_mov_avg=False, grouping_mov_avg_unit=100,
                          interval=None, make_stationary=False, class_weight=None,register_model=None):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(f"Initializing dataframe creation for series: {series_csv}", MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(
            config_settings["hist_data_conn_str"],
            config_settings["ml_reports_conn_str"],
            config_settings["classification_map_key"] if classif_key is None else classif_key,
            logger
        )

        dataMgm.process_train_XGBoost(symbol=symbol, series_csv=series_csv, d_from=d_from, d_to=d_to,
                                      model_output=model_output.replace('"', ""),
                                      classif_key=classif_key,
                                      n_estimators=int(n_estimators),
                                      max_depth=None if str(max_depth).lower() == "none" else int(max_depth),
                                      learning_rate=float(learning_rate),
                                      subsample=float(subsample),
                                      colsample_bytree=float(colsample_bytree),
                                      interval=interval.replace('_', " ") if interval is not None else None,
                                      grouping_unit=int(grouping_unit) if grouping_unit is not None else None,
                                      grouping_classif_criteria=grouping_classif_criteria,
                                      group_as_mov_avg=bool(group_as_mov_avg),
                                      grouping_mov_avg_unit=int(grouping_mov_avg_unit),
                                      make_stationary=make_stationary,
                                      class_weight=None if class_weight is None or str(class_weight).lower() == "none" else class_weight,
                                      register_model=register_model)

        logger.print(f"XGBoost model successfully trained for symbol {symbol} and variables {series_csv}", MessageType.INFO)

    except Exception as e:
        logger.print(f"CRITICAL ERROR running process_train_XGBoost: {str(e)}", MessageType.ERROR)


def process_train_RF(symbol, series_csv, d_from, d_to, model_output, classification_key,
                     n_estimators, max_depth, min_samples_split, criterion,
                     grouping_unit=None, grouping_classif_criteria=None,
                     group_as_mov_avg=False, grouping_mov_avg_unit=100,
                     interval=None, make_stationary=False, class_weight=None):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(f"Initializing dataframe creation for series : {series_csv}", MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(
            config_settings["hist_data_conn_str"],
            config_settings["ml_reports_conn_str"],
            config_settings["classification_map_key"] if classification_key is None else classification_key,
            logger
        )

        dataMgm.process_train_RF(symbol=symbol, series_csv= series_csv, d_from=d_from, d_to=d_to,
                                 model_output=model_output.replace('"', ""), classif_key=classification_key,
                                 n_estimators=int(n_estimators),
                                 max_depth=None if str(max_depth).lower() == "none" else int(max_depth),
                                 min_samples_split=int(min_samples_split), criterion=criterion,
                                 interval=interval.replace('_', " ") if interval is not None else None,
                                 grouping_unit=int(grouping_unit) if grouping_unit is not None else None,
                                 grouping_classif_criteria=grouping_classif_criteria,
                                 group_as_mov_avg=bool(group_as_mov_avg),
                                 grouping_mov_avg_unit=int(grouping_mov_avg_unit), make_stationary=make_stationary,
                                 class_weight=None if class_weight is None or str(
                                     class_weight).lower() == "none" else class_weight)

        logger.print(f"Random Forest model successfully trained for symbol {symbol} and variables {series_csv}",
                     MessageType.INFO)

    except Exception as e:
        logger.print(f"CRITICAL ERROR running process_train_RF: {str(e)}", MessageType.ERROR)

#
def process_train_XGBoost_cmd(cmd, cmd_param_list):
    # Required parameters
    symbol = ParamReader.get_param(cmd, "symbol")
    series_csv = ParamReader.get_param(cmd, "series_csv")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    model_output = ParamReader.get_param(cmd, "model_output")
    classif_key = ParamReader.get_param(cmd, "classif_key")

    # XGBoost-specific hyperparameters
    n_estimators = int(ParamReader.get_param(cmd, "n_estimators", True, def_value=100))
    max_depth = ParamReader.get_param(cmd, "max_depth", True, def_value=3)
    max_depth = None if str(max_depth).lower() == "none" else int(max_depth)

    learning_rate = float(ParamReader.get_param(cmd, "learning_rate", True, def_value=0.1))
    subsample = float(ParamReader.get_param(cmd, "subsample", True, def_value=1.0))
    colsample_bytree = float(ParamReader.get_param(cmd, "colsample_bytree", True, def_value=1.0))

    class_weight = ParamReader.get_param(cmd, "class_weight", True, def_value=None)
    class_weight = None if class_weight is None or str(class_weight).lower() == "none" else class_weight

    # Optional common flags
    interval = ParamReader.get_param(cmd, "interval", True, def_value=DataSetBuilder._1_DAY_INTERVAL)
    grouping_unit = ParamReader.get_param(cmd, "grouping_unit", True)
    grouping_classif_criteria = ParamReader.get_param(cmd, "grouping_classif_criteria", True)
    group_as_mov_avg = ParamReader.get_bool_param(cmd, "group_as_mov_avg", True, def_value=False)
    grouping_mov_avg_unit = ParamReader.get_param(cmd, "grouping_mov_avg_unit", True, def_value=100)
    make_stationary = ParamReader.get_bool_param(cmd, "make_stationary", True, def_value=False)

    #MFLow
    register_model = ParamReader.get_bool_param(cmd, "register_model", True, def_value=False)

    # Call processing method with parsed params
    process_train_XGBoost(symbol=symbol,
                          series_csv=series_csv,
                          d_from=d_from,
                          d_to=d_to,
                          model_output=model_output,
                          classif_key=classif_key,
                          n_estimators=n_estimators,
                          max_depth=max_depth,
                          learning_rate=learning_rate,
                          subsample=subsample,
                          colsample_bytree=colsample_bytree,
                          interval=interval,
                          grouping_unit=grouping_unit,
                          grouping_classif_criteria=grouping_classif_criteria,
                          group_as_mov_avg=group_as_mov_avg,
                          grouping_mov_avg_unit=grouping_mov_avg_unit,
                          class_weight=class_weight,
                          make_stationary=make_stationary,
                          register_model=register_model)

    print("Train XGBoost successfully finished...")


def process_train_RF_cmd(cmd, cmd_param_list):
    # Required parameters
    symbol = ParamReader.get_param(cmd, "symbol")
    series_csv = ParamReader.get_param(cmd, "series_csv")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    model_output = ParamReader.get_param(cmd, "model_output")
    classif_key = ParamReader.get_param(cmd, "classif_key")

    # RF-specific hyperparameters
    n_estimators = ParamReader.get_param(cmd, "n_estimators", True, def_value=100)
    max_depth = ParamReader.get_param(cmd, "max_depth", True, def_value=None)
    max_depth = None if str(max_depth).lower() == "none" else int(max_depth)
    class_weight = ParamReader.get_param(cmd, "class_weight", True, def_value=None)
    class_weight = None if class_weight is None or class_weight == "None" else class_weight

    min_samples_split = ParamReader.get_param(cmd, "min_samples_split", True, def_value=2)
    criterion = ParamReader.get_param(cmd, "criterion", True, def_value="gini")

    # Optional common flags
    interval = ParamReader.get_param(cmd, "interval", True, def_value=DataSetBuilder._1_DAY_INTERVAL)
    grouping_unit = ParamReader.get_param(cmd, "grouping_unit", True)
    grouping_classif_criteria = ParamReader.get_param(cmd, "grouping_classif_criteria", True)
    group_as_mov_avg = ParamReader.get_bool_param(cmd, "group_as_mov_avg", True, def_value=False)
    grouping_mov_avg_unit = ParamReader.get_param(cmd, "grouping_mov_avg_unit", True, def_value=100)
    make_stationary = ParamReader.get_bool_param(cmd, "make_stationary", True, def_value=False)

    # Call processing method with parsed params
    process_train_RF(symbol=symbol,
                     series_csv=series_csv,
                     d_from=d_from,
                     d_to=d_to,
                     model_output=model_output,
                     classification_key=classif_key,
                     n_estimators=int(n_estimators),
                     max_depth=None if max_depth is None else int(max_depth),
                     min_samples_split=int(min_samples_split),
                     criterion=criterion,
                     interval=interval,
                     grouping_unit=grouping_unit,
                     grouping_classif_criteria=grouping_classif_criteria,
                     group_as_mov_avg=group_as_mov_avg,
                     grouping_mov_avg_unit=grouping_mov_avg_unit,
                     class_weight=class_weight,
                     make_stationary=make_stationary)

    print(f"Train RF successfully finished...")


def process_traing_LSTM_cmd(cmd,cmd_param_list):
    symbol = ParamReader.get_param(cmd, "symbol")
    variables_csv = ParamReader.get_param(cmd, "variables_csv")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    model_output = ParamReader.get_param(cmd, "model_output")
    classif_key = ParamReader.get_param(cmd, "classif_key")
    epochs = ParamReader.get_param(cmd, "epochs")
    n_neurons = ParamReader.get_param(cmd, "n_neurons")
    timesteps = ParamReader.get_param(cmd, "timesteps")
    learning_rate = ParamReader.get_param(cmd, "learning_rate")
    dropout_rate = ParamReader.get_param(cmd, "dropout_rate")
    clipping_rate = ParamReader.get_param(cmd, "clipping_rate")
    reg_rate = ParamReader.get_param(cmd, "reg_rate")
    threshold_stop = ParamReader.get_param(cmd, "threshold_stop")
    interval = ParamReader.get_param(cmd, "interval",True,None)
    grouping_unit=ParamReader.get_param(cmd,"grouping_unit",True)
    grouping_classif_criteria=ParamReader.get_param(cmd,"grouping_classif_criteria",True)
    group_as_mov_avg=ParamReader.get_bool_param(cmd,"grouping_classif_criteria",True,def_value=False)
    grouping_mov_avg_unit=ParamReader.get_param(cmd,"grouping_mov_avg_unit",True,def_value=100)
    batch_size = ParamReader.get_param(cmd, "batch_size", True, def_value=1)
    inner_activation = ParamReader.get_param(cmd, "inner_activation", True, def_value=None)
    make_stationary = ParamReader.get_bool_param(cmd, "make_stationary", True,False)


    process_train_LSTM(symbol=symbol, variables_csv=variables_csv, d_from=d_from, d_to=d_to, model_output=model_output,
                       classification_key=classif_key, epochs=epochs, timestamps=timesteps, n_neurons=n_neurons,
                       learning_rate=learning_rate, reg_rate=reg_rate, dropout_rate=dropout_rate,
                       clipping_rate=clipping_rate, threshold_stop=threshold_stop, grouping_unit=grouping_unit,
                       grouping_classif_criteria=grouping_classif_criteria, group_as_mov_avg=group_as_mov_avg,
                       grouping_mov_avg_unit=grouping_mov_avg_unit, interval=interval, batch_size=batch_size,
                       inner_activation=inner_activation, make_stationary=make_stationary)  # will use default

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
    series_csv = ParamReader.get_param(cmd, "series_csv")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    classif_key = ParamReader.get_param(cmd, "classif_key")
    process_train_ml_algos(series_csv, d_from, d_to, classif_key)




def run_custom_regime_switch_detector(cmd):
    variables = ParamReader.get_param(cmd, "variables").split(",")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    regime_filter = ParamReader.get_param(cmd, "regime_switch_filter")
    regime_candle = ParamReader.get_param(cmd, "regime_candle")
    regime_window = int(ParamReader.get_param(cmd, "regime_window", optional=True, def_value=20))
    slope_threshold = float(ParamReader.get_param(cmd, "slope_threshold", optional=True, def_value=0.3))

    abs_value_threshold=None
    if regime_filter==MarketRegimes.ABS_VALUE.value:
        abs_value_threshold = float(ParamReader.get_param(cmd, "abs_value_threshold", optional=True, def_value=None))

    process_custom_regime_switch_detector(
        variables, d_from, d_to, regime_filter, regime_candle, regime_window, slope_threshold,abs_value_threshold
    )



def run_sliding_random_forest(cmd):
    symbol = ParamReader.get_param(cmd, "symbol")
    series_csv = ParamReader.get_param(cmd, "series_csv")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    classif_key = ParamReader.get_param(cmd, "classif_key")

    trade_comm = ParamReader.get_param(cmd, "trade_comm", optional=True, def_value=5)
    draw_predictions = ParamReader.get_param(cmd, "draw_predictions", optional=True, def_value=False)
    init_portf_size = ParamReader.get_param(cmd, "init_portf_size", optional=True, def_value=PortfolioPosition._DEF_PORTF_AMT)
    sliding_window_years = ParamReader.get_param(cmd, "sliding_window_years", optional=True, def_value=2)
    sliding_window_months = ParamReader.get_param(cmd, "sliding_window_months", optional=True, def_value=2)
    class_weight = ParamReader.get_param(cmd, "class_weight", optional=True, def_value="balanced")
    bias = ParamReader.get_param(cmd, "bias", optional=True, def_value="NONE")
    n_flip = ParamReader.get_param(cmd, "n_flip", optional=True, def_value=1)
    classif_threshold = ParamReader.get_param(cmd, "classif_threshold", optional=True, def_value=0.5)
    pos_regime_filters_csv = ParamReader.get_param(cmd, "pos_regime_filters_csv", optional=True, def_value="")
    neg_regime_filters_csv = ParamReader.get_param(cmd, "neg_regime_filters_csv", optional=True, def_value="")

    n_algo_param_dict = {
        "trade_comm": trade_comm,
        "init_portf_size": init_portf_size,
        "sliding_window_years": sliding_window_years,
        "sliding_window_months": sliding_window_months,
        "classif_key": classif_key,
        "class_weight": class_weight,
        "bias": bias,
        "draw_predictions": draw_predictions,
        "classif_threshold": classif_threshold,
        "n_flip": n_flip,
        "pos_regime_filters_csv": pos_regime_filters_csv,
        "neg_regime_filters_csv": neg_regime_filters_csv,
        "algos": "RF"
    }

    process_sliding_random_forest(symbol, series_csv, d_from, d_to, n_algo_param_dict)


def run_sliding_biased_trading_algo(cmd):
    symbol = ParamReader.get_param(cmd, "symbol")
    series_csv = ParamReader.get_param(cmd, "series_csv")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    bias = ParamReader.get_param(cmd, "bias",optional=True,def_value="NONE")
    classif_key = ParamReader.get_param(cmd, "classif_key")
    algos = ParamReader.get_param(cmd, "algos", optional=True, def_value=None)

    trade_comm = ParamReader.get_param(cmd, "trade_comm", optional=True, def_value=5)
    init_portf_size=ParamReader.get_param(cmd, "init_portf_size",optional=True,def_value=PortfolioPosition._DEF_PORTF_AMT)
    sliding_window_years = ParamReader.get_param(cmd, "sliding_window_years", optional=True, def_value=2)
    sliding_window_months = ParamReader.get_param(cmd, "sliding_window_months", optional=True, def_value=2)

    n_algo_param_dict = {}
    n_algo_param_dict["trade_comm"]=trade_comm
    n_algo_param_dict["init_portf_size"]=init_portf_size
    n_algo_param_dict["sliding_window_years"] = sliding_window_years
    n_algo_param_dict["sliding_window_months"] = sliding_window_months
    n_algo_param_dict["classif_key"] = classif_key
    n_algo_param_dict["algos"]=algos

    process_sliding_biased_trading_algo(symbol,series_csv, d_from, d_to,bias,n_algo_param_dict)

def run_biased_trading_algo(cmd):
    symbol = ParamReader.get_param(cmd, "symbol")
    series_csv = ParamReader.get_param(cmd, "series_csv")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    bias = ParamReader.get_param(cmd, "bias")

    trade_comm = ParamReader.get_param(cmd, "trade_comm", optional=True, def_value=5)
    init_portf_size=ParamReader.get_param(cmd, "init_portf_size",optional=True,def_value=PortfolioPosition._DEF_PORTF_AMT)

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
            print(f"Pct Profit. Size={summary.total_net_profit_str}")
            print(f"Est. Max Drawdown={summary.max_drawdown_on_MTM_str}")
            print("     =========== Portf Positions=========== ")
            for portf_pos in summary.portf_pos_summary:

                print(f"    --Side={portf_pos.side} Open Date={portf_pos.date_open} Close Date={portf_pos.date_close} Open Price={portf_pos.price_open} Close Price={portf_pos.price_close} --> Pct. Profit={portf_pos.calculate_pct_profit()}%")


    except Exception as e:
        logger.print("CRITICAL ERROR bootstrapping the system:{}".format(str(e)), MessageType.ERROR)


def process_sliding_random_forest(symbol, cmd_series_csv, d_from, d_to, n_algo_param_dict):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        global last_trading_dict
        logger.print("Evaluating trading performance for Random Forest model from {} to {}".format(d_from, d_to),
                     MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"],
                                         config_settings["ml_reports_conn_str"],
                                         n_algo_param_dict["classif_key"],
                                         logger)

        summary_dict_arr = dataMgm.sliding_train_and_evaluate_random_forest_performance(
            symbol, cmd_series_csv, d_from, d_to,
            last_trading_dict, n_algo_param_dict=n_algo_param_dict
        )

        last_trading_dict = summary_dict_arr[-1]

        print("Displaying all the different models predictions for the different algos:")

        for window in summary_dict_arr:
            for key in window.keys():
                print("============{}============ for {}".format(key, symbol))
                summary = window[key]
                print("From={} To={}".format(d_from, d_to))
                print(f"Init Portfolio={round(summary.portf_init_MTM, 2)} $")
                print(f"Final Portfolio={round(summary.portf_final_MTM, 2)} $")
                print(f"Pct Profit. Size={summary.total_net_profit_str}")
                print(f"Est. Max Drawdown={summary.max_drawdown_on_MTM_str}")
                print("     =========== Portf Positions=========== ")
                for portf_pos in summary.portf_pos_summary:
                    print(f"    --Side={portf_pos.side} Open Date={portf_pos.date_open} Close Date={portf_pos.date_close} "
                          f"Open Price={portf_pos.price_open} Close Price={portf_pos.price_close} --> Pct. Profit={portf_pos.calculate_pct_profit()}%")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Stack Trace:")
        traceback.print_exc()
        logger.print("CRITICAL ERROR running sliding RF evaluation: {}".format(str(e)), MessageType.ERROR)


def process_sliding_biased_trading_algo(symbol, cmd_series_csv, d_from, d_to, bias,n_algo_param_dict):
    loader = MLSettingsLoader()
    logger = Logger()
    try:
        global last_trading_dict
        logger.print("Evaluating trading performance for symbol from last model from {} to {}".format(d_from, d_to),
                     MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"], config_settings["ml_reports_conn_str"],
                                         n_algo_param_dict["classif_key"] ,logger)
        summary_dict_arr = dataMgm.sliding_train_and_evaluate_ml_performance(symbol, cmd_series_csv,
                                                                        d_from,d_to, bias,last_trading_dict,
                                                                        n_algo_param_dict=n_algo_param_dict)

        last_trading_dict = summary_dict_arr[-1]

        print("Displaying all the different models predictions for the different alogs:")

        for window in summary_dict_arr:
            for key in window.keys():

                print("============{}============ for {}".format(key, symbol))
                summary = window[key]
                print("From={} To={}".format(d_from, d_to))
                print(f"Init Portfolio={round(summary.portf_init_MTM, 2)} $")
                print(f"Final Portfolio={round(summary.portf_final_MTM,2)} $")
                print(f"Pct Profit. Size={summary.total_net_profit_str}")
                print(f"Est. Max Drawdown={summary.max_drawdown_on_MTM_str}")
                print("     =========== Portf Positions=========== ")
                for portf_pos in summary.portf_pos_summary:

                    print(f"    --Side={portf_pos.side} Open Date={portf_pos.date_open} Close Date={portf_pos.date_close} Open Price={portf_pos.price_open} Close Price={portf_pos.price_close} --> Pct. Profit={portf_pos.calculate_pct_profit()}%")


    except Exception as e:
        # Print the exception message
        print(f"An error occurred: {str(e)}")
        # Print the full stack trace
        print("Stack Trace:")
        traceback.print_exc()
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
    symbol = ParamReader.get_param(cmd, "symbol")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    period = ParamReader.get_param(cmd, "period",optional=True,def_value=None)

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
    symbol = ParamReader.get_param(cmd, "symbol")
    d_from = ParamReader.get_param(cmd, "from")
    d_to = ParamReader.get_param(cmd, "to")
    p = ParamReader.get_param(cmd, "p")
    d = ParamReader.get_param(cmd, "d")
    q = ParamReader.get_param(cmd, "q")
    step = ParamReader.get_param(cmd, "step")
    period = ParamReader.get_param(cmd, "period", optional=True, def_value=None)


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


def process_custom_regime_switch_detector(variables, d_from, d_to, regime_filter, regime_candle, regime_window,slope_threshold,abs_value_threshold):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(f"Running Regime Switch Detector from {d_from} to {d_to}", MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(config_settings["hist_data_conn_str"],
                                         config_settings["ml_reports_conn_str"],
                                         None,
                                         logger=logger)



        dataMgm.detect_and_save_regime_switch(
            variables=variables,
            d_from=d_from,
            d_to=d_to,
            regime_filter=regime_filter,
            regime_candle=regime_candle,
            regime_window=regime_window,
            slope_threshold=slope_threshold,
            abs_value_threshold=abs_value_threshold
        )

        logger.print(f"✅ Regime switch detection saved for candle: {regime_candle}", MessageType.INFO)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        logger.print("CRITICAL ERROR running CustomRegimeSwitchDetector: {}".format(str(e)), MessageType.ERROR)


def process_train_LSTM(symbol, variables_csv, d_from, d_to, model_output, classification_key,
                       epochs, timestamps, n_neurons, learning_rate, reg_rate, dropout_rate,clipping_rate,
                       threshold_stop,grouping_unit=None,grouping_classif_criteria=None,
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

        dataMgm.process_train_LSTM(symbol, variables_csv, d_from, d_to, model_output.replace('"', ""),
                                   classification_key, int(epochs), int(timestamps), int(n_neurons),
                                   float(learning_rate), float(reg_rate), float(dropout_rate),
                                   interval=interval.replace('_', " ") if interval is not None else None,
                                   clipping_rate=float(clipping_rate), threshold_stop=float(threshold_stop),
                                   grouping_unit=int(grouping_unit) if grouping_unit is not None else None,
                                   grouping_classif_criteria=grouping_classif_criteria,
                                   group_as_mov_avg=bool(group_as_mov_avg),
                                   grouping_mov_avg_unit=int(grouping_mov_avg_unit), batch_size=batch_size,
                                   inner_activation=inner_activation, make_stationary=make_stationary)

        # TODO ---> print backtesting output
        logger.print("Model successfully trained for symbol {} and variables {}".format(symbol, variables_csv),
                     MessageType.INFO)

    except Exception as e:
        logger.print("CRITICAL ERROR running process_train_LSTM:{}".format(str(e)), MessageType.ERROR)

def process_test_daily_XGBoost(symbol, series_csv, d_from, d_to, model_to_use, n_algo_param_dict):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(
            f"Initializing XGBoost model testing for symbol {symbol} and model {model_to_use} on {d_from}",
            MessageType.INFO
        )

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(
            config_settings["hist_data_conn_str"],
            config_settings["ml_reports_conn_str"],
            None,
            logger
        )

        dataMgm.process_test_scalping_XGBoost(
            symbol=symbol,
            series_csv=series_csv,
            model_to_use=model_to_use.replace('"', ""),
            d_from=d_from,
            d_to=d_to,
            n_algo_param_dict=n_algo_param_dict
        )

        logger.print(
            f"Displaying predictions for XGBoost model: symbol {symbol} and model {model_to_use} on {d_from}",
            MessageType.INFO
        )

    except Exception as e:
        logger.print(f"CRITICAL ERROR running process_test_daily_XGBoost: {str(e)}", MessageType.ERROR)


def process_test_daily_RF(symbol, series_csv, d_from, d_to, model_to_use,  n_algo_param_dict):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(
            f"Initializing RF model testing for symbol {symbol} and model {model_to_use} on {d_from}",
            MessageType.INFO
        )

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        dataMgm = AlgosOrchestationLogic(
            config_settings["hist_data_conn_str"],
            config_settings["ml_reports_conn_str"],
            None,
            logger
        )

        dataMgm.process_test_scalping_RF(
            symbol=symbol,
            series_csv=series_csv,
            model_to_use=model_to_use.replace('"', ""),
            d_from=d_from,
            d_to=d_to,
            n_algo_param_dict=n_algo_param_dict
        )

        logger.print(
            f"Displaying predictions for RF model: symbol {symbol} and model {model_to_use} on {d_from}",
            MessageType.INFO
        )

    except Exception as e:
        logger.print(f"CRITICAL ERROR running process_test_daily_RF: {str(e)}", MessageType.ERROR)


def process_test_daily_LSTM(symbol, variables_csv, d_from,d_to, timesteps, model_to_use, portf_size, trade_comm,
                            trading_algo,grouping_unit=None,n_params=[],interval=None,use_sliding_window=None,
                            make_stationary=True,classif_threshold=0.5):
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
                                               make_stationary=make_stationary,
                                               classif_threshold=classif_threshold
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

        trd_algos.process_backtest_slope_model(symbol, model_candle, d_from, d_to, portf_size, trading_algo,
                                               algo_params)

        logger.print(f"Slope Model backtest successfully shown for symbol {symbol} from {d_from} to {d_to}",
                     MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print("CRITICAL ERROR running process_backtest_slope_model:{}".format(str(e)), MessageType.ERROR)

def process_download_byma_interest_rates_logic(d_from, d_to=None):
    """
    Entry point for command #68 - Download BYMA Interest Rates.
    Loads configuration, instantiates AlgosOrchestationLogic,
    and delegates to its business logic method.
    """
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print("🏦 Starting download of BYMA Interest Rates", MessageType.INFO)

        # Load configuration (no BYMA-specific keys needed)
        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        # Instantiate orchestrator
        trd_algos = AlgosOrchestationLogic(
            config_settings["hist_data_conn_str"],
            config_settings["ml_reports_conn_str"],
            None,
            logger
        )

        # Parameters (kept for structural consistency)
        algo_params = {}

        # Delegate to logic layer
        trd_algos.process_download_byma_interest_rates(
            d_from=d_from,
            d_to=d_to,
            algo_params=algo_params
        )

        logger.print("✅ BYMA interest rates retrieved successfully", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"CRITICAL ERROR in process_download_byma_interest_rates_logic: {str(e)}", MessageType.ERROR)


def process_download_bcra_interest_rates_logic(d_from, d_to=None):
    """
    Entry point for command #67 - Download BCRA Interest Rates
    Loads configuration, instantiates AlgosOrchestationLogic,
    and delegates to its business logic method.
    """
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print("🏦 Starting download of BCRA Interest Rates", MessageType.INFO)

        # Load configuration from .ini file
        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        # Retrieve API key from config
        api_key = config_settings.get("BCRA_API_KEY", None)
        if not api_key:
            raise Exception("❌ Missing BCRA_API_KEY in config file (commands_mgr.ini)")

        # Instantiate orchestrator (main business logic class)
        trd_algos = AlgosOrchestationLogic(
            config_settings["hist_data_conn_str"],
            config_settings["ml_reports_conn_str"],
            None,
            logger
        )

        # Pass parameters to logic layer
        algo_params = {"bcra_api_key": api_key}

        # Delegate execution to logic layer
        trd_algos.process_download_bcra_interest_rates(
            d_from=d_from,
            d_to=d_to,
            algo_params=algo_params
        )

        logger.print("✅ BCRA interest rates retrieved successfully", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print(
            f"CRITICAL ERROR in process_download_bcra_interest_rates_logic: {str(e)}",
            MessageType.ERROR
        )


def process_download_sec_securities_logic():
    logger = Logger()

    try:
        logger.print("[SEC] Starting SEC securities download", MessageType.INFO)

        # Load configuration
        loader = MLSettingsLoader()
        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        # Instantiate orchestration logic
        trd_algos = AlgosOrchestationLogic(
            config_settings["hist_data_conn_str"],
            config_settings["ml_reports_conn_str"],
            None,
            logger
        )

        # Execute the process
        trd_algos.process_download_sec_securities()

        logger.print("[SEC] ✅ SEC Securities successfully downloaded and persisted", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"[SEC] ❌ Critical error downloading SEC securities: {str(e)}", MessageType.ERROR)


def __build_sec_metadata_orchestation__(logger):
    loader = MLSettingsLoader()
    config_settings = loader.load_settings("./configs/commands_mgr.ini")
    return SECMetadataOrchestationLogic(
        config_settings["ml_reports_conn_str"],
        logger,
        config_settings["SEC_USER_AGENT"]
    )


def process_download_sec_securities_metadata_logic(top=None, retry_errors=False):
    logger = Logger()

    try:
        logger.print("[SEC-META] Starting SEC securities metadata download", MessageType.INFO)

        orchestation = __build_sec_metadata_orchestation__(logger)
        result = orchestation.process_download_all_metadata(
            top=int(top) if top else None,
            include_errors=str(retry_errors).lower() in ("true", "1", "yes")
        )

        logger.print(f"[SEC-META] ✅ ok={result['ok']} fail={result['failed']} "
                     f"of {result['total']}", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"[SEC-META] ❌ Critical error: {str(e)}", MessageType.ERROR)


def process_download_sec_security_metadata_logic(symbol):
    logger = Logger()

    try:
        orchestation = __build_sec_metadata_orchestation__(logger)
        result = orchestation.process_download_single_metadata(symbol=symbol)

        if result.get("ok"):
            logger.print(f"[SEC-META] ✅ {symbol} updated", MessageType.INFO)
        else:
            logger.print(f"[SEC-META] ❌ {symbol} could not be updated", MessageType.ERROR)

    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"[SEC-META] ❌ Critical error on {symbol}: {str(e)}", MessageType.ERROR)


def process_tag_securities_from_csv_logic(tag, csv_path, tag_group="CUSTOM"):
    logger = Logger()

    try:
        orchestation = __build_sec_metadata_orchestation__(logger)
        result = orchestation.process_tag_securities_from_csv(tag, csv_path, tag_group=tag_group)

        logger.print(f"[SEC-TAG] ✅ {result['tag_code']}: {result['tagged']} tagged out of "
                     f"{result['read']} read, {len(result['not_found'])} unmatched",
                     MessageType.INFO)

        if result["not_found"]:
            logger.print(f"[SEC-TAG] unmatched: {', '.join(result['not_found'][:40])}",
                         MessageType.WARNING)

    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"[SEC-TAG] ❌ Critical error: {str(e)}", MessageType.ERROR)

def process_create_average_variable_logic(symbols, output_symbol, d_from, d_to):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(f"📊 Starting average variable creation: {symbols}", MessageType.INFO)

        cfg = loader.load_settings("./configs/commands_mgr.ini")

        trd_algos = AlgosOrchestationLogic(
            cfg["hist_data_conn_str"],
            cfg["ml_reports_conn_str"],
            None,
            logger
        )

        trd_algos.process_create_average_variable(
            symbols=symbols,
            output_symbol=output_symbol,
            d_from=d_from,
            d_to=d_to
        )

        logger.print("✅ Average Variable successfully created", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"CRITICAL ERROR in process_create_average_variable_logic: {str(e)}", MessageType.ERROR)


def process_create_ratio_variable_logic(numerator, denominator, multiplier, d_from, d_to, output_symbol):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(f"🧮 Starting ratio variable creation: {numerator}/{denominator} * {multiplier}", MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        trd_algos = AlgosOrchestationLogic(
            config_settings["hist_data_conn_str"],
            config_settings["ml_reports_conn_str"],
            None,
            logger
        )

        trd_algos.process_create_ratio_variable(
            numerator=numerator,
            denominator=denominator,
            multiplier=multiplier,
            d_from=d_from,
            d_to=d_to,
            output_symbol=output_symbol
        )

        logger.print("✅ Ratio Variable successfully created and persisted", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"CRITICAL ERROR running process_create_ratio_variable_logic: {str(e)}", MessageType.ERROR)


def process_create_spread_variable_bulk_logic(diff_indicators, output_symbols, d_from, d_to=None):
    logger = Logger()

    logger.print(f"[BULK][SPREAD] Starting bulk spread creation for {len(diff_indicators)} item(s)", MessageType.INFO)

    for i, (diff_expr, output_symbol) in enumerate(zip(diff_indicators, output_symbols)):
        logger.print(f"[BULK][SPREAD][{i+1}/{len(diff_indicators)}] Processing: {diff_expr} → {output_symbol}", MessageType.INFO)

        try:
            process_create_spread_variable_logic(
                diff_indicators=diff_expr,
                d_from=d_from,
                d_to=d_to,
                output_symbol=output_symbol
            )

            logger.print(f"[BULK][SPREAD][{i+1}] ✅ Spread created: {output_symbol}", MessageType.INFO)

        except Exception as e:
            print(traceback.format_exc())
            logger.print(f"[BULK][SPREAD][{i+1}] ❌ Failed to create spread {output_symbol}: {str(e)}", MessageType.ERROR)

    logger.print(f"[BULK][SPREAD] ✅ Bulk spread creation complete", MessageType.INFO)


def process_create_spread_variable_logic(diff_indicators, d_from, d_to,output_symbol):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(f"🧪 Starting spread variable creation with {len(diff_indicators.split(','))} input variables", MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        trd_algos = AlgosOrchestationLogic(
            config_settings["hist_data_conn_str"],
            config_settings["ml_reports_conn_str"],
            None,
            logger
        )

        trd_algos.process_create_spread_varaible(diff_indicators=diff_indicators, d_from=d_from, d_to=d_to,output_symbol=output_symbol)

        logger.print("✅ Spread Variable successfully created and persisted", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"CRITICAL ERROR running process_create_spread_variable_logic: {str(e)}", MessageType.ERROR)

def process_download_financial_data_logic_bulk(symbol, d_from, d_to, algo_params):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        vendor = algo_params.get("vendor", "").upper()
        vendor_params = algo_params.get("vendor_params", {})

        logger.print(f"[BULK] Starting download for '{symbol}' (vendor: {vendor})", MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        if vendor == InformationVendors.FRED.value:
            vendor_params["api_key"] = config_settings["FRED_API_KEY"]
        elif vendor == InformationVendors.TRADINGVIEW.value:
            vendor_params["tradingview_user"] = config_settings["TRADING_VIEW_USER"]
            vendor_params["tradingview_pwd"] = config_settings["TRADING_VIEW_PWD"]
        else:
            raise Exception(f"[BULK] Unsupported vendor: {vendor}")

        algo_params["vendor_params"] = vendor_params

        trd_algos = AlgosOrchestationLogic(
            config_settings["hist_data_conn_str"],
            config_settings["ml_reports_conn_str"],
            None,
            logger
        )


        trd_algos.process_download_financial_data_bulk(symbol, d_from, d_to, algo_params)

        logger.print(f"[BULK] ✅ Success: {symbol} from {vendor}", MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"[BULK][ERROR] {symbol} → {str(e)}", MessageType.ERROR)



def process_download_financial_data_logic(symbol, d_from, d_to, algo_params):
    loader = MLSettingsLoader()
    logger = Logger()

    try:
        logger.print(f"Initializing financial data download for symbol '{symbol}' from {d_from} to {d_to}", MessageType.INFO)

        config_settings = loader.load_settings("./configs/commands_mgr.ini")

        # Inject vendor-specific config parameters here if needed
        if "vendor" in algo_params and algo_params["vendor"].upper() == "FRED":
            fred_key = config_settings["FRED_API_KEY"]
            if "vendor_params" not in algo_params:
                algo_params["vendor_params"] = {}
            algo_params["vendor_params"]["api_key"] = fred_key

        if "vendor" in algo_params and algo_params["vendor"].upper() == "TRADINGVIEW":
            tradingview_user = config_settings["TRADING_VIEW_USER"]
            tradingview_pwd = config_settings["TRADING_VIEW_PWD"]
            if "vendor_params" not in algo_params:
                algo_params["vendor_params"] = {}
            algo_params["vendor_params"]["tradingview_user"] = tradingview_user
            algo_params["vendor_params"]["tradingview_pwd"] = tradingview_pwd

        trd_algos = AlgosOrchestationLogic(
            config_settings["hist_data_conn_str"],
            config_settings["ml_reports_conn_str"],
            None,
            logger
        )

        trd_algos.process_download_financial_data(symbol, d_from, d_to, algo_params)

        logger.print(f"Financial data for symbol '{symbol}' successfully downloaded from {d_from} to {d_to}",
                     MessageType.INFO)

    except Exception as e:
        print(traceback.format_exc())
        logger.print(f"CRITICAL ERROR running process_download_financial_data_logic: {str(e)}", MessageType.ERROR)


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

        trd_algos.process_backtest_slope_model_on_custom_etf(etf_path, model_candle, d_from, d_to, portf_size,
                                                             trading_algo, algo_params)

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

def process_download_byma_interest_rates(cmd):
    """
    CLI command handler for 'DownloadBYMAInterestRates'.
    Parses parameters and delegates execution to logic layer entrypoint.
    """
    # Extract parameters from console command
    d_from = ParamReader.get_param(cmd, "from", True, None)
    d_to = ParamReader.get_param(cmd, "to", False, None)

    # Call the logic entrypoint
    process_download_byma_interest_rates_logic(d_from, d_to)


def process_download_bcra_interest_rates(cmd):
    d_from = ParamReader.get_param(cmd, "from", True, None)
    d_to = ParamReader.get_param(cmd, "to", False, None)
    process_download_bcra_interest_rates_logic(d_from, d_to)


def process_commands(cmd):
    cmd_param_list = cmd.split(" ")

    if cmd_param_list[0] == "TrainMLAlgos":
        run_train_ml_algo(cmd)
    elif cmd_param_list[0] == "RunPredictionsLastModel":
        process_run_predictions_last_model(cmd_param_list,cmd_param_list[1],cmd_param_list[2],cmd_param_list[3])
    elif cmd_param_list[0] == "EvalBiasedTradingAlgo":
        run_biased_trading_algo(cmd)
    elif cmd_param_list[0] == "EvalSlidingBiasedTradingAlgo":
        run_sliding_biased_trading_algo(cmd)
    elif cmd_param_list[0] == "EvaluateARIMA":
        #params_validation("EvaluateARIMA", cmd_param_list, 5)
        process_eval_ARIMA_cmd(cmd)
    elif cmd_param_list[0] == "PredictARIMA":
        process_predict_ARIMA_cmd(cmd)
    elif cmd_param_list[0] == "EvalSingleIndicatorAlgo":
        ParamReader.params_validation("EvalSingleIndicatorAlgo", cmd_param_list, 7)
        process_eval_single_indicator_algo(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                           cmd_param_list[5], cmd_param_list[6])
    elif cmd_param_list[0] == "EvalMLBiasedAlgo":
        ParamReader.params_validation("EvalMLBiasedAlgo", cmd_param_list, 8)
        process_eval_ml_biased_algo(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                    cmd_param_list[5], cmd_param_list[6], cmd_param_list[7])
    elif cmd_param_list[0] == "TrainNeuralNetworkAlgo":
        ParamReader.params_validation("TrainNeuralNetworkAlgo", cmd_param_list, 10)
        process_train_neural_network_algo(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                          int(cmd_param_list[5]), float(cmd_param_list[6]), int(cmd_param_list[7]),
                                          cmd_param_list[8], cmd_param_list[9])
    elif cmd_param_list[0] == "BacktestNeuralNetworkAlgo":
        ParamReader.params_validation("BacktestNeuralNetworkAlgo", cmd_param_list, 7)
        process_backtest_neural_network_algo(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3], cmd_param_list[4],
                                             cmd_param_list[5], cmd_param_list[6])

    elif cmd_param_list[0] == "TrainLSTM":
        process_traing_LSTM_cmd(cmd,cmd_param_list)
    elif cmd_param_list[0] == "TrainRF":
        process_train_RF_cmd(cmd,cmd_param_list)
    elif cmd_param_list[0] == "TrainXGBoost":
        process_train_XGBoost_cmd(cmd,cmd_param_list)
    elif cmd_param_list[0] == "TestXGBoost":
        process_test_XGBoost_cmd(cmd)
    elif cmd_param_list[0] == "TrainLSTMWithGrouping":
        process_traing_LSTM_cmd(cmd,cmd_param_list)
    elif cmd_param_list[0] == "DailyCandlesGraph":

        ParamReader.params_validation("DailyCandlesGraph", cmd_param_list, 5)
        process_daily_candles_graph(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3],
                                    cmd_param_list[4])
    elif cmd_param_list[0] == "IndicatorCandlesGraph":

        ParamReader.params_validation("IndicatorCandlesGraph", cmd_param_list, 6)
        process_indicator_candles_graph(cmd_param_list[1], cmd_param_list[2], cmd_param_list[3],
                                    cmd_param_list[4],cmd_param_list[5])
    elif cmd_param_list[0] == "TestDailyLSTM":
        process_test_LSTM_cmd(cmd)
    elif cmd_param_list[0] == "BacktestSlopeModel":
        process_backtest_slope_model(cmd)
    elif cmd_param_list[0] == "TestDailyRF":
        process_test_RF_cmd(cmd)
    elif cmd_param_list[0] == "BacktestSlopeModelOnCustomETF":
        process_backtest_slope_model_on_custom_etf(cmd)
    elif cmd_param_list[0] == "TestDailyLSTMWithGrouping":
        process_test_LSTM_cmd(cmd)

    elif cmd_param_list[0] == "CreateSintheticIndicator":
        process_create_sinthetic_indicator(cmd)
    elif cmd_param_list[0] == "DownloadFinancialData":
        process_download_financial_data(cmd)
    elif cmd_param_list[0] == "DownloadFinancialDataBulk":
        process_download_financial_data_bulk(cmd)
    #
    elif cmd_param_list[0] == "DisplayOrderRoutingScreen":
        process_display_order_routing_screen(cmd)
    elif cmd_param_list[0] == "BiasMainLandingPage":
        process_bias_main_landing_page(cmd)

    elif cmd_param_list[0] == "EvalSlidingRandomForest":
        run_sliding_random_forest(cmd)
    elif cmd_param_list[0] == "CustomRegimeSwitchDetector":
        run_custom_regime_switch_detector(cmd)
    elif cmd_param_list[0] == "CreateLightweightIndicator":
        process_create_lightweight_indicator(cmd)
    elif cmd_param_list[0] == "CreateSpreadVariable":
        process_create_spread_variable(cmd)
    elif cmd_param_list[0] == "CreateSpreadVariableBulk":
        process_create_spread_variable_bulk(cmd)
    elif cmd_param_list[0] == "DownloadSECSecurities":
        process_download_sec_securities(cmd)
    elif cmd_param_list[0] == "DownloadSECSecuritiesMetadata":
        process_download_sec_securities_metadata(cmd)
    elif cmd_param_list[0] == "DownloadSECSecurityMetadata":
        process_download_sec_security_metadata(cmd)
    elif cmd_param_list[0] == "TagSecuritiesFromCSV":
        process_tag_securities_from_csv(cmd)
    elif cmd_param_list[0] == "DownloadBCRAInterestRates":
        process_download_bcra_interest_rates(cmd)
    elif cmd_param_list[0] == "CreateRatioVariable":
        process_create_ratio_variable(cmd)
    elif cmd_param_list[0] == "CreateAverageVariable":
        process_create_average_variable(cmd)
    elif cmd_param_list[0] == "DownloadBYMAInterestRates":
        process_download_byma_interest_rates(cmd)

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
