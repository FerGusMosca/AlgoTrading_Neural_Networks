from enum import Enum


class IndicatorType(Enum):

    DIRECT_SLOPE = "direct_slope"
    INV_SLOPE = "inv_slope"
    ARIMA = "arima"
    POS_THRESHOLD = "pos_threshold"
    SARIMA="sarima"
    INV_ARIMA = "inv_arima"
    INV_SARIMA="inv_sarima"