from enum import Enum


class IndicatorType(Enum):

    DIRECT_SLOPE = "direct_slope"
    INV_SLOPE = "inv_slope"
    ARIMA = "arima"
    INV_ARIMA = "inv_arima"