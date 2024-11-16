import math

import pyodbc

from business_entities.economic_value import EconomicValue
from common.util.date_handler import DateHandler

_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_SYMBOL_IDX=0
_DATE_IDX=1
_INTERVAL_IDX=2
_OPEN_IDX=3
_HIGH_IDX=4
_LOW_IDX=5
_CLOSE_IDX=6
_TRADE_IDX=7
_CASH_VOLUME_IDX=8
_NOMINAL_VOLUME_IDX=9

class EconomicSeriesManager:

    def __init__(self,connection_string):
        self.connection = pyodbc.connect(connection_string)

    def build_economic_value(self, row):
        econ_val = EconomicValue(p_symbol=str(row[_SYMBOL_IDX]),p_interval=str(row[_INTERVAL_IDX]),
                                 p_date= DateHandler.convert_str_date(str(row[_DATE_IDX]),_DATE_FORMAT))

        econ_val.open=float(row[_OPEN_IDX]) if row[_OPEN_IDX] is not None else None
        econ_val.high = float(row[_HIGH_IDX]) if row[_HIGH_IDX] is not None else None
        econ_val.low = float(row[_LOW_IDX]) if row[_LOW_IDX] is not None else None
        econ_val.close = float(row[_CLOSE_IDX]) if row[_CLOSE_IDX] is not None else None
        econ_val.trade = float(row[_TRADE_IDX]) if row[_TRADE_IDX] is not None else None
        econ_val.cash_volume = float(row[_CASH_VOLUME_IDX]) if row[_CASH_VOLUME_IDX] is not None else None
        econ_val.nominal_volume = float(row[_NOMINAL_VOLUME_IDX]) if row[_NOMINAL_VOLUME_IDX] is not None else None
        return econ_val

    def build_empty_economic_value(self, symbol,interval, dfrom, dto):
        econ_val = EconomicValue(p_symbol=symbol,p_interval=interval,p_date= dfrom)

        econ_val.open= math.nan
        econ_val.high = math.nan
        econ_val.low = math.nan
        econ_val.close = math.nan
        econ_val.trade = math.nan
        econ_val.cash_volume =math.nan
        econ_val.nominal_volume = math.nan
        return econ_val


    def get_economic_values(self,symbol,interval, dfrom, dto):
        economic_values=[]
        with self.connection.cursor() as cursor:
            params = (symbol,interval,dfrom,dto)
            cursor.execute("{CALL GetCandles (?,?,?,?)}", params)

            found=False
            for row in cursor:
                found=True
                economic_value=None
                economic_value=self.build_economic_value(row)
                economic_values.append(economic_value)
            if not found:
                economic_value = self.build_empty_economic_value(symbol,interval, dfrom, dto)
                economic_values.append(economic_value)

        return economic_values