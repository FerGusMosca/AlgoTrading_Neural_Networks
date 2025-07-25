import pyodbc

from business_entities.date_range_classification_value import DateRangeClassificationValue
from common.util.financial_calculations.date_handler import DateHandler

_DATE_FORMAT = "%Y-%m-%d"
_ID_IDX = 0
_KEY_IDX = 1
_DATE_START_IDX = 2
_DATE_END_IDX = 3
_CLASSIFICATION_IDX = 4


class DateRangeClassificationManager:

    def __init__(self, connection_string):
        self.connection = pyodbc.connect(connection_string)

    def build_date_range_classification_value(self, row):
        date_range_classif = DateRangeClassificationValue(p_id=str(row[_ID_IDX]), p_key=str(row[_KEY_IDX]),
                                                          p_date_start=DateHandler.convert_str_date(
                                                              str(row[_DATE_START_IDX]), _DATE_FORMAT),
                                                          p_date_end=DateHandler.convert_str_date(
                                                              str(row[_DATE_END_IDX]), _DATE_FORMAT),
                                                          p_classification=str(row[_CLASSIFICATION_IDX]))

        return date_range_classif

    def get_date_range_classification_values(self, key, d_from=None, d_to=None):
        date_range_values = []
        with self.connection.cursor() as cursor:
            params = (key,d_from,d_to)
            cursor.execute("{CALL GetDateRangeClassifications (?,?,?)}", params)

            for row in cursor:
                date_range_value = None
                date_range_value = self.build_date_range_classification_value(row)
                date_range_values.append(date_range_value)

        return date_range_values



