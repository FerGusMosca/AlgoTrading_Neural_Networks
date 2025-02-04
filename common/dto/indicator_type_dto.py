import csv


class IndicatorTypeDTO():

    def __init__(self,indicator,type):
        self.indicator=indicator
        self.type=type


    @staticmethod
    def load_indicator_type_data(indicators_csv, indicator_types_csv):
        indicator_objects = []

        indicator_values = indicators_csv.strip().split(',')
        type_values = indicator_types_csv.strip().split(',')

        # Make sure they have the same size
        if len(indicator_values) != len(type_values):
            raise ValueError("Lines do not have the same size.")

        # Create the objects
        for indicator, type_ in zip(indicator_values, type_values):
            obj = IndicatorTypeDTO(indicator.strip(), type_.strip())
            indicator_objects.append(obj)

        return indicator_objects