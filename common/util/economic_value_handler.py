from business_entities.economic_value import EconomicValue
from common.enums.intervals import Intervals
from common.util.csv_reader import CSVReader


class EconomicValueHandler:

    @staticmethod
    def __parse_date(date_str):
        """
        Attempts to parse the date string and returns a date set to the last day of the month.
        First, it tries the format '%b-%y' (e.g., Mar-03). If unsuccessful, it tries additional formats.
        """
        import calendar
        from datetime import datetime

        # Attempt using the expected format: 'Mar-03'
        try:
            dt = datetime.strptime(date_str, "%b-%y")
            # Set the day to the last day of the month
            last_day = calendar.monthrange(dt.year, dt.month)[1]
            return dt.replace(day=last_day)
        except ValueError:
            pass

        # Try other date formats
        date_formats = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"]
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                last_day = calendar.monthrange(dt.year, dt.month)[1]
                return dt.replace(day=last_day)
            except ValueError:
                continue

        print(f"Warning: Date format for '{date_str}' not recognized.")
        return None

    @staticmethod
    def load_economic_series(file_path, symbol):
        """
        Loads an economic series from a CSV file and returns a list of EconomicValue objects.
        The CSV file should have two columns: Date and Value.
        The date is parsed and adjusted to the last day of the month.
        All price fields (open, high, low, close, trade) are set to the numeric value from the CSV.
        Volume fields are set to 0 since they do not apply.
        The interval is set to Intervals.DAY.
        """
        dates_csv = CSVReader.extract_col_arr(file_path, 0)
        values_csv = CSVReader.extract_col_arr(file_path, 1)
        economic_values = []

        # Remove header if present
        if dates_csv and dates_csv[0].lower() == "date":
            dates_csv = dates_csv[1:]
            values_csv = values_csv[1:]

        for date_str, value_str in zip(dates_csv, values_csv):
            parsed_date = EconomicValueHandler.__parse_date(date_str)

            try:
                numeric_value = float(value_str)
            except ValueError:
                numeric_value = None  # Alternatively, skip this row

            econ_val = EconomicValue(
                p_symbol=symbol,
                p_interval=Intervals.DAY,  # Use the 'DAY' enum
                p_date=parsed_date,
                p_open=numeric_value,
                p_high=numeric_value,
                p_low=numeric_value,
                p_close=numeric_value,
                p_trade=numeric_value,
                p_cash_volume=0,       # Volume does not apply
                p_nominal_volume=0     # Volume does not apply
            )
            economic_values.append(econ_val)

        return economic_values
