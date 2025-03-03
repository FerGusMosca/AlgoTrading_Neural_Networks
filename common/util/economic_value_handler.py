from business_entities.economic_value import EconomicValue
from common.enums.intervals import Intervals
from common.util.csv_reader import CSVReader
from datetime import datetime, timedelta
import calendar

class EconomicValueHandler:

    @staticmethod
    def __parse_date(date_str):
        """
        Attempts to parse the date string and returns a date set to the 1st day of the month.
        Handles formats like:
            - 'Mar-23'  →  2023-03-01
            - '23-Mar'  →  2023-03-01
            - '03/2023' →  2023-03-01
            - Other common formats ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y")
        """
        import calendar
        from datetime import datetime

        # Define possible date formats
        date_formats = [
            ("%b-%y", True),  # "Mar-23" → Assume day = 1st
            ("%d-%b", False),  # "23-Mar" → Assume day = 1st, infer year as current
            ("%m/%Y", True),  # "03/2023" → Assume day = 1st
            ("%d/%m/%Y", None),  # Standard formats
            ("%Y-%m-%d", None),
            ("%d-%m-%Y", None)
        ]

        for fmt, force_first_day in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                year = dt.year
                month = dt.month

                # If format only provides month & year, or if it's a "23-Mar" case, assume the 1st day
                if force_first_day is not None:
                    return datetime(year, month, 1)

                return dt  # Return as is for full date formats
            except ValueError:
                continue

        print(f"Warning: Date format for '{date_str}' not recognized.")
        return None

    @staticmethod
    def __adjust_date__(parsed_date: datetime, add_days: int) -> datetime:
        """
        Adjusts the given date based on the `add_days` parameter.

        - If `add_days == 30`, it sets the date to the last day of the month.
        - Otherwise, it simply adds the given number of days.

        Args:
            parsed_date (datetime): The initial parsed date.
            add_days (int): The number of days to add or adjust.

        Returns:
            datetime: The adjusted date.
        """
        if not parsed_date:
            return None  # Handle invalid dates gracefully

        if add_days == 30:
            # Set to the last day of the month
            last_day = calendar.monthrange(parsed_date.year, parsed_date.month)[1]
            return parsed_date.replace(day=last_day)

        # Otherwise, add the given number of days
        return parsed_date + timedelta(days=add_days)

    @staticmethod
    def load_economic_series(file_path, symbol,delimeter=';',add_days=0):
        """
        Loads an economic series from a CSV file and returns a list of EconomicValue objects.
        The CSV file should have two columns: Date and Value.
        The date is parsed and adjusted to the last day of the month.
        All price fields (open, high, low, close, trade) are set to the numeric value from the CSV.
        Volume fields are set to 0 since they do not apply.
        The interval is set to Intervals.DAY.
        """
        dates_csv = CSVReader.extract_col_arr(file_path, 0,delimeter)
        values_csv = CSVReader.extract_col_arr(file_path, 1,delimeter)
        economic_values = []

        # Remove header if present
        if dates_csv and dates_csv[0].lower() == "date":
            dates_csv = dates_csv[1:]
            values_csv = values_csv[1:]

        for date_str, value_str in zip(dates_csv, values_csv):
            parsed_date = EconomicValueHandler.__parse_date(date_str)
            parsed_date=EconomicValueHandler.__adjust_date__(parsed_date,add_days=add_days)

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
