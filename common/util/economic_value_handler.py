from business_entities.economic_value import EconomicValue
from common.enums.intervals import Intervals
from common.util.csv_reader import CSVReader
from datetime import datetime, timedelta
import calendar

class EconomicValueHandler:

    @staticmethod
    def __parse_date__(date_str):
        """
        Converts a date string like '17-Dec' into a datetime object like 2017-12-01.

        Format:
            - 'DD-MMM' → YYYY-MM-01 (e.g., '17-Dec' → 2017-12-01)
        """
        from datetime import datetime

        try:
            # Split the string into number and month (e.g., "17-Dec" → "17" and "Dec")
            day_str, month_str = date_str.split('-')

            # Convert the number part to an integer (e.g., "17" → 17)
            day = int(day_str)

            # Convert the abbreviated month to a number (e.g., "Dec" → 12)
            month = datetime.strptime(month_str, "%b").month

            # Assume the year is 2000 + the number (e.g., 17 → 2017)
            year = 2000 + day

            # Return a datetime object with day fixed to 1
            return datetime(year, month, 1)
        except (ValueError, IndexError):
            # Handle invalid formats or parsing errors
            print(f"⚠ Warning: Date format for '{date_str}' not recognized. Expected format: 'DD-MMM' (e.g., '17-Dec')")
            return None

    # Test cases
    test_dates = ['17-Dec', '23-Mar', '24-Jan', '15-Feb']
    for date in test_dates:
        result = __parse_date__(date)
        print(f"{date} → {result}")

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
            print("⚠ Warning: Parsed date is None, cannot adjust.")
            return None  # Prevents errors when date is missing

        # Ensure the parsed date is in a valid range (avoid 1900s default)
        if parsed_date.year < 1950:
            print(f"⚠ Warning: Year {parsed_date.year} is too old, setting default year to current.")
            parsed_date = parsed_date.replace(year=datetime.now().year)

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
            parsed_date = EconomicValueHandler.__parse_date__(date_str)
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
