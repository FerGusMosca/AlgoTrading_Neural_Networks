import numpy as np
from scipy.stats import linregress

from common.enums.columns_prefix import ColumnsPrefix
from common.enums.market_regimes import MarketRegimes
from common.enums.on_off_indicator_values import OnOffIndicatorValue
from common.enums.regime_zones import RegimeZones
from logic_layer.trading_algos.base_class_daily_trading_backtester import BaseClassDailyTradingBacktester



class SlopeCalculator(BaseClassDailyTradingBacktester):
    _DATE_COL="date"
    def __init__(self):
        pass


    @staticmethod
    def set_def_value(series_df,col_prefix,def_value):
        series_df[col_prefix] = def_value
        return series_df


    @staticmethod
    def calculate_slope(values):
        # Calculate linear regression slope
        x = np.arange(len(values))
        slope, _, _, _, _ = linregress(x, values)

        return slope


    @staticmethod
    def calculate_indicator_slope(series_df, slope_units, indicator):
        """
        Calculates the slope of the specified indicator based on the last `slope_units` non-NaN values.

        Args:
            series_df (pd.DataFrame): DataFrame containing at least a 'Date' column and the specified indicator column.
            slope_units (int): Number of recent readings to consider for slope calculation.
            indicator (str): Name of the column representing the indicator.

        Returns:
            pd.DataFrame: DataFrame with a new column '<indicator>_Slope' containing the calculated slopes.
        """
        # Ensure the DataFrame is sorted by date
        series_df = series_df.sort_values(by=SlopeCalculator._DATE_COL).reset_index(drop=True)

        # Initialize a new column for the slope
        slope_column_name = f"{indicator}{ColumnsPrefix.SLOPE_POSFIX.value}"
        series_df[slope_column_name] = np.nan

        # Iterate through the DataFrame rows
        for i in range(len(series_df)):
            # Select the last `slope_units` non-NaN values of the indicator up to the current row
            recent_data = series_df.loc[:i, indicator].dropna().iloc[-slope_units:]

            if len(recent_data) == slope_units:
                # Prepare x (relative time) and y (indicator values) for linear regression
                x = np.arange(len(recent_data))
                y = recent_data.values

                # Calculate the slope using linear regression
                slope, _, _, _, _ = linregress(x, y)

                # Assign the calculated slope to the current row
                series_df.loc[i, slope_column_name] = slope

        # Interpolate missing values in the slope column
        series_df[slope_column_name] = series_df[slope_column_name].interpolate(method='linear')

        return series_df


    @staticmethod
    def calculate_indicator_on_off_slope(series_df, indicator):
        """
        Calculates the slope of the specified indicator based on the last `slope_units` non-NaN values.

        Args:
            series_df (pd.DataFrame): DataFrame containing at least a 'Date' column and the specified indicator column.
            indicator (str): Name of the column representing the indicator.

        Returns:
            pd.DataFrame: DataFrame with a new column '<indicator>_Slope' containing the calculated slopes.
        """
        # Ensure the DataFrame is sorted by date
        series_df = series_df.sort_values(by=SlopeCalculator._DATE_COL).reset_index(drop=True)

        # Initialize a new column for the slope
        slope_column_name = f"{indicator}{ColumnsPrefix.SLOPE_POSFIX.value}"
        series_df[slope_column_name] = np.nan

        # Iterate through the DataFrame rows
        last_recent_data=None
        for i in range(len(series_df)):

            recent_data = series_df.loc[i, indicator]

            if not np.isnan(recent_data):
                last_recent_data=recent_data
            if last_recent_data is not None:
                series_df.loc[i, slope_column_name] =  OnOffIndicatorValue.ON.value if last_recent_data>0 else OnOffIndicatorValue.OFF.value
            else:
                series_df.loc[i, slope_column_name] = OnOffIndicatorValue.OFF.value

        # Interpolate missing values in the slope column
        series_df[slope_column_name] = series_df[slope_column_name].interpolate(method='linear')

        return series_df

    @staticmethod
    def classify_slope_regime(values, dates, regime_filter, window, slope_threshold=0.3, abs_value_threshold=None):
        signals = []

        for i in range(len(values)):
            if i < window:
                signals.append(0)
                continue

            window_vals = values[i - window:i]

            if any(v is None or np.isnan(v) for v in window_vals):
                signals.append(0)
                continue

            x = np.arange(window)
            y = np.array(window_vals)
            if np.any(np.isnan(y)) or np.all(y == y[0]):
                slope = 0.0
            else:
                slope = np.polyfit(x, y, 1)[0]

            avg_val = np.mean(y)
            dynamic_threshold = slope_threshold * abs(avg_val)

            if regime_filter == MarketRegimes.HIGH_VOLATILITY.value:
                if slope > dynamic_threshold:
                    signal = 1
                elif slope < -dynamic_threshold:
                    signal = -1
                else:
                    signal = 0
            elif regime_filter == MarketRegimes.ABS_VALUE.value:
                current_val = values[i]
                signal = 1 if current_val is not None and not np.isnan(
                    current_val) and current_val > abs_value_threshold else 0
            else:
                signal = 0

            signals.append(signal)

        return signals

