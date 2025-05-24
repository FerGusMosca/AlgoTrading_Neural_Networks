import numpy as np
import pandas as pd

from common.util.light_logger import LightLogger


class FinancialCalculationsHelper():


    @staticmethod
    def calculate_max_drawdown_on_different_positions(trading_summary_df,column_name):
        portfolio_values = trading_summary_df[column_name].values
        max_values = np.maximum.accumulate(portfolio_values)  # Máximos acumulados
        drawdowns = (portfolio_values - max_values) / max_values  # Caídas relativas
        max_cum_drawdown = np.min(drawdowns)  # La caída máxima
        return max_cum_drawdown

    @staticmethod
    def calculate_max_total_drawdown( daily_profits):
        """
        This function calculates the maximum drawdown over a period based on daily profits.

        Parameters:
        daily_profits (list or np.array): Array containing daily profits (positive or negative).

        Returns:
        float: The maximum drawdown over the period.
        """
        # Convert daily profits to a numpy array if it's not already
        daily_profits = np.array(daily_profits)

        # Initialize variables to track the maximum drawdown
        max_drawdown = 0
        cumulative_drawdown = 0

        # Loop through daily profits to calculate the maximum drawdown
        for profit in daily_profits:
            cumulative_drawdown += profit
            if cumulative_drawdown > 0:
                cumulative_drawdown = 0
            if cumulative_drawdown < max_drawdown:
                max_drawdown = cumulative_drawdown

        return -1 * abs(max_drawdown)


    @staticmethod
    def calculate_max_drawdown_with_prices(trading_summary_df, prices_df, trading_symbol):
        # Column names for the asset prices
        close_col = f"close_{trading_symbol}"

        max_drawdowns = []

        # Iterate over each row in trading_summary_df
        for _, row in trading_summary_df.iterrows():
            # Extract position dates and side
            start_date = pd.to_datetime(row['open'])
            end_date = pd.to_datetime(row['close'])
            side = row['side']

            # Filter series_df for the corresponding date range
            position_prices = prices_df[(prices_df['date'] >= start_date) & (prices_df['date'] <= end_date)][close_col]

            if position_prices.empty:
                continue  # Skip if no data for the date range

            # Convert prices to float if necessary
            position_prices = position_prices.astype(float)

            # Calculate the drawdown based on the side of the position
            if side == "LONG":
                # Long position: Drawdown = (peak - trough) / peak
                peak = position_prices.cummax()
                drawdown = (peak - position_prices) / peak
            elif side == "SHORT":
                # Short position: Drawdown = (trough - peak) / trough
                trough = position_prices.cummin()
                drawdown = (position_prices - trough) / trough
            else:
                raise ValueError(f"Unknown side: {side}")

            # Find the maximum drawdown for this position
            max_drawdown = drawdown.max()
            max_drawdowns.append(max_drawdown)

        # Return the highest drawdown among all positions
        return max(max_drawdowns) if max_drawdowns else 0

    @staticmethod
    def max_drawdown_on_MTM(MT_values):
        """
        Calculate the maximum drawdown from a list of portfolio MTM values.

        :param MT_values: List or array of portfolio values over time.
        :return: Maximum drawdown as a float (e.g., -0.12 means -12%)
        """
        MT_values = np.array(MT_values)

        if len(MT_values) < 2:
            LightLogger.do_log("[WARNING] Empty or insufficient MTM values — skipping drawdown calc.")
            return 0.0  # No drawdown possible with 0 or 1 value

        max_values = np.maximum.accumulate(MT_values)  # Running max
        drawdowns = (MT_values - max_values) / max_values  # Relative drop from peak
        max_drawdown = np.min(drawdowns)  # Most negative value is the max drawdown

        return max_drawdown