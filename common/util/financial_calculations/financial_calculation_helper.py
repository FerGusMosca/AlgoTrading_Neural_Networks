import numpy as np
import pandas as pd

from common.util.logging.light_logger import LightLogger


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

    @staticmethod
    def get_max_cum_drawdown_in_arr(returns_arr):
        """
        Calculates the maximum cumulative drawdown from a list of periodic returns.

        :param returns_arr: List of returns (e.g., [r1, r2, ..., rn] where r = (P_t/P_{t-1}) - 1)
        :return: Maximum cumulative drawdown as float (e.g., -0.25 for -25%)
        """
        if not returns_arr:
            return 0.0

        # Reconstruct portfolio values starting at 1
        portf_vals = [1.0]
        for r in returns_arr:
            portf_vals.append(portf_vals[-1] * (1 + r))

        # Compute cumulative max drawdown on that reconstructed portfolio series
        max_val = portf_vals[0]
        max_drawdown = 0.0
        for val in portf_vals:
            if val > max_val:
                max_val = val
            drawdown = (val - max_val) / max_val
            max_drawdown = min(max_drawdown, drawdown)

        return max_drawdown

    @staticmethod
    def get_max_cum_yearly_drawdown(year, mtm_dict, dd_dict):
        monthly_returns = mtm_dict.get(year, [])
        monthly_drawdowns = dd_dict.get(year, [])

        if not monthly_returns or not monthly_drawdowns:
            return 0

        assert len(monthly_returns) == len(monthly_drawdowns), "Mismatch in lengths"

        n = len(monthly_returns)
        min_combined = float('inf')

        for j in range(n):
            # Sumar desde i hasta j-1, luego agregar dd[j]
            for i in range(j + 1):
                cum_return = sum(monthly_returns[i:j])
                combined = cum_return + monthly_drawdowns[j]
                min_combined = min(min_combined, combined)

        return min_combined




