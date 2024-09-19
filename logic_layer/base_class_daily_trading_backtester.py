import numpy as np
class BaseClassDailyTradingBacktester:


    _LONG_POS="LONG"
    _SHORT_POS="SHORT"
    _FLAT_POS="FLAT"
    def __init__(self):
        pass



    def calculate_max_total_drawdown(self, daily_profits):
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

        return -1* abs(max_drawdown)

    def __calculate_day_trading_summary__(self, trading_summary_df):
        """
        This method calculates the daily trading summary including:
        - Total net profit for the day.
        - Total number of positions.
        - Maximum drawdown, defined as the maximum cumulative loss during the day.

        Parameters:
        trading_summary_df (pd.DataFrame): A DataFrame with trading positions containing the following columns:
                                           'close', 'price_close', 'unit_gross_profit', 'total_gross_profit',
                                           'total_net_profit'

        Returns:
        daily_net_profit (float): The sum of all 'total_net_profit' values for the day.
        total_positions (int): The number of positions closed in the day.
        max_cum_drawdown (float): The maximum drawdown defined as the maximum cumulative loss during the day.
        """
        # 1. Sum all values in the 'total_net_profit' column
        daily_net_profit = trading_summary_df['total_net_profit'].sum()

        # 2. Count the total number of positions (number of rows in the DataFrame)
        total_positions = len(trading_summary_df)

        # 3. Calculate the maximum drawdown as the maximum cumulative loss during the day
        max_cum_drawdown = 0
        current_drawdown = 0

        for profit in trading_summary_df['total_net_profit']:
            if profit < 0:
                current_drawdown += profit
                max_cum_drawdown = min(max_cum_drawdown, current_drawdown)
            else:
                current_drawdown = 0  # Reset the drawdown when there's a profit

        return daily_net_profit, total_positions, max_cum_drawdown, trading_summary_df