from datetime import datetime, timedelta

import numpy as np


class PortfolioSummaryAnalyzer:


    def __init__(self):
        pass

    @staticmethod
    def calculate_portfolio_metrics(result, init_portf, start, end):
        """
        Calculate the final portfolio value, cumulative maximum drawdown, and format results into an array for each strategy.

        Args:
            result (list): List of dictionaries containing portfolio performance metrics for multiple strategies.
            init_portf (float): Initial portfolio value.
            start (datetime): Start date as a datetime object.
            end (datetime): End date as a datetime object.

        Returns:
            dict: Dictionary with strategy names as keys and their respective metrics as values.
                  Each strategy's metrics include final portfolio value, cumulative max drawdown, and formatted results array.
        """
        # Extract strategies dynamically from the first dictionary in the result list
        if not result:
            return {}

        # Get the first dictionary and extract strategy keys
        first_dict = result[0]
        strategies =first_dict.keys()

        if not strategies:
            return {}

        # Initialize results dictionary to store metrics for each strategy
        strategy_results = {strategy: {
            'final_portfolio_value': 0.0,
            'cumulative_max_drawdown': "0.0%",
            'formatted_results': []
        } for strategy in strategies}

        # Process each strategy
        for strategy in strategies:
            # Initialize variables for this strategy
            portfolio_value = init_portf
            portfolio_values = [init_portf]  # Track portfolio value over time for drawdown calculation
            max_drawdown = 0.0  # Cumulative maximum drawdown
            formatted_results = []

            # Define bimonthly periods
            current_date = start
            period_index = 0

            # Process each dictionary in the result array
            while current_date <= end and period_index < len(result):
                # Get the current dictionary
                perf_dict = result[period_index]

                # Extract the strategy-specific PortfSummary object
                strategy_summary = perf_dict.get(strategy)
                if not strategy_summary:
                    # Skip if the strategy is not present in this period
                    period_index += 1
                    current_date = current_date + timedelta(days=60)
                    continue

                # Extract the total net profit (convert from percentage string to decimal)
                total_net_profit_str = strategy_summary.total_net_profit_str
                total_net_profit=strategy_summary.total_net_profit

                # Extract the max drawdown for this period (convert from percentage string to decimal)
                period_drawdown=strategy_summary.max_drawdown_on_MTM

                # Apply the return to the portfolio value
                portfolio_value = portfolio_value * (1 + (total_net_profit/100))
                portfolio_values.append(portfolio_value)

                # Calculate cumulative profit percentage
                cum_profit = ((portfolio_value / init_portf) - 1) * 100.0

                # Calculate running maximum drawdown
                values_array = np.array(portfolio_values)
                peaks = np.maximum.accumulate(values_array)
                drawdowns = (values_array - peaks) / peaks
                current_max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
                max_drawdown = min(max_drawdown, current_max_drawdown)

                # Determine the year and bimonthly period
                year = current_date.year
                month = current_date.month
                if 1 <= month <= 2:
                    period = "Jan-Feb"
                elif 3 <= month <= 4:
                    period = "Mar-Apr"
                elif 5 <= month <= 6:
                    period = "May-Jun"
                elif 7 <= month <= 8:
                    period = "Jul-Aug"
                elif 9 <= month <= 10:
                    period = "Sep-Oct"
                else:
                    period = "Nov-Dec"

                # Add the result to the formatted array for this strategy
                formatted_results.append({
                    'Year': year,
                    'Period': period,
                    'Profit %': f"{round(total_net_profit,2)} %",
                    'Drawdown': round(period_drawdown * 100, 2),
                    'Cum Profit': f"{round(cum_profit,2)} %",
                    'Cum Drawdown': round(max_drawdown * 100, 2)
                })

                # Move to the next bimonthly period (add 2 months)
                current_date = current_date + timedelta(days=60)
                period_index += 1

            # Store the results for this strategy
            strategy_results[strategy] = {
                'final_portfolio_value': f"{round(portfolio_value,2)} $",
                'cumulative_max_drawdown': f"{round(max_drawdown * 100, 2)} %",
                'formatted_results': formatted_results
            }

        return strategy_results