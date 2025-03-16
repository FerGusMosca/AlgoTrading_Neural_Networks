import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from common.util.date_handler import DateHandler


class PortfolioSummaryAnalyzer:

    _OUTPUT_DIR="./output/csv"

    def __init__(self):
        pass

    @staticmethod
    def calculate_portfolio_metrics(result, init_portf, start, end, timestamp):
        """
        Calculate the final portfolio value, cumulative maximum drawdown, and format results into an array for each strategy.
        Export the results to separate CSV files for each strategy with a timestamp, updating incrementally.

        Args:
            result (list): List of dictionaries containing portfolio performance metrics for multiple strategies.
            init_portf (float): Initial portfolio value.
            start (datetime): Start date as a datetime object.
            end (datetime): End date as a datetime object.
            timestamp (str): Timestamp to include in the filenames.

        Returns:
            dict: Dictionary with strategy names as keys and their respective metrics as values.
                  Each strategy's metrics include final portfolio value, cumulative max drawdown, and formatted results array.
        """
        # Extract strategies dynamically from the first dictionary in the result list
        if not result or not result[0]:
            return {}

        # Get the first dictionary and extract strategy keys
        first_dict = result[0]
        strategies = first_dict.keys()

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

            # Process each dictionary in the result array
            for period_index, perf_dict in enumerate(result):
                # Extract the strategy-specific PortfSummary object
                strategy_summary = perf_dict.get(strategy)

                # Use the start of the period (date_open) to determine the bimonthly period
                year = strategy_summary.year
                period=strategy_summary.period

                # Use total_net_profit and max_drawdown_on_MTM directly (assuming they are floats or strings)
                total_net_profit = strategy_summary.total_net_profit
                max_drawdown_on_mtm = strategy_summary.max_drawdown_on_MTM

                # Convert to decimal for calculation
                total_net_profit_decimal = total_net_profit
                period_drawdown = max_drawdown_on_mtm

                # Apply the return to the portfolio value
                portfolio_value = portfolio_value * (1 + total_net_profit_decimal)
                portfolio_values.append(portfolio_value)

                # Calculate cumulative profit percentage
                cum_profit = ((portfolio_value / init_portf) - 1) * 100.0

                # Update maximum drawdown (track the worst drawdown observed)
                max_drawdown = min(max_drawdown, period_drawdown)

                # Add the result to the formatted array for this strategy
                formatted_results.append({
                    'Year': year,
                    'Period': period,
                    'Profit %': round(total_net_profit*100,2),
                    'Drawdown %': round(max_drawdown_on_mtm*100,2),
                    'Cum Profit $': round(cum_profit, 2),
                    'Max. Cum Drawdown %': round(max_drawdown * 100, 2),  # Maximum drawdown up to this period
                    'Portfolio $': round(portfolio_value, 2)
                })

                # Update the CSV file incrementally
                PortfolioSummaryAnalyzer.strategy_results_to_csv(
                    formatted_results,
                    portfolio_value,
                    max_drawdown,
                    strategy,
                    timestamp,
                    incremental=False
                )

            # Store the results for this strategy
            strategy_results[strategy] = {
                'final_portfolio_value': portfolio_value,
                'cumulative_max_drawdown': f"{round(max_drawdown * 100, 2)}%",
                'formatted_results': formatted_results
            }

        return strategy_results

    @staticmethod
    def strategy_results_to_csv(strategy_results, portfolio_value, max_drawdown, strategy, timestamp,
                                incremental=False):
        """
        Export strategy results to a CSV file with the specified structure.

        Args:
            strategy_results (list): List of dictionaries containing period-wise results.
            portfolio_value (float): Final portfolio value.
            max_drawdown (float): Cumulative maximum drawdown (as a decimal, e.g., -0.052 for -5.2%).
            strategy (str): Name of the strategy.
            timestamp (str): Timestamp to include in the filename.
            incremental (bool): If True, append to the existing file; if False, overwrite.
        """
        # Format portfolio_value with commas and max_drawdown as a percentage
        formatted_portfolio_value = f"${portfolio_value:,.2f}"  # e.g., $1,029,997.59
        formatted_max_drawdown = f"{round(max_drawdown * 100, 2)}%"  # e.g., -5.20%

        # Create a DataFrame for the header rows
        header_data = [
            {'Year': 'Final Portfolio Value', 'Period': formatted_portfolio_value},
            {'Year': 'Cumulative Max Drawdown', 'Period': formatted_max_drawdown}
        ]
        header_df = pd.DataFrame(header_data)

        # Add empty columns to header_df to match the main DataFrame's columns
        for col in ['Profit %', 'Portfolio $', 'Drawdown %', 'Cum Profit $', 'Max. Cum Drawdown %']:
            header_df[col] = ''

        # Convert strategy_results to a DataFrame
        results_df = pd.DataFrame(strategy_results)

        # Concatenate the results and then append the header at the end
        df = pd.concat([results_df, header_df], ignore_index=True)

        # Save to CSV
        filename = f"{strategy.replace(' ', '_').replace('-', '_')}_{timestamp}_results.csv"
        filepath = os.path.join(PortfolioSummaryAnalyzer._OUTPUT_DIR, filename)

        # Ensure the output directory exists
        os.makedirs(PortfolioSummaryAnalyzer._OUTPUT_DIR, exist_ok=True)

        # Overwrite the file (incremental logic removed since we're always overwriting)
        df.to_csv(filepath, index=False)