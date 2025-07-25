import os
from datetime import datetime

import pandas as pd

from common.util.financial_calculations.financial_calculation_helper import FinancialCalculationsHelper


class PortfolioSummaryAnalyzer:

    _OUTPUT_DIR="./output/csv"

    def __init__(self):
        pass

    @staticmethod
    def __group_returns_n_drawdown__(summary_dict_arr, strategy_key):
        all_max_MTMs_dict = {}
        all_max_drawdowns_dict = {}

        for entry in summary_dict_arr:
            if strategy_key in entry:
                summary = entry[strategy_key]
                year = summary.year

                mtm_return = (summary.portf_final_MTM / summary.portf_init_MTM) - 1
                drawdown = summary.max_daily_drawdown

                if year not in all_max_MTMs_dict:
                    all_max_MTMs_dict[year] = []
                if year not in all_max_drawdowns_dict:
                    all_max_drawdowns_dict[year] = []

                all_max_MTMs_dict[year].append(mtm_return)
                all_max_drawdowns_dict[year].append(drawdown)

        return all_max_MTMs_dict, all_max_drawdowns_dict

    @staticmethod
    def calculate_portfolio_metrics(result, init_portf, start, end, timestamp):
        """
        Calculate the final portfolio value, cumulative maximum drawdown, and format results into an array for each strategy.
        Export the results to separate CSV files for each strategy with a timestamp, updating incrementally.
        Include annual summaries with CAGR and Max DD per year.

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
        final_results = {strategy: {
            'final_portfolio_value': 0.0,
            'cumulative_max_drawdown': "0.0%",
            'formatted_results': [],
            'annual_summary': []  # Lista para el resumen anual por estrategia
        } for strategy in strategies}

        # Process each strategy
        for strategy in strategies:
            # Initialize variables for this strategy
            portfolio_value = init_portf
            portfolio_values = [init_portf]  # Track portfolio value over time for drawdown calculation
            max_drawdown = 0.0  # Cumulative maximum drawdown
            annual_portfolio_values = {}  # Track portfolio value by year
            annual_drawdowns = {}  # Track drawdowns by year
            formatted_results = []  # Array de diccionarios para los resultados periodicos

            # Process each dictionary in the result array
            for period_index, perf_dict in enumerate(result):
                # Extract the strategy-specific PortfSummary object
                strategy_summary = perf_dict.get(strategy)

                # Use the start of the period (date_open) to determine the bimonthly period
                year = strategy_summary.year
                period = strategy_summary.period

                # Use total_net_profit and max_drawdown_on_MTM directly (assuming they are floats or strings)
                total_net_profit = strategy_summary.total_net_profit
                max_drawdown_on_mtm = strategy_summary.max_drawdown_on_MTM

                # Convert to decimal for calculation
                total_net_profit_decimal = total_net_profit
                period_drawdown = max_drawdown_on_mtm

                # Apply the return to the portfolio value
                portfolio_value = portfolio_value * (1 + total_net_profit_decimal)
                portfolio_values.append(portfolio_value)

                # Store portfolio value and drawdown for annual summary
                if year not in annual_portfolio_values:
                    annual_portfolio_values[year] = []
                    annual_drawdowns[year] = []
                annual_portfolio_values[year].append(portfolio_value)
                annual_drawdowns[year].append(period_drawdown)

                # Calculate cumulative profit percentage
                cum_profit = ((portfolio_value / init_portf) - 1) * 100.0

                # Update maximum drawdown (track the worst drawdown observed)
                max_drawdown = min(max_drawdown, period_drawdown)

                # Add the result to the formatted array for this strategy
                formatted_results.append({
                    'Year': year,
                    'Period': period,
                    'Profit %': round(total_net_profit * 100, 2),
                    'Drawdown %': round(max_drawdown_on_mtm * 100, 2),
                    'Cum Profit $': round(cum_profit, 2),
                    'Max. Cum Drawdown %': round(max_drawdown * 100, 2),
                    'Portfolio $': round(portfolio_value, 2)
                })

            # Calculate annual summaries for this strategy
            annual_summary = []
            years = sorted(annual_portfolio_values.keys())
            for year in years:
                year_end_portf = annual_portfolio_values[year][-1]  # Last portfolio value of the year
                year_start_portf = annual_portfolio_values[year][0] if len(
                    annual_portfolio_values[year]) > 1 else init_portf
                num_years = 1.0  # Assuming one year per block for simplicity
                cagr = ((year_end_portf / year_start_portf) ** (1 / num_years) - 1) * 100 if num_years > 0 else 0.0
                max_dd_year = min(annual_drawdowns[year]) * 100  # Convert to percentage

                annual_summary.append({
                    'Year': year,
                    'CAGR %': round(cagr, 2),
                    'Max DD %': round(max_dd_year, 2)
                })

            #CAGR
            num_years_total = len(years)
            cagr_total = ((portfolio_value / init_portf) ** (
                        1 / num_years_total) - 1) * 100 if num_years_total > 0 else 0.0

            annual_summary.append({'Year': 'Cumulative CAGR', 'Period': f"{round(cagr_total, 2)}%"})

            # Update the CSV file with the formatted results and annual summary
            PortfolioSummaryAnalyzer.__strategy_results_to_csv__(formatted_results, portfolio_value, max_drawdown,
                                                                 strategy, timestamp, annual_summary=annual_summary,
                                                                 incremental=False, cagr_total=cagr_total)

            # Store the results for this strategy
            final_results[strategy] = {
                'final_portfolio_value': portfolio_value,
                'cumulative_max_drawdown': f"{round(max_drawdown * 100, 2)}%",
                'formatted_results': formatted_results,
                'annual_summary': annual_summary
            }

        return final_results

    @staticmethod
    def convert_summary_dict_arr_to_dataframe(summary_dict_arr,symbol, series_csv, strategy_key="SLIDING_RF", init_portf=100000,
                                              timestamp="",):
        """
        Converts summary_dict_arr to DataFrames and saves them to CSV, with correct cumulative metrics.
        """
        formatted_results = []
        annual_portfolio_values = {}
        annual_drawdowns = {}

        max_drawdown_accum = float("inf")  # for cumulative min tracking

        all_max_MTMs_dict, all_max_drawdowns_dict= PortfolioSummaryAnalyzer.__group_returns_n_drawdown__(summary_dict_arr, strategy_key)

        for entry in summary_dict_arr:
            summary = entry[strategy_key]
            year = summary.year
            period = summary.period
            portf_value = summary.portf_final_MTM
            portf_init = summary.portf_init_MTM
            max_drawdown = summary.max_daily_drawdown
            profit_pct = ((portf_value / portf_init) - 1) * 100
            cum_profit_nominal = portf_value - init_portf
            max_drawdown_accum = min(max_drawdown_accum, max_drawdown)

            formatted_results.append({
                "Year": year,
                "Period": period,
                "Profit %": round(profit_pct, 2),
                "Drawdown %": round(max_drawdown * 100, 2),
                "Cum Profit $": round(cum_profit_nominal, 2),
                "Max. Cum Drawdown %": round(max_drawdown_accum * 100, 2),
                "Portfolio $": round(portf_value, 2)
            })

            if year not in annual_portfolio_values:
                annual_portfolio_values[year] = []
                annual_drawdowns[year] = []
            annual_portfolio_values[year].append(portf_value)
            annual_drawdowns[year].append(max_drawdown)

        annual_summary = []
        prev_portf = init_portf
        for year in sorted(annual_portfolio_values.keys()):
            y_end = annual_portfolio_values[year][-1]
            y_start = prev_portf
            cagr = ((y_end / y_start) ** 1 - 1) * 100
            max_dd_year = FinancialCalculationsHelper.get_max_cum_yearly_drawdown(year,all_max_MTMs_dict,all_max_drawdowns_dict) * 100
            #max_dd_year = min(annual_drawdowns[year]) * 100

            annual_summary.append({
                "Year": year,
                "CAGR %": round(cagr, 2),
                "Max DD %": round(max_dd_year, 2)
            })
            prev_portf = y_end

        cumulative_max_drawdown = min([row["Max. Cum Drawdown %"] for row in formatted_results])
        years_total = len(annual_summary)
        final_portf = formatted_results[-1]["Portfolio $"]
        cagr_total = ((final_portf / init_portf) ** (1 / years_total) - 1) * 100 if years_total > 0 else 0.0

        PortfolioSummaryAnalyzer.__strategy_results_to_csv__(
            formatted_results=formatted_results,
            portfolio_value=final_portf,
            max_drawdown=cumulative_max_drawdown / 100,
            strategy=strategy_key,
            timestamp=timestamp,
            annual_summary=annual_summary ,
            cagr_total=round(cagr_total,2),
            incremental=False,
            symbol=symbol,
            series_csv=series_csv
        )

        return formatted_results, annual_summary

    @staticmethod
    def __strategy_results_to_csv__(formatted_results, portfolio_value, max_drawdown, strategy, timestamp,
                                annual_summary=None,cagr_total=None, incremental=False,symbol="?",series_csv="?"):
        """
        Export strategy results to a CSV file with the specified structure, including annual summaries.

        Args:
            formatted_results (list): List of dictionaries containing period-wise results.
            portfolio_value (float): Final portfolio value.
            max_drawdown (float): Cumulative maximum drawdown (as a decimal, e.g., -0.052 for -5.2%).
            strategy (str): Name of the strategy.
            timestamp (str): Timestamp to include in the filename.
            annual_summary (list): List of dictionaries with annual summaries (CAGR and Max DD).
            incremental (bool): If True, append to the existing file; if False, overwrite.
        """
        # Format portfolio_value with commas and max_drawdown as a percentage
        formatted_portfolio_value = f"${portfolio_value:,.2f}"  # e.g., $1,029,997.59

        # Create a DataFrame for the header rows
        header_data = [
            {'Year': 'Cumulative CAGR', 'Period': f"{round(cagr_total, 2)}%"},
            {'Year': 'Final Portfolio Value', 'Period': formatted_portfolio_value},
            {'Year': 'Symbol', 'Period': symbol},
            {'Year': 'Series CSV', 'Period': series_csv}
        ]

        header_df = pd.DataFrame(header_data)

        # Add empty columns to header_df to match the main DataFrame's columns
        for col in ['Profit %', 'Portfolio $', 'Drawdown %', 'Cum Profit $', 'Max. Cum Drawdown %']:
            header_df[col] = ''

        # Convert formatted_results to a DataFrame
        results_df = pd.DataFrame(formatted_results)

        # Create a DataFrame for annual summaries if provided
        annual_summary_df = pd.DataFrame(annual_summary) if annual_summary else pd.DataFrame()

        # Concatenate the results, annual summary, and header
        df = pd.concat([results_df, annual_summary_df, header_df], ignore_index=True)

        # Save to CSV
        filename = f"{strategy.replace(' ', '_').replace('-', '_')}_{timestamp}_results.csv"
        filepath = os.path.join(PortfolioSummaryAnalyzer._OUTPUT_DIR, filename)

        # Ensure the output directory exists
        os.makedirs(PortfolioSummaryAnalyzer._OUTPUT_DIR, exist_ok=True)

        # Overwrite the file
        df.to_csv(filepath, index=False)