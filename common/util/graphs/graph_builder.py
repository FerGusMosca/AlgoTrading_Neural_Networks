import plotly.graph_objs as go
from common.enums.side import Side
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors

class GraphBuilder:

    @staticmethod
    def build_candles_graph(min_candles_df, mov_avg_unit=20):
        # Ensure that the 'date' column is of type datetime
        min_candles_df['date'] = pd.to_datetime(min_candles_df['date'])

        # Calculate the moving average based on the 'close' price
        min_candles_df['moving_avg'] = min_candles_df['close'].rolling(window=mov_avg_unit).mean()

        # Create the candlestick chart using Plotly's go.Figure
        fig = go.Figure(data=[go.Candlestick(x=min_candles_df['date'],
                                             open=min_candles_df['open'],
                                             high=min_candles_df['high'],
                                             low=min_candles_df['low'],
                                             close=min_candles_df['close'],
                                             name='Candlesticks')])

        # Add the moving average line (in red) to the chart
        fig.add_trace(go.Scatter(x=min_candles_df['date'],
                                 y=min_candles_df['moving_avg'],
                                 mode='lines',
                                 line=dict(color='red', width=2),
                                 name=f'Moving Avg ({mov_avg_unit} min)'))

        # Update chart layout for better readability
        fig.update_layout(
            title='Candlestick Chart with Moving Average',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False  # Hide the range slider for simplicity
        )

        # Show the final chart
        fig.show()

    @staticmethod
    def plot_prices_with_trades(symbol_prices_df, summary_dict_arr, strategy_name):
        # Ensure 'date' column is in datetime format
        symbol_prices_df['date'] = pd.to_datetime(symbol_prices_df['date'])

        # Create base price plot in blue
        plt.figure(figsize=(16, 6))
        plt.plot(symbol_prices_df['date'], symbol_prices_df['close'], label='Price', color='blue')

        # Loop through all trade summaries
        for summary in summary_dict_arr:
            for pos in summary[strategy_name].portf_pos_summary:
                start = pos.date_open
                end = pos.date_close
                color = 'green' if pos.side == Side.LONG.value else 'red'

                # Highlight the segment corresponding to each trade
                mask = (symbol_prices_df['date'] >= start) & (symbol_prices_df['date'] <= end)
                plt.plot(
                    symbol_prices_df.loc[mask, 'date'],
                    symbol_prices_df.loc[mask, 'close'],
                    linewidth=3,
                    color=color
                )

        plt.title("Prices with Trade Periods")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=True)

    @staticmethod
    def plot_long_probability_distributions(prob,threshold):
        # Log distribution of predicted probabilities
        plt.hist(prob, bins=50)
        plt.title(f"Distribution of LONG probabilities (threshold={threshold})")
        plt.xlabel("Probability of LONG")
        plt.ylabel("Frequency")
        plt.savefig("long_prob_distribution.png")

    @staticmethod
    def plot_feature_importances(model, feature_names, output_path="feature_importance.png"):
        import matplotlib.pyplot as plt
        import numpy as np

        # Validate consistency between feature names and model internals
        if len(feature_names) != len(model.feature_importances_):
            raise ValueError(
                f"[ERROR] Feature mismatch: model has {len(model.feature_importances_)} features "
                f"but received {len(feature_names)} feature names."
            )

        # Compute importance ranking
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]

        # Log feature importance values
        print("\n[FEATURE IMPORTANCE RANKING]")
        for i in indices:
            print(f"{feature_names[i]:<35} → {importances[i]:.4f}")

        # Plot bar chart
        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), sorted_features, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_path)  # Use plt.show() if running interactively

    @staticmethod
    def plot_2_series_overlapped(pivot_df, benchmark_df, benchmark_symbol, output_symbol):
        import matplotlib.dates as mdates

        # Ensure datetime index
        pivot_df.index = pd.to_datetime(pivot_df["date"])

        # Create base figure
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot Lightweight Indicator (left axis)
        ax1.plot(
            pivot_df.index,
            pivot_df[output_symbol],
            label=output_symbol,
            color="blue"
        )
        ax1.set_ylabel("Lightweight Indicator", color="blue")
        ax1.tick_params(axis='y', labelcolor='blue')

        # Plot Benchmark (right axis)
        if benchmark_df is not None:
            benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
            benchmark_pivot = benchmark_df.pivot(index="date", columns="symbol", values="close")

            # Align to common date range
            common_start = pivot_df.index.min()
            common_end = pivot_df.index.max()
            benchmark_pivot = benchmark_pivot[
                (benchmark_pivot.index >= common_start) & (benchmark_pivot.index <= common_end)
                ]

            ax2 = ax1.twinx()
            ax2.plot(
                benchmark_pivot.index,
                benchmark_pivot[benchmark_symbol],
                label=benchmark_symbol,
                color="orange",
                linestyle="--"
            )
            ax2.set_ylabel(benchmark_symbol, color="orange")
            ax2.tick_params(axis='y', labelcolor='orange')

        # Combined legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        if benchmark_df is not None:
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
        else:
            ax1.legend(loc="upper left")

        # Improve X-axis date formatting
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        fig.autofmt_xdate()

        # Final tweaks
        ax1.set_title("Lightweight Indicator vs Benchmark (dual axis)")
        ax1.grid(True)

        # Enable interactive hover (optional)
        try:
            mplcursors.cursor(ax1.lines + (ax2.lines if benchmark_df is not None else []), hover=True)
        except Exception:
            pass

        plt.tight_layout()
        plt.show()


