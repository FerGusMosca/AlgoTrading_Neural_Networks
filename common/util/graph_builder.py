import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from common.enums.side import Side


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
