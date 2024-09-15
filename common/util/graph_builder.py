import pandas as pd
import plotly.graph_objs as go

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
