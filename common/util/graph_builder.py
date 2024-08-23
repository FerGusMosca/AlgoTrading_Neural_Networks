import pandas as pd
import plotly.graph_objs as go

class GraphBuilder:


    @staticmethod
    def build_candles_graph (min_candles_df):
        min_candles_df['date'] = pd.to_datetime(min_candles_df['date'])

        fig = go.Figure(data=[go.Candlestick(x=min_candles_df['date'],
                                             open=min_candles_df['open'],
                                             high=min_candles_df['high'],
                                             low=min_candles_df['low'],
                                             close=min_candles_df['close'])])

        fig.update_layout(
            title='Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False
        )

        # Muestra el gráfico
        fig.show()
