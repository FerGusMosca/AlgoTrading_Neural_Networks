from business_entities.portf_position import PortfolioPosition
from logic_layer.trading_algos.base_class_daily_trading_backtester import BaseClassDailyTradingBacktester
import pandas as pd

class OnlySignalNMinMovAvgBacktester(BaseClassDailyTradingBacktester):
    _N_BUFFER = 10
    _N_MOV_AVG = 80
    def __init__(self):
        """
        Initializes the backtester.

        :param prices: Time series of prices.
        :param n_min_periods: Minimum number of minutes to hold a position.
        :param mov_avg_window: Window size (in minutes) for moving average calculation.
        :param buffer_minutes: Minimum number of minutes to wait before opening a new position.
        """
        pass

    def __calc_mov_avg__(self,predictions_df, current_time, n_mov_avg_unit):
        """
        Calculates the moving average of the asset price for the past n minutes from the given current_time.

        :param predictions_df: DataFrame containing the asset price data with columns 'formatted_date' and 'trading_symbol_price'.
        :param current_time: The current time as a pandas datetime object.
        :param n_mov_avg_unit: The number of minutes to consider for the moving average.
        :return: The calculated moving average of prices over the last n minutes.
        """
        # Ensure 'formatted_date' column is in datetime format
        predictions_df['formatted_date'] = pd.to_datetime(predictions_df['formatted_date'])

        # Define the start of the time window (n minutes before the current_time)
        time_window_start = current_time - pd.Timedelta(minutes=n_mov_avg_unit)

        # Filter the DataFrame for rows where 'formatted_date' is within the time window (inclusive of current_time)
        filtered_df = predictions_df[
            (predictions_df['formatted_date'] >= time_window_start) &
            (predictions_df['formatted_date'] <= current_time)
            ]

        # Calculate and return the moving average of 'trading_symbol_price' in the filtered window
        if not filtered_df.empty and len(filtered_df)>n_mov_avg_unit:
            return filtered_df['trading_symbol_price'].mean()
        else:
            # Return NaN if no data is available in the specified time window
            return float('nan')


    def __eval_momentum__(self,predictions_df,trade_action, current_time,current_price, n_mov_avg_unit):

        if trade_action==self._LONG_POS and self.__calc_mov_avg__(predictions_df, current_time, n_mov_avg_unit) <=current_price:
            return  True
        elif trade_action==self._LONG_POS:
            return  False
        elif trade_action==self._SHORT_POS and self.__calc_mov_avg__(predictions_df, current_time, n_mov_avg_unit) >=current_price:
            return  True
        elif trade_action == self._SHORT_POS:
            return  False
        else:
            raise Exception(f"Invalid trade_action : {trade_action}")
    def __summarize_trading_positions__(self, predictions_df, portf_size, net_commissions, n_min_buffer,n_mov_avg_unit):
        """
        Summarizes trading positions using the N_MIN_BUFFER_W_FLIP algorithm.

        Parameters:
        result_df (pd.DataFrame): DataFrame containing trading data with columns ['trading_symbol', 'formatted_date', 'action', 'trading_symbol_price']
        pos_size (float): The size of each position in monetary terms.
        net_commissions (float): The net commissions to be deducted from the profit.
        n_min (int): The number of minutes to wait before flipping positions (N-minute buffer).

        Returns:
        pd.DataFrame: DataFrame summarizing the trading positions with columns ['symbol', 'open', 'close', 'side', 'price_open', 'price_close', 'unit_gross_profit', 'total_gross_profit', 'total_net_profit', 'portfolio_size', 'pos_size']
        """
        # Initialize an empty list to store the rows of the trading summary
        trading_summary_df =  self.__initialize_dataframe__()

        portf_pos = None
        future_time=None

        for index, row in predictions_df.iterrows():
            current_symbol = row['trading_symbol']
            current_time = pd.to_datetime(row['formatted_date'])
            current_action = row['action']
            current_price = row['trading_symbol_price']

            # Debug print
            print(f"Processing row {index}: time={current_time}, action={current_action}, price={current_price}")

            if portf_pos is None:
                if current_action in [self._LONG_POS, self._SHORT_POS]:
                    # Wait for n_min minutes from the first signal
                    if index + n_min_buffer < len(predictions_df):
                        future_action = predictions_df.iloc[index + n_min_buffer]['action']
                        future_price = predictions_df.iloc[index + n_min_buffer]['trading_symbol_price']
                        future_time =  pd.to_datetime(predictions_df.iloc[index + n_min_buffer]['formatted_date'])
                        if (future_action == current_action
                            and self.__eval_momentum__(predictions_df,future_action,future_time,future_price,n_mov_avg_unit)):
                            # Open position
                            portf_pos = PortfolioPosition(current_symbol)
                            portf_pos.open_pos(current_action, future_time, future_price)
                            trading_summary_df = self.__open_portf_position__(portf_pos, portf_size, trading_summary_df)
                            continue

            else: #Position Opened in t+10

                if current_time <=future_time:
                    continue
                #We only close everything because of mom
                if not self.__eval_momentum__(predictions_df, portf_pos.side, current_time, current_price, n_mov_avg_unit):
                    trading_summary_df = self.__close_portf_position__(portf_pos, current_time, current_price,
                                                                       portf_size, net_commissions, trading_summary_df)
                    portf_pos = None

            # Handle closing positions at the end of the day
            if index == len(predictions_df) - 1:
                if portf_pos is not None:
                    trading_summary_df = self.__close_portf_position__(portf_pos, current_time, current_price,
                                                                       portf_size, net_commissions, trading_summary_df)
                    portf_pos = None

        return trading_summary_df



    def backtest_daily_predictions(self,rnn_predictions_df,portf_size,trade_comm,n_algo_params):

        if(len(n_algo_params)==2):
            OnlySignalNMinMovAvgBacktester.N_BUFFER=int(n_algo_params[0])
            OnlySignalNMinMovAvgBacktester._N_MOV_AVG = int(n_algo_params[1])

        trading_summary_df = self.__summarize_trading_positions__(rnn_predictions_df, portf_size, trade_comm,
                                                                  self._N_BUFFER,self._N_MOV_AVG )

        return self.__calculate_day_trading_single_pos_summary__("only_signal_n_min_mov_avg", trading_summary_df, None)



