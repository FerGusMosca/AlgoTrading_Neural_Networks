import pandas as pd

from logic_layer.trading_algos.base_class_daily_trading_backtester import BaseClassDailyTradingBacktester


class NMinBufferWFlipDailyTradingBacktester(BaseClassDailyTradingBacktester):

    _N_BUFFER=10
    def __init__(self):
        pass

    def __summarize_trading_positions__(self, result_df, portf_size, net_commissions, n_min):
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
        summary_rows = []

        position_open = False
        position_side = None
        entry_price = None
        entry_time = None
        entry_symbol = None
        future_action=None
        future_price=None
        future_time=None

        for index, row in result_df.iterrows():
            current_symbol = row['trading_symbol']
            current_time = pd.to_datetime(row['formatted_date'])
            current_action = row['action']
            current_price = row['trading_symbol_price']

            # Debug print
            print(f"Processing row {index}: time={current_time}, action={current_action}, price={current_price}")

            if not position_open:
                if current_action in [self._LONG_POS, self._SHORT_POS]:
                    # Wait for n_min minutes from the first signal
                    if index + n_min < len(result_df):
                        future_action = result_df.iloc[index + n_min]['action']
                        future_price = result_df.iloc[index + n_min]['trading_symbol_price']
                        future_time =  pd.to_datetime( result_df.iloc[index + n_min]['formatted_date'])
                        if future_action == current_action:
                            # Open position
                            position_open = True
                            position_side = current_action
                            entry_price = future_price
                            entry_time = future_time
                            entry_symbol = current_symbol
                            continue

            elif position_open:

                if current_time <=future_time:
                    continue

                if position_side == self._LONG_POS:
                    if current_action == self._SHORT_POS or current_action == self._FLAT_POS:

                        self.__close_position__(entry_symbol,entry_time,position_side,current_time,current_price,entry_price,
                                                portf_size,net_commissions,summary_rows)
                        current_action=self._FLAT_POS
                        position_open = False

                elif position_side == self._SHORT_POS:
                    if current_action == self._LONG_POS or current_action == self._FLAT_POS:
                        self.__close_position__(entry_symbol, entry_time, position_side, current_time, current_price,
                                                entry_price,
                                                portf_size, net_commissions, summary_rows)
                        current_action = self._FLAT_POS
                        position_open=False

            # Handle closing positions at the end of the day
            if index == len(result_df) - 1:
                if position_open:
                    self.__close_final_position__(entry_symbol,entry_time, position_side, current_time, current_price, entry_price, portf_size, net_commissions, summary_rows)
                    current_action = self._FLAT_POS
                    position_open = False
        # Convert the list of dictionaries to a DataFrame
        trading_summary_df = pd.DataFrame(summary_rows)

        return trading_summary_df


    #region Public Methods

    def backtest_daily_predictions(self,rnn_predictions_df,portf_size,trade_comm):

        trading_summary_df = self.__summarize_trading_positions__(rnn_predictions_df, portf_size, trade_comm,
                                                                  self._N_BUFFER)

        return self.__calculate_day_trading_summary__(trading_summary_df)
    #endregion