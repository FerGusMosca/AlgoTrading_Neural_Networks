import pandas as pd

from business_entities.portf_position import PortfolioPosition
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
        trading_summary_df =  self.__initialize_dataframe__()

        portf_pos = None
        future_time=None

        for index, row in result_df.iterrows():
            current_symbol = row['trading_symbol']
            current_time = pd.to_datetime(row['formatted_date'])
            current_action = row['action']
            current_price = row['trading_symbol_price']

            # Debug print
            print(f"Processing row {index}: time={current_time}, action={current_action}, price={current_price}")

            if portf_pos is None:
                if current_action in [self._LONG_POS, self._SHORT_POS]:
                    # Wait for n_min minutes from the first signal
                    if index + n_min < len(result_df):
                        future_action = result_df.iloc[index + n_min]['action']
                        future_price = result_df.iloc[index + n_min]['trading_symbol_price']
                        future_time =  pd.to_datetime( result_df.iloc[index + n_min]['formatted_date'])
                        if future_action == current_action:
                            # Open position
                            portf_pos = PortfolioPosition(current_symbol)
                            portf_pos.open_pos(current_action, future_time, future_price)
                            trading_summary_df = self.__open_portf_position__(portf_pos, portf_size, trading_summary_df)
                            continue

            else: #Position Opened in t+10

                if current_time <=future_time:
                    continue

                if portf_pos.side == self._LONG_POS:
                    if current_action == self._SHORT_POS or current_action == self._FLAT_POS:
                        trading_summary_df = self.__close_portf_position__(portf_pos, current_time, current_price,
                                                                           portf_size, net_commissions,
                                                                           trading_summary_df)
                        portf_pos = None

                elif portf_pos.side == self._SHORT_POS:
                    if current_action == self._LONG_POS or current_action == self._FLAT_POS:
                        trading_summary_df = self.__close_portf_position__(portf_pos, current_time, current_price,
                                                                           portf_size, net_commissions,
                                                                           trading_summary_df)
                        portf_pos = None

            # Handle closing positions at the end of the day
            if index == len(result_df) - 1:
                if portf_pos is not None:
                    trading_summary_df = self.__close_portf_position__(portf_pos, current_time, current_price,
                                                                       portf_size, net_commissions,
                                                                       trading_summary_df)
                    portf_pos = None

        return trading_summary_df


    #region Public Methods

    def backtest_daily_predictions(self,rnn_predictions_df,portf_size,trade_comm,n_algo_params):

        if(len(n_algo_params)>0):
            NMinBufferWFlipDailyTradingBacktester.N_BUFFER=int(n_algo_params[0])

        trading_summary_df = self.__summarize_trading_positions__(rnn_predictions_df, portf_size, trade_comm,
                                                                  self._N_BUFFER)

        return self.__calculate_day_trading_summary__(trading_summary_df)
    #endregion