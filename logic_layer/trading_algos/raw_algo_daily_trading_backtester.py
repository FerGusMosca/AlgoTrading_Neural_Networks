import pandas as pd

from business_entities.portf_position import PortfolioPosition
from logic_layer.trading_algos.base_class_daily_trading_backtester import BaseClassDailyTradingBacktester


class RawAlgoDailyTradingBacktester(BaseClassDailyTradingBacktester):

    def __init__(self):
        pass


    #region Private Methods

    def __summarize_trading_positions__(self, result_df, portf_size, net_commissions):
        """
        Summarizes trading positions from the result_df DataFrame.

        Parameters:
        result_df (pd.DataFrame): DataFrame containing trading data with columns ['trading_symbol', 'formatted_date', 'action', 'trading_symbol_price']
        portfolio_size (float): The total size of the portfolio in monetary terms.
        net_commissions (float): The net commissions to be deducted from the profit.

        Returns:
        pd.DataFrame: DataFrame summarizing the trading positions with columns ['symbol', 'open', 'close', 'side', 'price_open', 'price_close', 'unit_gross_profit', 'total_gross_profit', 'total_net_profit', 'portfolio_size', 'pos_size']
        """

        # Initialize an empty DataFrame to store the trading summary
        trading_summary_df = self.__initialize_dataframe__()

        # Variables to keep track of the open position
        portf_pos = None

        # Iterate through each row in the result_df DataFrame
        for index, row in result_df.iterrows():
            current_symbol = row['trading_symbol']
            current_time = row['formatted_date']
            current_action = row['action']
            current_price = row['trading_symbol_price']

            # If a new position is opened
            if (not portf_pos) and (current_action in [self._LONG_POS,self._SHORT_POS]):

                portf_pos= PortfolioPosition(current_symbol)
                portf_pos.open_pos(current_action,current_time,current_price)

                trading_summary_df=self.__open_portf_position__(portf_pos, portf_size, trading_summary_df)

            # If an open position is closed
            elif portf_pos is not None and (
                    (portf_pos.side == self._LONG_POS and current_action in [self._SHORT_POS, self._FLAT_POS]) or
                    (portf_pos.side == self._SHORT_POS and current_action in [self._LONG_POS, self._FLAT_POS])
            ):
                trading_summary_df=self.__close_portf_position__(portf_pos, current_time, current_price, portf_size, net_commissions, trading_summary_df)
                portf_pos=None

            # Handle closing positions at the end of the day
            if index == len(result_df) - 1:
                if portf_pos is not None:
                    trading_summary_df=self.__close_portf_position__(portf_pos, current_time, current_price, portf_size, net_commissions,
                                                                     trading_summary_df)
                    portf_pos = None
        # Return the trading summary DataFrame
        return trading_summary_df

    #endregion


    #region Public Methods

    def backtest_daily_predictions(self,rnn_predictions_df,portf_size,trade_comm):

        trading_summary_df = self.__summarize_trading_positions__(rnn_predictions_df, portf_size, trade_comm)

        return self.__calculate_day_trading_summary__(trading_summary_df)

    #endregion