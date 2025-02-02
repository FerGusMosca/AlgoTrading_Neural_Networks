class StrategySummaryDTO():


    def __init__(self,algo,daily_net_profit,total_positions,max_cum_drawdown,trading_summary_df):
        self.algo=algo
        self.daily_net_profit=daily_net_profit
        self.total_positions=total_positions
        self.max_cum_drawdown=max_cum_drawdown
        self.trading_summary_df=trading_summary_df