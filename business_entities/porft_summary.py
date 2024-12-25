from business_entities.portf_position_summary import PortfPositionSummary
from common.util.financial_calculation_helper import FinancialCalculationsHelper


class PortfSummary:

    def __init__(self,symbol,p_portf_position_size,p_trade_comm=0,p_trading_algo=None,
                 p_algo_params=[]):
        self.trading_algo=p_trading_algo
        self.n_algo_params=p_algo_params
        self.symbol=symbol
        self.total_net_profit=0

        self.portf_pos_summary=[]
        self.max_cum_drawdowns=[]
        self.daily_profits=[]

        self.portf_pos_size=p_portf_position_size
        self.trade_comm=p_trade_comm

        self.max_drawdown=0
        self.max_daily_drawdown=0
        self.accum_positions=0


    def append_position_summary(self,pos_summary):
        self.portf_pos_summary.append(pos_summary)



    def calculate_th_nom_profit(self):

        accum=0
        for portf_pos_summary  in self.portf_pos_summary:
            accum+=portf_pos_summary.th_nom_profit


        return accum


    def update_max_drawdown(self):
        self.max_daily_drawdown = min(self.max_cum_drawdowns) if len(self.max_cum_drawdowns) is None else 0
        self.max_drawdown = FinancialCalculationsHelper.calculate_max_total_drawdown(self.daily_profits)


    def calculate_last_portf_position_summary(self,day):
        position_summary= PortfPositionSummary(day,self.daily_profits[-1],self.max_cum_drawdowns[-1])
        self.portf_pos_summary.append(position_summary)
        return  position_summary
