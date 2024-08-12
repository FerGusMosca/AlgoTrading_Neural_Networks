class PortfSummary:

    def __init__(self,symbol,p_portf_position_size):
        self.symbol=symbol
        self.portf_pos_summary=[]
        self.portf_pos_size=p_portf_position_size
        self.max_drawdown=0


    def append_position_summary(self,pos_summary):
        self.portf_pos_summary.append(pos_summary)



    def calculate_th_nom_profit(self):

        accum=0
        for portf_pos_summary  in self.portf_pos_summary:
            accum+=portf_pos_summary.th_nom_profit


        return accum
