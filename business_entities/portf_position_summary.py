class PortfPositionSummary():


    def __init__(self,p_portf_pos,p_pct_profit,p_th_nom_profit, p_nom_pos_size):

        self.portf_pos=p_portf_pos
        self.pct_profit=p_pct_profit
        self.th_nom_profit=p_th_nom_profit
        self.nom_pos_size=p_nom_pos_size
        self.max_drawdown=None



    def __init__(self,p_day,p_day_nom_profit,p_day_drawdown):

        self.day=p_day
        self.day_nom_profit=p_day_nom_profit
        self.day_drawdown=p_day_drawdown