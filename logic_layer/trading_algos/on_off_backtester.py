from logic_layer.trading_algos.slope_backtester import SlopeBacktester


class OnOffBacktester(SlopeBacktester):


    def __init__(self):
        pass


    def long_signal(self,current_value,current_slope):
        return current_value>0

    def close_long_signal(self,current_value,current_slope):
        return  current_value<0

    def get_algo_name(self):
        return "on_off_value"
