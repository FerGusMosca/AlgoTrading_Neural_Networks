import math


class ETFPositionDTO():

    def __init__(self,p_weight,p_symbol):
        self.weight=float(p_weight)
        self.symbol=p_symbol



    @staticmethod
    def build_positions_dto_arr(symbols_csv,weights_csv):
        symbols_list=symbols_csv.split(",")
        weights_list=weights_csv.split(",")

        pos_dto_arr=[]
        for index, (symbol, weight) in enumerate(zip(symbols_list, weights_list), start=1):
            pos_dto=ETFPositionDTO(weight,symbol)
            pos_dto_arr.append(pos_dto)

        return  pos_dto_arr

    @staticmethod
    def validate_weights(etf_positions_dto_arr):

        sum=0

        for etf_pos in etf_positions_dto_arr:
            sum+=etf_pos.weight

        if not math.isclose(sum, 1.0, rel_tol=1e-9):
            raise Exception("ETF composition must be 100% between all the weights (range 0 to 1)")
