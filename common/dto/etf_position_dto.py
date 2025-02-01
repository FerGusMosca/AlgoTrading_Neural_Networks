import math


class ETFPositionDTO():

    def __init__(self,p_weight,p_symbol):
        self.weight=float(p_weight)
        self.symbol=p_symbol

        self.active=True
        self.active_weight=self.weight



    @staticmethod
    def build_etf_constituents_dto_arr(symbols_csv,weights_csv):
        symbols_list=symbols_csv.split(",")
        weights_list=weights_csv.split(",")

        etfs_const_arr=[]
        for index, (symbol, weight) in enumerate(zip(symbols_list, weights_list), start=1):
            pos_dto=ETFPositionDTO(weight,symbol)
            etfs_const_arr.append(pos_dto)

        return  etfs_const_arr

    @staticmethod
    def validate_weights(etf_positions_dto_arr):

        sum=0

        for etf_pos in etf_positions_dto_arr:
            sum+=etf_pos.weight

        if not math.isclose(sum, 1.0, rel_tol=1e-9):
            raise Exception("ETF composition must be 100% between all the weights (range 0 to 1)")


    @staticmethod
    def recalculate_weights(etfs_const_arr, etfs_active_const_arr):
        """
        Recalculates the weights of active ETF positions based on the original weights from etfs_const_arr.
        :param etfs_const_arr: List of all ETFPositionDTO instances representing the full ETF composition.
        :param etfs_active_const_arr: List of ETFPositionDTO instances representing only active positions (traded securities).
        """
        # Calculate total original weight of active positions
        total_weight_active = sum(position.weight for position in etfs_active_const_arr if position.active)

        # Calculate adjustment factor
        if total_weight_active == 0:
            return  # Avoid division by zero if no active positions

        original_total_weight = sum(position.weight for position in etfs_const_arr)
        adjustment_factor = original_total_weight / total_weight_active

        # Recalculate active weights
        for position in etfs_active_const_arr:
            if position.active:
                position.active_weight = position.weight * adjustment_factor




