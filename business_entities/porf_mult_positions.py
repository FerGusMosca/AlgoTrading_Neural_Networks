import numpy as np

from common.enums.columns_prefix import ColumnsPrefix


class PortfMultPositions():
    def __init__(self,date,p_symbols_csv,side,portf_positions_arr,init_MTM):
        self.symbol=p_symbols_csv
        self.date_open=date
        self.side=side
        self.portf_positions=portf_positions_arr
        self.price_open=init_MTM
        self.init_MTM=init_MTM


    def calcualte_MTM(self,prices_row,date):

        final_MTM=0
        for portf_position in self.portf_positions:
            curr_price = prices_row[ColumnsPrefix.CLOSE_PREFIX.value + portf_position.symbol]

            if np.isnan(curr_price):
                raise Exception(f"Could not find a price for symbol {portf_position.symbol} on date {date}! ")

            final_MTM+=curr_price*portf_position.units

        return final_MTM