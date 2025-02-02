import numpy as np

from common.enums.columns_prefix import ColumnsPrefix


class PortfMultPositions():
    def __init__(self,date,p_symbols_csv,side,portf_positions_arr,init_MTM):
        self.symbol=p_symbols_csv
        self.date_open=date
        self.side=side
        self.portf_positions=portf_positions_arr
        self.price_open=round(init_MTM,2)
        self.init_MTM=init_MTM
        self.daily_MTMs = [init_MTM]


    def calculate_and_append_MTM(self,prices_row,date,error_if_missing=True):

        final_MTM=0
        for portf_position in self.portf_positions:
            curr_price = prices_row[ColumnsPrefix.CLOSE_PREFIX.value + portf_position.symbol]

            if np.isnan(curr_price):
                if error_if_missing:
                    raise Exception(f"Could not find a price for symbol {portf_position.symbol} on date {date}! ")
                else:
                    final_MTM=None
                    break #nothing more to calculate
            else:
                final_MTM+=curr_price*portf_position.units

        if final_MTM is not None:
            self.append_today_MTM(final_MTM)

        return final_MTM

    def append_today_MTM(self,today_MTM):
        self.daily_MTMs.append(today_MTM)