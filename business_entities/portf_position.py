from datetime import timedelta

import pandas as pd

from business_entities.detailed_MTM import DetailedMTM


class PortfolioPosition():

    _SIDE_LONG = "LONG"
    _SIDE_SHORT = "SHORT"
    _DEF_PORTF_AMT = 100000


    @staticmethod
    def get_def_portf_amt():
        return  PortfolioPosition._DEF_PORTF_AMT

    def __init__(self,p_symbol):
        self.symbol =p_symbol
        self.units=None
        self.side=None
        self.date_open=None
        self.price_open=None
        self.date_close=None
        self.price_close=None
        self.daily_MTMs=[]
        self.detailed_MTMs=[]


    def open_pos(self,side,date,price,units=None):
        self.side=side
        self.date_open=date
        self.units=units

        if isinstance(price, pd.Series):
            self.price_open = float(price.iloc[0])
        else:
            self.price_open=float(price)
        if self.units is not None:
            self.append_MTM(date,price*units)

    def close_pos(self, date, price):
        self.date_close = date

        if isinstance(price, pd.Series):
            self.price_close = float(price.iloc[0])
        else:
            self.price_close = float(price)

        if self.units is not None:
            self.calculate_and_append_MTM(date, self.price_close)

    def append_MTM(self, date,MTM):
        self.daily_MTMs.append(MTM)
        self.detailed_MTMs.append(DetailedMTM(date,MTM))

    def calculate_and_append_MTM(self, date, price):
        if price is not None and self.units is not None:
            if self.side == PortfolioPosition._SIDE_LONG:
                final_MTM = price * self.units
            elif self.side == PortfolioPosition._SIDE_SHORT:
                price_diff = self.price_open - price
                final_MTM = self.price_open * self.units + price_diff * self.units
            else:
                raise Exception(f"Unknown position side for symbol {self.symbol}")

            self.append_MTM(date, final_MTM)
            return final_MTM

        return price

    def is_open(self):
        return self.date_close is None


    def calculate_pct_profit(self):
        if self.price_close is not None and self.price_open is not None:

            if self.side==PortfolioPosition._SIDE_LONG:
                if self.price_open >0:
                    return round( ((self.price_close-self.price_open)/self.price_open)*100,2)
                else:
                    raise Exception("Could not divide by 0 on price_open=0 for symbol {}".format(self.symbol))
            elif self.side == PortfolioPosition._SIDE_SHORT:
                    if self.price_open > 0:
                        return round(((self.price_open- self.price_close ) / self.price_open) * 100, 2)
                    else:
                        raise Exception("Could not divide by 0 on price_open=0 for symbol {}".format(self.symbol))
        else :
            raise Exception("Position for symbol {} not properly closed!  cannot calculate the pct profit".format(self.symbol))

    def calculate_th_nom_profit(self, portf_amt=_DEF_PORTF_AMT):
        pct_profit = self.calculate_pct_profit()

        if portf_amt is not None:
            return round((pct_profit / 100) * portf_amt)

        elif self.units is not None and self.price_open is not None:
            implied_portf_amt = self.units * self.price_open
            return round((pct_profit / 100) * implied_portf_amt)

        else:
            raise Exception(f"Cannot calculate theoretical nominal profit: "
                            f"missing portf_amt, price_open or units for symbol {self.symbol}")

    @staticmethod
    def fill_missing_dates(portfolio_list: list) -> list:
        """
        Receives a list of PortfolioRecord objects ordered by date (ascending)
        and returns a new list where any missing dates are filled in.
        The missing dates are filled with the last available MTM value.

        Parameters:
            portfolio_list (list): List of PortfolioRecord objects representing active portfolio days.

        Returns:
            list: A new list of PortfolioRecord objects with consecutive dates.
        """
        # Ensure the list is sorted by date in ascending order
        portfolio_list.sort(key=lambda x: x.date)

        # This will hold the final list with consecutive dates
        filled_list = []
        last_mtm = None

        # Define the date range from the first record's date to the last record's date
        start_date = portfolio_list[0].date
        end_date = portfolio_list[-1].date

        # Initialize index for iterating through the original list
        index = 0
        n = len(portfolio_list)

        # Iterate over each day in the date range
        current_date = start_date
        while current_date <= end_date:
            # If there is a record for the current_date, use it and update last_mtm
            if index < n and portfolio_list[index].date == current_date:
                last_mtm = portfolio_list[index].MTM
                filled_list.append(portfolio_list[index])
                index += 1
            else:
                # If there is no record for current_date, create a new record using the last available MTM
                filled_list.append(DetailedMTM(current_date, last_mtm))
            current_date += timedelta(days=1)

        return filled_list
