from datetime import datetime, timedelta

from tensorboard import summary

from business_entities.porft_summary import PortfSummary
from business_entities.portf_position import PortfolioPosition
from business_entities.portf_position_summary import PortfPositionSummary
import numpy as np

from common.util.date_handler import DateHandler
from common.util.financial_calculation_helper import FinancialCalculationsHelper


class IndicatorBasedTradingBacktester:

    _INDIC_CLASSIF_COL="classification"
    _INDIC_CLASSIF_START_DATE_COL = "date_start"
    _INDIC_CLASSIF_END_DATE_COL = "date_end"

    def __init__(self):
        pass


    #region private Methods

    def __get_side__(self,indic_cassif_row,inv ):

        if inv  :
            if indic_cassif_row[IndicatorBasedTradingBacktester._INDIC_CLASSIF_COL]==PortfolioPosition._SIDE_LONG:
                return PortfolioPosition._SIDE_SHORT
            else:
                return PortfolioPosition._SIDE_LONG
        else:
            return indic_cassif_row[IndicatorBasedTradingBacktester._INDIC_CLASSIF_COL] #Nothing to switch

    def __switch_side__(self,side ):

        if side == PortfolioPosition._SIDE_LONG:
            return PortfolioPosition._SIDE_SHORT
        elif side == PortfolioPosition._SIDE_SHORT:
            return PortfolioPosition._SIDE_LONG
        else:
            raise Exception("Invalid side switching sides! : {}".format(side))

    def __eval_invert_side__(self,side,invert):

        if invert:
            return  self.__switch_side__(side)
        else:
            return side
    def __get_date_price__(self,series_df,symbol,date):


        try:
            if date in series_df['date'].values:
                series_row = series_df.loc[series_df['date'] == date].iloc[0]
            else:
                series_row = series_df.sort_values(by='date', ascending=False).iloc[0]

            return series_row[symbol]

        except Exception as e:
            raise Exception("Could not find a price for symbol {} and date {}  on series_df!!!".format(symbol,date))
    def __calculate_max_drawdown__(self,position_list):
        # Init vars
        cumulative_return = 0.0
        max_return = 0.0
        max_drawdown = 0.0

        for position in position_list:
            pct_profit = position.calculate_pct_profit() / 100  # Convertir porcentaje a decimal
            cumulative_return += pct_profit

            # Actualizar el máximo retorno acumulado
            if cumulative_return > max_return:
                max_return = cumulative_return

            # Calcular el drawdown
            drawdown = cumulative_return - max_return

            # Actualizar el max drawdown
            if drawdown < max_drawdown:
                max_drawdown = drawdown

        # Convertir el max drawdown a porcentaje
        return round(max_drawdown * 100, 2)

    def calculate_portfolio_performance(self,symbol,portf_positions_arr):

        summary=PortfSummary(symbol,PortfolioPosition._DEF_PORTF_AMT)

        for pos in portf_positions_arr:
            pct_profit=pos.calculate_pct_profit()
            th_nom_profit=pos.calculate_th_nom_profit()

            portf_pos_summary=PortfPositionSummary(pos,pct_profit,th_nom_profit,PortfolioPosition.get_def_portf_amt())
            summary.append_position_summary(portf_pos_summary)


        summary.max_drawdown=self.__calculate_max_drawdown__(portf_positions_arr)

        return summary


    def calculate_portfolio_performance_summary_extended(self,symbol,portf_positions_arr,
                                                         ref_date=None):
        # Step 1: Calculate the initial and last daily MTM for profit calculation
        # We assume daily_MTMs are consistent across positions for simplicity
        first_pos = portf_positions_arr[0]
        last_pos = portf_positions_arr[-1]

        # Extract daily MTMs (list of values, e.g., [46.16, 47.14, ...])
        daily_mtms = first_pos.daily_MTMs
        if not daily_mtms or len(daily_mtms) < 2:
            return None, None, None, None

        # Get the first and last daily MTM values
        init_daily_mtm = first_pos.daily_MTMs[0]  # First MTM value
        last_daily_mtm = last_pos.daily_MTMs[-1]  # Last MTM value

        # Calculate portfolio sizes using units (number of shares/units held)
        init_portf_size = init_daily_mtm
        last_portf_size = last_daily_mtm

        # Calculate the percentage profit
        profit_pct_str = None
        profit_pct=0
        if init_portf_size != 0:  # Avoid division by zero
            profit_pct=(last_portf_size / init_portf_size)-1
            profit_pct_str= str(round(profit_pct*100,2)) + " %"

        # Step 2: Combine all daily MTMs into a single list
        all_daily_mtms = []
        for pos in portf_positions_arr:
            if hasattr(pos, 'daily_MTMs') and pos.daily_MTMs:
                all_daily_mtms.extend(pos.daily_MTMs)

        # Step 3: Calculate the maximum drawdown
        max_drawdown_str = None
        max_drawdown_pct = None
        if all_daily_mtms:
            max_drawdown_pct = FinancialCalculationsHelper.max_drawdown_on_MTM(all_daily_mtms)
            max_drawdown_pct=max_drawdown_pct #we want the value in 0-1 range
            max_drawdown_str = str(round(max_drawdown_pct * 100, 2)) + " %"

        period=None
        year=None
        if ref_date is not None:
            period=DateHandler.get_two_month_period_from_date(ref_date)
            year=ref_date.year

        summary = PortfSummary(symbol,last_daily_mtm,p_period=period,p_year=year)

        summary.portf_pos_size=init_daily_mtm
        summary.portf_init_MTM=init_daily_mtm
        summary.portf_final_MTM=last_daily_mtm

        summary.total_net_profit=profit_pct
        summary.total_net_profit_str = profit_pct_str
        summary.max_drawdown_on_MTM_str=max_drawdown_str
        summary.max_drawdown_on_MTM=max_drawdown_pct


        summary.portf_pos_summary=portf_positions_arr
        summary.max_cum_drawdowns=[max_drawdown_str]

        return summary

    def __same_side_pred__(self, pred,bias):
        return  pred==bias

    def __closing_divirgence__ (self,side,pred,bias):
        return  not (side==pred and pred==bias)

    def __find_classif_in_date__(self,indic_classif_df,date,classif_col):
        bias_record_index = (indic_classif_df['date_start'] <= date) & (indic_classif_df['date_end'] >= date)
        biased_record = indic_classif_df[bias_record_index]
        if biased_record.empty:
            return None
        else:
            return biased_record.iloc[0][classif_col]

    def __open_pos__(self,symbol,side,open_date,open_price):

        if  np.isnan(open_price) or open_price is None:
            raise Exception("Could not find a valid price for symbol {} on date {}".format(symbol,open_date))

        open_pos = PortfolioPosition(symbol)
        open_pos.open_pos(side, open_date, open_price)
        return open_pos

    def __close_pos__(self,curr_portf_pos,date,close_price):


        if  np.isnan(close_price) or close_price is None:
            raise Exception("Could not find a valid price for symbol {} on date {}".format(curr_portf_pos.symbol,date))

        curr_portf_pos.close_pos(date, close_price)

    #endregion
    
    #region Public Methods

    #NOTE:   Indicator based strategies consdier that the same price is used to close and open the next pos
    #This means. If closing a LONG position at 300.21, the system assumes that closing at 300.21 and goes SHORT at the same price
    def backtest_indicator_based_strategy(self,symbol,symbol_df,indic_classif_df,inv):
        last_pos=None
        portf_positions_arr=[]
        for index, row in indic_classif_df.iterrows():

            side = self.__get_side__(row, inv)
            if last_pos is None:
                last_pos = PortfolioPosition(symbol)
                open_date=row[IndicatorBasedTradingBacktester._INDIC_CLASSIF_START_DATE_COL]
                open_price = self.__get_date_price__(symbol_df, symbol, open_date)
                last_pos.open_pos(side, open_date, open_price)

            close_date=row[IndicatorBasedTradingBacktester._INDIC_CLASSIF_END_DATE_COL]
            close_price= self.__get_date_price__(symbol_df, symbol, close_date)
            last_pos.close_pos(close_date,close_price)
            portf_positions_arr.append(last_pos)

            side = self.__switch_side__(side)
            last_pos = PortfolioPosition(symbol)
            last_pos.open_pos(side, close_date, close_price)


        return self.calculate_portfolio_performance(symbol, portf_positions_arr)

    #tihs does not train anything Just uses the last trained indicator
    def backtest_ML_indicator_biased_strategy(self,symbol,series_df,indic_classif_df,inverted,predictions_dict):
        portf_positions_arr = {}

        for algo in predictions_dict.keys():

            curr_portf_pos = None
            portf_pos = []
            predictions_df = predictions_dict[algo]
            last_date=None
            last_price=None

            for index,day in predictions_df.iterrows():

                try:
                    date = day["date"]
                    last_date=date
                    ml_pred= self.__eval_invert_side__( day["Prediction"],inverted)
                    classif_pred=self.__find_classif_in_date__(indic_classif_df,date,"classification")
                    symbol_price= self.__get_date_price__(series_df, symbol, date)
                    last_price=symbol_price

                    if curr_portf_pos is None: #we are flat
                        if self.__same_side_pred__(ml_pred,classif_pred):
                            curr_portf_pos=self.__open_pos__(symbol,ml_pred,date,symbol_price)
                            portf_pos.append(curr_portf_pos)


                    elif curr_portf_pos is not None and self.__closing_divirgence__(curr_portf_pos.side,ml_pred,classif_pred):
                        self.__close_pos__(curr_portf_pos,date,symbol_price)
                        curr_portf_pos=None

                        if self.__same_side_pred__(ml_pred, classif_pred):#We see if we have to re open the pos
                            curr_portf_pos = self.__open_pos__(symbol, ml_pred, date, symbol_price)
                            portf_pos.append(curr_portf_pos)

                    else:
                        pass

                except Exception as e:
                    raise Exception("Error processing day {} for algo {}".format(day["date"], algo))

            if(curr_portf_pos is not None and curr_portf_pos.is_open()):
                self.__close_pos__(curr_portf_pos, last_date, last_price)

            portf_positions_arr[algo]= self.calculate_portfolio_performance(symbol, portf_pos)



        return  portf_positions_arr


    #endregion



