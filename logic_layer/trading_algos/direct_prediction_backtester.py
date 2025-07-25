from business_entities.portf_position import PortfolioPosition
from common.util.logging.light_logger import LightLogger
from logic_layer.trading_algos.slope_backtester import SlopeBacktester


class DirectPredictionBacktester(SlopeBacktester):

    def __init__(self):
        pass

    def backtest(self, symbol, symbol_prices_df,predictions_dic,bias,last_trading_dict, n_algo_param_dict,
                 init_last_portf_size_dict=None):
        portf_pos_dict = {}

        for algo in predictions_dic.keys():
            curr_portf_pos = None
            last_side = None
            portf_positions = []
            predictions_df = predictions_dic[algo]
            LightLogger.do_log("----Processing algo {}".format(algo))
            last_portf_size = n_algo_param_dict["init_portf_size"] if init_last_portf_size_dict is None else init_last_portf_size_dict[algo]

            net_commissions = 0
            current_date=None
            current_price=None

            for index, day in predictions_df.iterrows():
                if not self.__eval_exists_value_on_df__(symbol_prices_df, "date", day["date"], symbol):
                    continue  # We ignore days when we have no prices

                current_date = day[SlopeBacktester._DATE_COL]
                current_pred = day["Prediction"]
                current_price = self.__extract_value_from_row__(symbol_prices_df, "date", current_date, symbol)

                try:

                    if curr_portf_pos is None and last_side is None:
                        if self.__validate_bias__(current_pred, bias):
                            current_price = self.__eval_reuse_reference_price__(algo, last_trading_dict,
                                                                                current_pred,current_date,
                                                                                current_price)
                            LightLogger.do_log("-Opening {} pos for ref_price= {} on {}".format(current_pred,current_price,current_date))
                            curr_portf_pos = PortfolioPosition(symbol)
                            new_portf_size, net_commissions = self.__extract_commission__(last_portf_size,n_algo_param_dict)
                            curr_portf_pos.open_pos(current_pred, current_date, current_price,units=new_portf_size/current_price)
                            last_side = current_pred

                    elif last_side != current_pred:  # change the side
                        # 1- Close the old position
                        final_MTM = curr_portf_pos.calculate_and_append_MTM(current_date, current_price)
                        last_portf_size = final_MTM - net_commissions
                        curr_portf_pos.close_pos(current_date, current_price)
                        LightLogger.do_log("-Closing {} pos for ref_price= {} on {} for pct profit={}% (nom. profit={})".format(curr_portf_pos.side, current_price, current_date,curr_portf_pos.calculate_pct_profit(), curr_portf_pos.calculate_th_nom_profit()))
                        portf_positions.append(curr_portf_pos)

                        # 2- Open the new one?
                        if self.__validate_bias__(current_pred, bias):

                            if curr_portf_pos is not None:
                                # 2- Open the new one
                                LightLogger.do_log("-Opening new {} pos for ref_price= {} on {}".format(current_pred,current_price,current_date))
                                curr_portf_pos = PortfolioPosition(symbol)
                                new_portf_size, net_commissions = self.__extract_commission__(last_portf_size,n_algo_param_dict)
                                curr_portf_pos.open_pos(day["Prediction"], current_date, current_price,units=new_portf_size/current_price)
                                last_side = current_pred
                        else:  # 3-We go flat
                            curr_portf_pos = None
                            last_side = None

                    else: #open w/no change
                        curr_portf_pos.calculate_and_append_MTM(current_date, current_price)

                except Exception as e:
                    raise Exception("Error processing day {} for algo {}:{}".format(current_date, algo, str(e)))

            # We add the last position
            if curr_portf_pos is not None:
                final_MTM = curr_portf_pos.calculate_and_append_MTM(current_date, current_price)
                last_portf_size = final_MTM - net_commissions
                curr_portf_pos.close_pos(current_date, current_price)
                portf_positions.append(curr_portf_pos)

                LightLogger.do_log("-Closing last {} pos for ref_price= {} on {}  for pct profit={}% (nom. profit={})".format(curr_portf_pos.side, current_price, current_date,curr_portf_pos.calculate_pct_profit(), curr_portf_pos.calculate_th_nom_profit()))

            portf_pos_dict[algo] = portf_positions

        return portf_pos_dict