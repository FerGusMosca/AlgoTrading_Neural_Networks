from common.enums.side import Side
from logic_layer.trading_algos.direct_prediction_backtester import DirectPredictionBacktester
from business_entities.portf_position import PortfolioPosition
from common.enums.sliding_window_strategy import SlidingWindowStrategy
from common.util.light_logger import LightLogger
from common.util.date_handler import DateHandler
from enum import Enum

from logic_layer.trading_algos.slope_backtester import SlopeBacktester


class NFlipPredictionBacktester(SlopeBacktester):

    def backtest(self, symbol, symbol_prices_df, predictions_dic, last_trading_dict, n_algo_param_dict,
                 init_last_portf_size_dict=None):

        portf_pos_dict = {}

        n_flip = n_algo_param_dict.get("n_flip", 1)
        bias = n_algo_param_dict.get("bias", SlidingWindowStrategy.NONE.value)

        for algo, predictions_df in predictions_dic.items():
            curr_portf_pos = None
            last_side = None
            portf_positions = []
            LightLogger.do_log("----Processing algo {}".format(algo))

            last_portf_size = (
                n_algo_param_dict["init_portf_size"]
                if init_last_portf_size_dict is None
                else init_last_portf_size_dict[algo]
            )
            net_commissions = 0
            current_date = None
            current_price = None

            flip_counter = {Side.LONG.value: 0, Side.SHORT.value: 0}

            for index, day in predictions_df.iterrows():
                if not self.__eval_exists_value_on_df__(symbol_prices_df, "date", day["date"], symbol):
                    continue

                current_date = day[self._DATE_COL]
                current_pred = day["Prediction"]
                current_price = self.__extract_value_from_row__(symbol_prices_df, "date", current_date, symbol)

                flip_counter[current_pred] += 1
                opposite_side = Side.SHORT.value if current_pred == Side.LONG.value else Side.LONG.value
                flip_counter[opposite_side] = 0

                try:
                    if curr_portf_pos is None and flip_counter[current_pred] >= n_flip:
                        if self.__validate_bias__(current_pred, bias):
                            current_price = self.__eval_reuse_reference_price__(algo, last_trading_dict,
                                                                                current_pred, current_date, current_price)
                            LightLogger.do_log(f"-Opening {current_pred} pos for ref_price= {current_price} on {current_date}")
                            curr_portf_pos = PortfolioPosition(symbol)
                            new_portf_size, net_commissions = self.__extract_commission__(last_portf_size, n_algo_param_dict)
                            curr_portf_pos.open_pos(current_pred, current_date, current_price,
                                                    units=new_portf_size / current_price)
                            last_side = current_pred

                    elif curr_portf_pos is not None and current_pred != last_side:
                        # Only close if the opposite signal appears at least n_flip times
                        if flip_counter[current_pred] >= n_flip:
                            final_MTM = curr_portf_pos.calculate_and_append_MTM(current_date, current_price)
                            last_portf_size=self.__apply_commisions__(final_MTM,net_commissions)
                            curr_portf_pos.close_pos(current_date, current_price)
                            curr_portf_pos.append_MTM(current_date,last_portf_size)
                            LightLogger.do_log(f"-Closing {last_side} pos for ref_price= {current_price} on {current_date} "
                                               f"for pct profit={curr_portf_pos.calculate_pct_profit()}% "
                                               f"(nom. profit={curr_portf_pos.calculate_th_nom_profit()})")
                            portf_positions.append(curr_portf_pos)

                            if self.__validate_bias__(current_pred, bias):
                                LightLogger.do_log(f"-Opening new {current_pred} pos for ref_price= {current_price} on {current_date}")
                                curr_portf_pos = PortfolioPosition(symbol)
                                new_portf_size, net_commissions = self.__extract_commission__(last_portf_size, n_algo_param_dict)
                                curr_portf_pos.open_pos(current_pred, current_date, current_price,
                                                        units=new_portf_size / current_price)
                                last_side = current_pred
                            else:
                                curr_portf_pos = None
                                last_side = None
                        else:
                            curr_portf_pos.calculate_and_append_MTM(current_date, current_price)

                    elif curr_portf_pos is not None:
                        curr_portf_pos.calculate_and_append_MTM(current_date, current_price)

                except Exception as e:
                    raise Exception(f"Error processing day {current_date} for algo {algo}:{str(e)}")

            if curr_portf_pos is not None:
                final_MTM = curr_portf_pos.calculate_and_append_MTM(current_date, current_price)
                last_portf_size=self.__apply_commisions__(final_MTM,net_commissions)
                curr_portf_pos.close_pos(current_date, current_price)
                curr_portf_pos.append_MTM(current_date,last_portf_size)
                portf_positions.append(curr_portf_pos)

                LightLogger.do_log(f"-Closing last {curr_portf_pos.side} pos for ref_price= {current_price} on {current_date} "
                                   f"for pct profit={curr_portf_pos.calculate_pct_profit()}% "
                                   f"(nom. profit={curr_portf_pos.calculate_th_nom_profit()})")

            portf_pos_dict[algo] = portf_positions

        return portf_pos_dict
