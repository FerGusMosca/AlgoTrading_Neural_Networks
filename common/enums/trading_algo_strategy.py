from enum import Enum

class TradingAlgoStrategy(Enum):
    RAW_ALGO = "RAW_ALGO"
    N_MIN_BUFFER_W_FLIP = "N_MIN_BUFFER_W_FLIP"
    ONLY_SIGNAL_N_MIN_PLUS_MOV_AVG = "ONLY_SIGNAL_N_MIN_PLUS_MOV_AVG"