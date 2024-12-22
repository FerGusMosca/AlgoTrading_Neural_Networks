from enum import Enum

class SlidingWindowStrategy(Enum):
    NONE = "NONE"
    CUT_INPUT_DF = "CUT_INPUT_DF"
    GET_FAKE_DATA = "GET_FAKE_DATA"