from enum import Enum

class GroupCriteria(Enum):
    GO_FIRST = "GO_FIRST"
    GO_HIGHEST_COUNT = "GO_HIGHEST_COUNT"
    GO_FLAT_ON_DIFF = "GO_FLAT_ON_DIFF"