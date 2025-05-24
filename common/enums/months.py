from enum import Enum


class Months(Enum):
    JAN = (1, "Jan")
    FEB = (2, "Feb")
    MAR = (3, "Mar")
    APR = (4, "Apr")
    MAY = (5, "May")
    JUN = (6, "Jun")
    JUL = (7, "Jul")
    AUG = (8, "Aug")
    SEP = (9, "Sep")
    OCT = (10, "Oct")
    NOV = (11, "Nov")
    DEC = (12, "Dec")

    def __init__(self, number, label):
        self.number = number
        self.label = label

    @staticmethod
    def from_number(number: int) -> "Months":
        for month in Months:
            if month.value[0] == number:
                return month
        raise ValueError(f"No month found for number {number}")

    @staticmethod
    def label_from_number(number: int) -> str:
        return Months.from_number(number).label
