from datetime import datetime


class DateHandler:


    @staticmethod
    def get_two_month_period_from_date(date):

        month = date.month
        if 1 <= month <= 2:
            period = "Jan-Feb"
        elif 3 <= month <= 4:
            period = "Mar-Apr"
        elif 5 <= month <= 6:
            period = "May-Jun"
        elif 7 <= month <= 8:
            period = "Jul-Aug"
        elif 9 <= month <= 10:
            period = "Sep-Oct"
        else:
            period = "Nov-Dec"

        return  period

    @staticmethod
    def evaluate_consecutive_days(day_1, day_2):

        if abs((day_2 - day_1).days) == 1:
            return True
        else:
            if (abs((day_2 - day_1).days) ==3) and day_1.weekday()==4 and day_2.weekday()==0 : #Friday To Monday
                return  True
            else:
                return  False


    @staticmethod
    def convert_str_date(date,format):
        try:
            converted_date = datetime.strptime(date, format)
            return  converted_date
        except Exception as e:
            raise Exception("Could not convert date {} to datetime w/format {}".format(date,format))