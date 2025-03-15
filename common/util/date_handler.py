from datetime import datetime


class DateHandler:

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