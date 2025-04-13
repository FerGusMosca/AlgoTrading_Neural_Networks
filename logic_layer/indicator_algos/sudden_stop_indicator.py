import pandas as pd
from dateutil.relativedelta import relativedelta


class SuddenStopIndicator():

    def __init__(self,indicator,units,eval_period,blackout_period):
        self.reset_counter=0
        self.units=units
        self.indicator=indicator
        self.eval_period=eval_period
        self.blackout_period=blackout_period
        self.sudden_stop=False
        self.date=None

    def has_declined(self, indicator_series_df):
        """
        Check if indicator has dropped by self.units over the last self.eval_period months
        ending at self.current_date.

        Returns True if the condition is met, False otherwise.
        """

        indicator_col = f"close_{self.indicator}"

        # ✅ Parse current date to datetime
        current_date = pd.to_datetime(self.date)

        # ✅ Filter rows where indicator has non-null values
        df = indicator_series_df[["date", indicator_col]].dropna()

        # ✅ Ensure date is datetime and sort ascending
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # ✅ Get current value for the indicator at current_date
        current_row = df[df["date"] == current_date]
        if current_row.empty:
            print(f"No data found for date {self.date} in column {indicator_col}")
            return False

        current_value = current_row[indicator_col].values[0]

        # ✅ Define lookback window
        start_date = current_date - relativedelta(months=self.eval_period)

        # ✅ Subset rows within the evaluation period
        window_df = df[(df["date"] >= start_date) & (df["date"] < current_date)]

        if window_df.empty:
            print("No historical data in evaluation window.")
            return  False

        # ✅ Check if any value in the period was >= current + units (i.e. current_value + drop)
        threshold = current_value + self.units
        sudden_stop=any(window_df[indicator_col] >= threshold)

        if sudden_stop:
            print(f"SUDDEN STOP scenario for indicator {self.indicator} on {self.date}: curr_val={current_date}")


        print(f"[EVAL] {current_date} → sudden_stop={self.sudden_stop}, reset_counter={self.reset_counter}")

        return sudden_stop

    def eval_sudden_stop(self, date, indicator_series_df):
        self.date = date

        # ✅ Always check has_declined unless we are inside blackout
        if self.sudden_stop:
            if self.reset_counter >= self.blackout_period:
                self.reset_counter = 0
                self.sudden_stop = False
            else:
                self.reset_counter += 1
                return True  # still in blackout, status is ON
        # 🔁 if not in blackout or just reset, check again
        if not self.sudden_stop:
            print(f"[CHECK] Calling has_declined for {self.date}")
            self.sudden_stop = self.has_declined(indicator_series_df)
            print(f"[CHECK] has_declined returned: {self.sudden_stop}")

            self.reset_counter = 0 if self.sudden_stop else self.reset_counter

        print(f"[POST-EVAL] {self.date} final sudden_stop={self.sudden_stop}, reset_counter={self.reset_counter}")
        return self.sudden_stop
