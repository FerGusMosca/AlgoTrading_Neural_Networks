from tvDatafeed import TvDatafeed, Interval
import pandas as pd
from datetime import datetime

from common.enums.intervals import Intervals


class TradingViewDownloader:
    def __init__(self, algo_params: dict):
        self.params = algo_params
        self.tv = self._connect_to_tradingview()
        self.interval_str = self.params.get("interval", "1d")
        self.exchange=self.params.get("exchange")

        self.interval_map={
            "1m": Interval.in_1_minute,
            "5m": Interval.in_5_minute,
            "15m": Interval.in_15_minute,
            "30m": Interval.in_30_minute,
            "1h": Interval.in_1_hour,
            "1d": Interval.in_daily,
            "1W": Interval.in_weekly,
            "1M": Interval.in_monthly
        }

        #This is the apps interval , so we can translate the tvData enums w/this app Intervals enum
        self.interval_translation_map=    {
            "1m": Intervals.MIN_1,
            "5m": Intervals.MIN_5,
            "15m": Intervals.MIN_15,
            "30m": Intervals.MIN_30,
            "1h": Intervals.HOUR_1,
            "1d": Intervals.DAY,
            "1W": Intervals.WEEK,
            "1M": Intervals.MONTH,
        }

    def _connect_to_tradingview(self):
        if "session" in self.params and "token" in self.params:
            print("✅ Connected to TradingView via session/token")
            return TvDatafeed(session=self.params["session"], token=self.params["token"])
        elif "session" in self.params:
            print("✅ Connected to TradingView via session/token")
            return TvDatafeed(session=self.params["session"])
        elif "tradingview_user" in self.params and "tradingview_pwd" in self.params:
            print("✅ Connected to TradingView via username/password")
            return TvDatafeed(username=self.params["tradingview_user"], password=self.params["tradingview_pwd"])
        else:
            raise Exception("❌ Missing credentials: must provide either (session+token) or (username+password)")

    def get_interval_enum_translation(self) -> Intervals:

        interval_enum = self.interval_translation_map.get(self.interval_str)
        if not interval_enum:
            raise Exception(f"❌ Unsupported interval for persistence: {self.interval_str}")
        return interval_enum

    def download(self, symbol: str, from_date=None, to_date=None) -> pd.DataFrame:



        interval = self.interval_map.get(self.interval_str)
        if not interval:
            raise Exception(f"❌ Invalid interval: {self.interval_str}")

        print(f"⬇️ Downloading {symbol} ({self.exchange}) @ {self.interval_str}")
        df = self.tv.get_hist(symbol=symbol, exchange=self.exchange, interval=interval, n_bars=5000)

        if df is None:
            raise Exception(f"❌ No data returned for {symbol} ({self.exchange}) @ {self.interval_str}")

        if from_date:
            df = df[df.index >= pd.to_datetime(from_date)]
        if to_date:
            df = df[df.index <= pd.to_datetime(to_date)]

        print(f"📊 {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")
        return df


