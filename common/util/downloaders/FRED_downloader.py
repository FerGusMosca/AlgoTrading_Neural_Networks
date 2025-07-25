import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime

class FredDownloader:
    def __init__(self, algo_params: dict):
        self.api_key = algo_params.get("api_key")
        if not self.api_key:
            raise Exception("❌ Missing FRED API key in vendor_params['api_key']")
        self.params = algo_params
        print("✅ FRED Downloader initialized")

    def download(self, symbol: str, from_date=None, to_date=None) -> pd.DataFrame:
        print(f"⬇️ Downloading {symbol} from FRED")

        # Default date range if not provided
        from_date = pd.to_datetime(from_date) if from_date else pd.to_datetime("1900-01-01")
        to_date = pd.to_datetime(to_date) if to_date else pd.Timestamp.today()

        # Load data
        df = pdr.DataReader(symbol, "fred", from_date, to_date, api_key=self.api_key)

        # Ensure consistent formatting
        df = df.rename(columns={symbol: "value"})
        df.index.name = "date"
        df = df.reset_index()

        print(f"📊 {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()}")
        return df
