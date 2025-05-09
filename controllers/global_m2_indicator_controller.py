# controllers/global_m2_indicator_controller.py
from datetime import datetime, timedelta

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pandas as pd
from tvDatafeed import TvDatafeedLive, Interval

from framework.common.logger.message_type import MessageType

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

class GlobalM2IndicatorController:
    def __init__(self, config_settings, logger):
        self.config_settings = config_settings
        self.logger = logger
        self.router = APIRouter()
        self.templates = templates

        self.tradingview_user="alien.zimzum@gmail.com"
        self.tradingview_pwd="VyU1062V"

        # TV login (can be skipped if you're already authenticated)
        self.tv = TvDatafeedLive(self.tradingview_user,self.tradingview_pwd)   # Optionally: TvDatafeed(username="your_user", password="your_pass")

        # Define supported countries and currency symbols
        self.COUNTRIES = {
            "USM2": "USD", "CNM2": "CNY", "JPM2": "JPY", "INM2": "INR",
            "EUM2": "EUR", "GBM2": "GBP", "BRM2": "BRL", "MXM2": "MXN"
        }

        # Define routes
        self.router.get("/", response_class=HTMLResponse)(self.display_page)
        self.router.post("/calculate")(self.calculate_indicator)

    def get_m2_global_data(self, start_date: str, end_date: str, offset_days: int = 0) -> pd.DataFrame:
        all_data = []

        for m2_symbol, currency_code in self.COUNTRIES.items():
            try:
                self.logger.do_log(f"Fetching M2 for {m2_symbol}", MessageType.INFO)

                df_m2 = self.tv.get_hist(
                    symbol=m2_symbol,
                    exchange='ECONOMICS',
                    interval=Interval.in_monthly,
                    n_bars=500
                )

                df_fx = self.tv.get_hist(
                    symbol=f"{currency_code}USD",
                    exchange='FX_IDC',
                    interval=Interval.in_monthly,
                    n_bars=500
                )

                if df_m2 is None or df_fx is None or df_m2.empty or df_fx.empty:
                    self.logger.do_log(f"Skipping {m2_symbol} due to missing data", MessageType.WARNING)
                    continue

                # Normalización y renombre
                df_m2.index = pd.to_datetime(df_m2.index)
                df_fx.index = pd.to_datetime(df_fx.index)

                df_m2 = df_m2[['close']].rename(columns={'close': 'M2'})
                df_m2['Date'] = df_m2.index.normalize()

                df_fx = df_fx[['close']].rename(columns={'close': 'FX'})
                df_fx['Date'] = df_fx.index.normalize()

                # Merge y cálculo
                df = df_m2.merge(df_fx, on='Date', how='inner')
                df.dropna(inplace=True)
                df['M2_USD'] = df['M2'] * df['FX']

                # Aplicar offset temporal
                df['Date'] = df['Date'] + timedelta(days=offset_days)

                all_data.append(df[['Date', 'M2_USD']])

            except Exception as e:
                self.logger.warndo_loging(f"⚠️ Error fetching data for {m2_symbol}: {e}", MessageType.INFO)
                continue

        if not all_data:
            return pd.DataFrame()

        df_all = pd.concat(all_data)

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        self.logger.do_log(f"Final df_all head: {df_all.head()}", MessageType.INFO)
        self.logger.do_log(f"Min date: {df_all['Date'].min()}, Max date: {df_all['Date'].max()}", MessageType.INFO)
        self.logger.do_log(f"Filtering from {start_dt} to {end_dt}", MessageType.INFO)

        df_all = df_all[(df_all["Date"] >= start_dt) & (df_all["Date"] <= end_dt)]
        df_global = df_all.groupby("Date").sum().reset_index()
        df_global.rename(columns={"M2_USD": "Global_M2"}, inplace=True)

        return df_global

    async def display_page(self, request: Request):
        return self.templates.TemplateResponse("global_m2_indicator.html", {"request": request})

    async def calculate_indicator(
            self,
            request: Request,
            start_date: str = Form(...),
            end_date: str = Form(...),
            offset_days: str = Form("0")
    ):
        try:
            offset_int = int(offset_days)
            df_m2 = self.get_m2_global_data(start_date, end_date, offset_int)

            self.logger.do_log(f"Final Global M2 dataframe:\n{df_m2}", MessageType.INFO)

            if df_m2.empty:
                return JSONResponse(content={"message": "No data available in this date range."}, status_code=200)

            return JSONResponse(content={
                "dates": df_m2["Date"].dt.strftime('%Y-%m-%d').tolist(),
                "values": df_m2["Global_M2"].round(2).tolist()
            }, status_code=200)
        except Exception as e:
            self.logger.do_log(f"Error calculating Global M2: {str(e)}", MessageType.ERROR)
            raise HTTPException(status_code=500, detail="Failed to calculate indicator.")