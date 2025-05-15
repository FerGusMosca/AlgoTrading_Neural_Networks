from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Body
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from tvDatafeed import TvDatafeedLive, Interval
from typing import List
from framework.common.logger.message_type import MessageType

class GlobalM2IndicatorController:
    def __init__(self, config_settings, logger):
        self.config_settings = config_settings
        self.logger = logger
        self.router = APIRouter()
        self.templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

        self.tv = TvDatafeedLive("alien.zimzum@gmail.com", "VyU1062V")

        self.COUNTRIES = {
            "USM2": "USD", "CNM2": "CNY", "JPM2": "JPY", "INM2": "INR",
            "EUM2": "EUR", "GBM2": "GBP", "BRM2": "BRL", "MXM2": "MXN"
        }

        self.router.get("/", response_class=HTMLResponse)(self.display_page)
        self.router.post("/calculate")(self.calculate_indicator)

    def get_m2_global_data(self, start_date: str, end_date: str, offset_days: int = 0, currencies: list[str] = None) -> \
    tuple[pd.DataFrame, list[str]]:
        all_data = []
        log_msgs = []

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        self.logger.do_log(f"Fetching global M2 data from {start_dt} to {end_dt} (offset {offset_days} days)",
                           MessageType.INFO)

        filtered_countries = {m2: cur for m2, cur in self.COUNTRIES.items() if
                              cur in currencies} if currencies else self.COUNTRIES

        for m2_symbol, currency_code in filtered_countries.items():
            try:
                self.logger.do_log(f"Fetching M2 for {m2_symbol}", MessageType.INFO)

                df_m2 = self.tv.get_hist(
                    symbol=m2_symbol,
                    exchange='ECONOMICS',
                    interval=Interval.in_monthly,
                    n_bars=200
                )

                df_fx = self.tv.get_hist(
                    symbol=f"{currency_code}USD",
                    exchange='FX_IDC',
                    interval=Interval.in_monthly,
                    n_bars=200
                )

                if df_m2 is None or df_fx is None or df_m2.empty or df_fx.empty:
                    self.logger.do_log(f"Skipping {m2_symbol} due to missing data", MessageType.WARNING)
                    continue

                df_m2.index = pd.to_datetime(df_m2.index)
                df_fx.index = pd.to_datetime(df_fx.index)

                df_m2 = df_m2[['close']].rename(columns={'close': 'M2'})
                df_m2['Date'] = df_m2.index.normalize()

                df_fx = df_fx[['close']].rename(columns={'close': 'FX'})
                df_fx['Date'] = df_fx.index.normalize()

                df = df_m2.merge(df_fx, on='Date', how='inner')
                df.dropna(inplace=True)
                df['M2_USD'] = df['M2'] * df['FX']

                df['Date'] = df['Date'] + pd.Timedelta(days=offset_days)
                start_dt_offset = start_dt + pd.Timedelta(days=offset_days)
                end_dt_offset = end_dt + pd.Timedelta(days=offset_days)

                log_msgs.append(
                    f"✅ Downloaded {currency_code} M2: found data from {df['Date'].min().date()} to {df['Date'].max().date()}")

                all_data.append(df[['Date', 'M2_USD']])

            except Exception as e:
                self.logger.do_log(f"⚠️ Error fetching data for {m2_symbol}: {e}", MessageType.ERROR)
                continue

        if not all_data:
            self.logger.do_log("No data retrieved for any country in the given range", MessageType.WARNING)
            return pd.DataFrame(), log_msgs

        df_all = pd.concat(all_data)

        self.logger.do_log(f"Final df_all head: {df_all.head()}", MessageType.INFO)
        self.logger.do_log(f"Min date: {df_all['Date'].min()}, Max date: {df_all['Date'].max()}", MessageType.INFO)
        self.logger.do_log(f"Filtering from {start_dt_offset} to {end_dt_offset}", MessageType.INFO)

        df_all = df_all[(df_all["Date"] >= start_dt_offset) & (df_all["Date"] <= end_dt_offset)]
        df_global = df_all.groupby("Date").sum().reset_index()
        df_global.rename(columns={"M2_USD": "Global_M2"}, inplace=True)

        return df_global, log_msgs

    async def display_page(self, request: Request):
        return self.templates.TemplateResponse("global_m2_indicator.html", {
            "request": request,
            "available_currencies": list(self.COUNTRIES.values())
        })

    async def calculate_indicator(
            self,
            request: Request,
            start_date: str = Form(...),
            end_date: str = Form(...),
            offset_days: int = Form(0),
            currencies: list[str] = Form(...)
    ):
        try:
            self.logger.do_log(
                f"Received request to calculate Global M2 from {start_date} to {end_date} with offset {offset_days} for currencies {currencies}",
                MessageType.INFO)

            df_m2, log_msgs = self.get_m2_global_data(start_date, end_date, offset_days, currencies)

            self.logger.do_log(f"Final Global M2 dataframe:\n{df_m2}", MessageType.INFO)

            if df_m2.empty:
                return JSONResponse(content={
                    "message": "No data available in this date range.",
                    "logs": log_msgs
                }, status_code=200)

            return JSONResponse(content={
                "dates": df_m2["Date"].dt.strftime('%Y-%m-%d').tolist(),
                "values": df_m2["Global_M2"].round(2).tolist(),
                "logs": log_msgs
            }, status_code=200)

        except Exception as e:
            self.logger.do_log(f"Error calculating Global M2: {str(e)}", MessageType.ERROR)
            raise HTTPException(status_code=500, detail="Failed to calculate indicator.")

