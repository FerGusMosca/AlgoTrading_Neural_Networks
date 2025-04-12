from datetime import date, datetime

import pandas as pd
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from starlette.responses import JSONResponse

from common.util.csv_reader import CSVReader
from controllers.base_controller import BaseController
from framework.common.logger.message_type import MessageType
from logic_layer.algos_orchestation_logic import AlgosOrchestationLogic
router = APIRouter()

class DisplayCustomETFController(BaseController):
    def __init__(self,config_settings,logger):
        super().__init__()
        self.config_settings=config_settings
        self.logger=logger

        # 📌 Create an APIRouter instead of FastAPI instance
        self.router = APIRouter()
        templates_path = Path(__file__).parent.parent / "templates"
        self.templates = Jinja2Templates(directory=templates_path)

        # 📌 Define Routes
        self.router.get("/", response_class=HTMLResponse)(self.display_page)
        self.router.post("/upload_custom_etf")(self.upload_custom_etf)
        self.router.get("/get_chart_data")(self.get_chart_data_w_mmov)

    async def display_page(self, request: Request):
        """Renders the custom ETF upload page."""
        return self.templates.TemplateResponse("display_custom_etf.html", {"request": request})

    from fastapi import UploadFile, File, HTTPException


    def calc_mov_avgs(self,moving_avg):
        # ✅ Compute moving average if requested by user
        if moving_avg and moving_avg.strip().isdigit():
            window = int(moving_avg)
            if window <= 0:
                raise ValueError("Moving average must be a positive integer")

            # ✅ Get all MTM values (including None) to keep proper alignment
            values = [item.MTM for item in self.detailed_MTMS]

            # ✅ Create a pandas Series, keeping original index alignment
            mtm_series = pd.Series(values)

            # ✅ Calculate moving average, which will naturally produce NaNs for the first (window - 1) points
            ma_series = mtm_series.rolling(window=window).mean()

            # ✅ Convert result to a list of float/None values (for JSON serialization)
            self.moving_avg_values = [None if pd.isna(val) else float(val) for val in ma_series.tolist()]
        else:
            # No moving average requested or invalid input
            self.moving_avg_values = []

    async def upload_custom_etf(self,
                                file: UploadFile = File(...),
                                start_date: str = Form(...),
                                end_date: str = Form(...),
                                moving_avg: str = Form(None)):
        """Handles the file upload."""
        try:
            if not file:
                raise HTTPException(status_code=400, detail="No file uploaded.")


            self.logger.do_log(f"✅ File received: {file.filename}",MessageType.INFO)

            aol = AlgosOrchestationLogic(self.config_settings["hist_data_conn_str"],
                                         self.config_settings["ml_reports_conn_str"],
                                         None,self.logger)

            content = await file.read()
            file_content = content.decode("utf-8").splitlines()
            weights_csv = await CSVReader.extract_col_csv_from_content(file_content, 0)
            symbols_csv = await CSVReader.extract_col_csv_from_content(file_content, 1)
            dstart_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            dend_date = datetime.strptime(end_date, "%Y-%m-%d").date()
            self.detailed_MTMS= aol.model_custom_etf(weights_csv,symbols_csv,dstart_date,dend_date)

            self.calc_mov_avgs(moving_avg)

            self.logger.do_log( {"message": f"File '{file.filename}' uploaded successfully"},MessageType.INFO)

            return {"message": f"File '{file.filename}' uploaded successfully"}

        except Exception as e:
            import traceback
            error_message = f"❌ Error processing file: {str(e)}\n{traceback.format_exc()}"
            self.logger.do_log(error_message,MessageType.ERROR)  # Mostrar error detallado en la terminal
            raise HTTPException(status_code=500, detail=error_message)


