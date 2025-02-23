from datetime import date, datetime

import pandas as pd
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from starlette.responses import JSONResponse

from business_entities.detailed_MTM import DetailedMTM
from common.util.csv_reader import CSVReader
from common.util.file_writer import FileWriter
from controllers.base_controller import BaseController
from framework.common.logger.message_type import MessageType
from logic_layer.algos_orchestation_logic import AlgosOrchestationLogic
from logic_layer.data_set_builder import DataSetBuilder

router = APIRouter()


class DisplaySeriesController(BaseController):
    def __init__(self, config_settings, logger):
        super().__init__()
        self.config_settings = config_settings
        self.logger = logger
        self.detailed_MTMS = []

        # 📌 Create an APIRouter instead of FastAPI instance
        self.router = APIRouter()
        templates_path = Path(__file__).parent.parent / "templates"
        self.templates = Jinja2Templates(directory=templates_path)

        # 📌 Define Routes
        self.router.get("/", response_class=HTMLResponse)(self.display_page)
        self.router.post("/do_display")(self.do_display)
        self.router.get("/get_chart_data")(self.get_chart_data)

    async def display_page(self, request: Request):
        """Renders the custom ETF upload page."""
        return self.templates.TemplateResponse("display_series.html", {"request": request})

    from fastapi import UploadFile, File, HTTPException

    async def do_display(
            self,
            series_key: str = Form(...),
            start_date: str = Form(...),
            end_date: str = Form(...),
            time_interval: str = Form(...)
    ):
        """Handles the file upload."""
        try:

            ds_builder = DataSetBuilder(self.config_settings["hist_data_conn_str"],
                                        self.config_settings["ml_reports_conn_str"],
                                        None, self.logger)

            econ_val_arr = ds_builder.get_economic_values(series_key, start_date, end_date, time_interval)

            self.detailed_MTMS = []
            for econ_val in econ_val_arr:
                self.detailed_MTMS.append(DetailedMTM(econ_val.date, econ_val.open))

            return {"message": f"Series {series_key} successfully loaded"}

        except Exception as e:
            import traceback
            error_message = f"❌ Error processing series {series_key}: {str(e)}\n{traceback.format_exc()}"
            self.logger.do_log(error_message, MessageType.ERROR)  # Mostrar error detallado en la terminal
            raise HTTPException(status_code=500, detail=error_message)
