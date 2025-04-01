from datetime import date, datetime

import pandas as pd
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from starlette.responses import JSONResponse

from business_entities.detailed_MTM import DetailedMTM
from common.util.csv_reader import CSVReader
from common.util.date_handler import DateHandler
from common.util.file_writer import FileWriter
from common.util.slope_calculator import SlopeCalculator
from controllers.base_controller import BaseController
from framework.common.logger.message_type import MessageType
from logic_layer.ARIMA_models_analyzer import ARIMAModelsAnalyzer
from logic_layer.algos_orchestation_logic import AlgosOrchestationLogic
from logic_layer.data_set_builder import DataSetBuilder
from scipy.stats import linregress
import numpy as np
router = APIRouter()


class DisplaySeriesController(BaseController):
    def __init__(self, config_settings, logger):
        super().__init__()
        self.config_settings = config_settings
        self.logger = logger
        self.last_series_key=None
        self.last_from=None
        self.last_to=None
        self.last_interval=None

        # 📌 Create an APIRouter instead of FastAPI instance
        self.router = APIRouter()
        templates_path = Path(__file__).parent.parent / "templates"
        self.templates = Jinja2Templates(directory=templates_path)

        # 📌 Define Routes
        self.router.get("/", response_class=HTMLResponse)(self.display_page)
        self.router.post("/do_display")(self.do_display)
        self.router.get("/get_chart_data")(self.get_chart_data)
        self.router.post("/add_data")(self.add_data)
        self.router.post("/calculate_new_slope")(self.calculate_new_slope)
        self.router.post("/calculate_arima")(self.calculate_arima)

    async def add_data(self,date: str = Form(...), value: float = Form(...)):
        """
        Adds a new data point to the time series.

        Parameters:
            date (str): The date of the new data point (YYYY-MM-DD format).
            value (float): The value of the new data point.

        Returns:
            JSONResponse: A response indicating success or failure.
        """

        try:
            if self.last_interval is None or self.last_series_key is None:
                raise Exception(f"You must first load the series to be displayed")

            new_data = DetailedMTM(date, value)  # Create a new data object
            self.detailed_MTMS.append(new_data)  # Simulated insertion into the dataset

            ds_builder = DataSetBuilder(self.config_settings["hist_data_conn_str"],
                                        self.config_settings["ml_reports_conn_str"],
                                        None, self.logger)
            ds_builder.save_time_series_value(self.last_series_key,date,value,self.last_interval)

            return JSONResponse(content={"message": "Data point added successfully. Press Display to Refresh"}, status_code=200)

        except Exception as e:
            # Return an error response if something goes wrong
            return JSONResponse(content={"message": f"Error adding data: {str(e)}"}, status_code=500)

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

            self.last_series_key=series_key
            self.last_from=start_date
            self.last_to=end_date
            self.last_interval=time_interval

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

    async def calculate_new_slope(self, slope_units: int = Form(...), new_value: float = Form(None)):
        """
        Calculates the regression slope for a given number of data points.

        Parameters:
            slope_units (int): The number of recent points to use for slope calculation.
            new_value (float, optional): A hypothetical new value to include in the slope calculation.

        Returns:
            JSONResponse: The calculated slope or an error message.
        """
        try:
            if not self.detailed_MTMS or len(self.detailed_MTMS) < slope_units:
                return JSONResponse(content={"message": "Not enough data points for regression."}, status_code=400)

            # Get last 'slope_units' values
            recent_data = self.detailed_MTMS[-slope_units:]
            values = np.array([item.MTM for item in recent_data])

            if new_value is not None:
                values = np.append(values[1:], new_value)  # Replace first value with new_value

            slope=SlopeCalculator.calculate_slope(values)

            return JSONResponse(content={"slope": round(slope, 5)}, status_code=200)

        except Exception as e:
            return JSONResponse(content={"message": f"Error calculating slope: {str(e)}"}, status_code=500)

    async def calculate_arima(self, p: int = Form(...), d: int = Form(...), q: int = Form(...),
                              s: int = Form(None), forecast_periods: int = Form(...),
                              new_value: float = Form(None)):
        try:
            cmd_param_dict = {}

            if p is not None:
                cmd_param_dict["p"] = p

            if d is not None:
                cmd_param_dict["d"] = d

            if q is not None:
                cmd_param_dict["q"] = q

            if s is not None:
                cmd_param_dict["s"] = s
            else:
                cmd_param_dict["s"]=None

            cmd_param_dict["period"] = None
            cmd_param_dict["step"] = forecast_periods
            cmd_param_dict["inv_steps"] = forecast_periods
            d_from=DateHandler.convert_str_date(self.last_from, "%Y-%m-%d")
            d_to = DateHandler.convert_str_date(self.last_to, "%Y-%m-%d")

            dataMgm = AlgosOrchestationLogic(self.config_settings["hist_data_conn_str"],
                                             self.config_settings["ml_reports_conn_str"],
                                             self.config_settings["classification_map_key"], self.logger)
            pred_dict = dataMgm.process_ARIMA_predictions(self.last_series_key, d_from, d_to, cmd_param_dict)


            #TODO eval what to do with this
            return None
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calculating ARIMA: {str(e)}")

