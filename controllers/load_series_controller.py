from datetime import date, datetime

import pandas as pd
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from starlette.responses import JSONResponse

from common.util.csv_reader import CSVReader
from common.util.file_writer import FileWriter
from framework.common.logger.message_type import MessageType
from logic_layer.algos_orchestation_logic import AlgosOrchestationLogic
from logic_layer.data_set_builder import DataSetBuilder

router = APIRouter()

class LoadSeriesController:
    def __init__(self,config_settings,logger):

        self.config_settings=config_settings
        self.logger=logger
        self.detailed_MTMS=[]

        # 📌 Create an APIRouter instead of FastAPI instance
        self.router = APIRouter()
        templates_path = Path(__file__).parent.parent / "templates"
        self.templates = Jinja2Templates(directory=templates_path)

        # 📌 Define Routes
        self.router.get("/", response_class=HTMLResponse)(self.display_page)
        self.router.post("/upload_series")(self.upload_series)
        self.router.get("/get_chart_data")(self.get_chart_data)

    async def display_page(self, request: Request):
        """Renders the custom ETF upload page."""
        return self.templates.TemplateResponse("load_series.html", {"request": request})

    from fastapi import UploadFile, File, HTTPException

    async def upload_series(self, file: UploadFile = File(...), series_key: str = Form(...)):
        """Handles the file upload."""
        try:
            if not file:
                raise HTTPException(status_code=400, detail="No file uploaded.")


            self.logger.do_log(f"✅ File received: {file.filename}",MessageType.INFO)

            ds_builder = DataSetBuilder(self.config_settings["hist_data_conn_str"],
                                        self.config_settings["ml_reports_conn_str"],
                                        None, self.logger)
            file_path = await FileWriter.dump_on_path(file)

            self.detailed_MTMS=ds_builder.save_time_series(file_path, series_key)

            self.logger.do_log( {"message": f"File '{file.filename}' uploaded successfully and persisted"},MessageType.INFO)

            return {"message": f"File '{file.filename}' uploaded successfully and persisted"}

        except Exception as e:
            import traceback
            error_message = f"❌ Error processing file: {str(e)}\n{traceback.format_exc()}"
            self.logger.do_log(error_message,MessageType.ERROR)  # Mostrar error detallado en la terminal
            raise HTTPException(status_code=500, detail=error_message)


    async def get_chart_data(self):
        """Returns Series data in JSON format for the chart. Handles empty dataset gracefully."""

        if not self.detailed_MTMS:
            return JSONResponse({
                "dates": [],
                "values": [],
                "message": "No Series data available. Please upload a file first."
            }, status_code=200)

        df = pd.DataFrame([{"date": obj.date, "MTM": obj.MTM} for obj in self.detailed_MTMS])

        if df.empty:
            return JSONResponse({
                "dates": [],
                "values": [],
                "message": "Series data is empty after processing. Check the uploaded file."
            }, status_code=200)

        df.sort_values(by="date", inplace=True)
        self.detailed_MTMS=[]#just one display
        return JSONResponse({
            "dates": df["date"].astype(str).tolist(),
            "values": df["MTM"].tolist()
        })

