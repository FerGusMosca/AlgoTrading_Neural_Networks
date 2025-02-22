from datetime import date, datetime

import pandas as pd
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from starlette.responses import JSONResponse

from common.util.csv_reader import CSVReader
from framework.common.logger.message_type import MessageType
from logic_layer.algos_orchestation_logic import AlgosOrchestationLogic
router = APIRouter()

class DisplayCustomETFController:
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
        self.router.post("/upload_custom_etf")(self.upload_custom_etf)
        self.router.get("/get_chart_data")(self.get_chart_data)

    async def display_page(self, request: Request):
        """Renders the custom ETF upload page."""
        return self.templates.TemplateResponse("display_custom_etf.html", {"request": request})

    from fastapi import UploadFile, File, HTTPException

    async def upload_custom_etf(self, file: UploadFile = File(...),
                                start_date: str = Form(...),
                                end_date: str = Form(...)):
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

            self.logger.do_log( {"message": f"File '{file.filename}' uploaded successfully"},MessageType.INFO)

            return {"message": f"File '{file.filename}' uploaded successfully"}

        except Exception as e:
            import traceback
            error_message = f"❌ Error processing file: {str(e)}\n{traceback.format_exc()}"
            self.logger.do_log(error_message,MessageType.ERROR)  # Mostrar error detallado en la terminal
            raise HTTPException(status_code=500, detail=error_message)


    async def get_chart_data(self):
        """Returns ETF data in JSON format for the chart. Handles empty dataset gracefully."""

        if not self.detailed_MTMS:
            return JSONResponse({
                "dates": [],
                "values": [],
                "message": "No ETF data available. Please upload a file first."
            }, status_code=200)

        df = pd.DataFrame([{"date": obj.date, "MTM": obj.MTM} for obj in self.detailed_MTMS])

        if df.empty:
            return JSONResponse({
                "dates": [],
                "values": [],
                "message": "ETF data is empty after processing. Check the uploaded file."
            }, status_code=200)

        df.sort_values(by="date", inplace=True)
        self.detailed_MTMS=[]#just one display
        return JSONResponse({
            "dates": df["date"].astype(str).tolist(),
            "values": df["MTM"].tolist()
        })

