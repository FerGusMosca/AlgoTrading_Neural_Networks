from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from business_entities.portf_position import PortfolioPosition
from common.util.std_in_out.file_writer import FileWriter
from controllers.base_controller import BaseController
from framework.common.logger.message_type import MessageType
from logic_layer.algos_orchestation_logic import AlgosOrchestationLogic


class SimulateIndicatorStrategy(BaseController):

    def __init__(self,config_settings,logger):
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
        self.router.post("/simulate_indicator")(self.simulate_indicator)
        self.router.get("/get_chart_data")(self.get_chart_data)

    async def display_page(self, request: Request):
        """Renders the custom ETF upload page."""
        return self.templates.TemplateResponse("simulate_indicator_strategy.html", {"request": request})


    async def simulate_indicator(self, file: UploadFile = File(...), indicator: str = Form(...),
                                 d_from: str = Form(...), d_to: str = Form(...),  portf_size: float = Form(...),
                                 comm: float = Form(...),trading_algo: str = Form(...), slope_units: int = Form(...),
                                 min_units_to_pred: int = Form(...)):

        try:
            if not file:
                raise HTTPException(status_code=400, detail="No file uploaded.")

            self.logger.do_log(f"✅ File received: {file.filename}", MessageType.INFO)

            aol = AlgosOrchestationLogic(self.config_settings["hist_data_conn_str"],
                                         self.config_settings["ml_reports_conn_str"],
                                         None, self.logger)

            file_path= await FileWriter.dump_on_path(file)
            dstart_date = datetime.strptime(d_from, "%Y-%m-%d").date()
            dend_date = datetime.strptime(d_to, "%Y-%m-%d").date()

            cmd_param_dict = {}
            cmd_param_dict["slope_units"] = slope_units
            cmd_param_dict["trade_comm"] = comm
            cmd_param_dict["days_to_add_to_date"]=min_units_to_pred

            summ_dto,portf_positions=aol.process_backtest_slope_model_on_custom_etf(file_path,indicator,dstart_date,dend_date,
                                                                                    portf_size,trading_algo,cmd_param_dict)

            detailed_positions_joined=[]
            for portf_pos in portf_positions:
                detailed_positions_joined.extend(portf_pos.detailed_MTMS)

            self.detailed_MTMS=PortfolioPosition.fill_missing_dates(detailed_positions_joined)
            self.logger.do_log({"message": f"File '{file.filename}' uploaded successfully"}, MessageType.INFO)

            return {"message": f"File '{file.filename}' uploaded successfully"}

        except Exception as e:
            import traceback
            error_message = f"❌ Error processing file: {str(e)}\n{traceback.format_exc()}"
            self.logger.do_log(error_message, MessageType.ERROR)  # Mostrar error detallado en la terminal
            raise HTTPException(status_code=500, detail=error_message)
