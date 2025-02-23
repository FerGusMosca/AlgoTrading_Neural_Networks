from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from controllers.display_custom_etf_controller import DisplayCustomETFController
from controllers.display_series_controller import DisplaySeriesController
from controllers.load_series_controller import LoadSeriesController
from controllers.routing_dashboard_controller import RoutingDashboardController
from controllers.simulate_indicator_strategy_controller import SimulateIndicatorStrategy
from framework.common.logger.message_type import MessageType


class MainDashboardController:
    def __init__(self, logger, config_settings):
        self.logger = logger

        # ✅ Create the main FastAPI instance
        self.app = FastAPI()

        ib_prod_ws = config_settings["IB_PROD_WS"]
        primary_prod_ws = config_settings["PRIMARY_PROD_WS"]
        ib_dev_ws = config_settings["IB_DEV_WS"]

        # ✅ Instantiate RoutingDashboardController (WITHOUT creating another FastAPI app)
        self.routing_dashboard = RoutingDashboardController(logger, ib_prod_ws, primary_prod_ws, ib_dev_ws)
        # ✅ Include the router from RoutingDashboardController
        self.app.include_router(self.routing_dashboard.router, prefix="/routing_dashboard")

        # 📌 Register Display Custom ETF Controller
        self.custom_etf_controller = DisplayCustomETFController(config_settings,logger)
        self.app.include_router(self.custom_etf_controller.router, prefix="/display_custom_etf")

        # 📌 Register Simulate Indicator Strategy Controller
        self.load_series_controller = LoadSeriesController(config_settings, logger)
        self.app.include_router(self.load_series_controller.router, prefix="/load_series")

        # 📌 Register Load Series Controller
        self.simulate_indicator_strategy = SimulateIndicatorStrategy(config_settings, logger)
        self.app.include_router(self.simulate_indicator_strategy.router, prefix="/simulate_indicator_strategy")

        # 📌 Register Display Series Controller
        self.display_series_controller = DisplaySeriesController(config_settings, logger)
        self.app.include_router(self.display_series_controller.router, prefix="/display_series")

        # ✅ Set up the templates directory
        templates_path = Path(__file__).parent.parent / "templates"
        self.templates = Jinja2Templates(directory=templates_path)

        # ✅ Define the main route
        self.app.get("/", response_class=HTMLResponse)(self.main_dashboard)

        # ✅ Serve static files (CSS, JS, etc.)
        self.app.mount("/static", StaticFiles(directory="static"), name="static")

    async def main_dashboard(self, request: Request):
        """Renders the main landing page."""
        return self.templates.TemplateResponse("main_dashboard.html", {"request": request})

    def display(self, port=8000):
        """Starts the main dashboard server with all integrated dashboards."""

        def run():
            self.logger.do_log(f"Starting Main Dashboard on port {port}...", MessageType.INFO)
            import uvicorn
            uvicorn.run(self.app, host="127.0.0.1", port=port)

        import threading
        threading.Thread(target=run, daemon=True).start()

    async def main_dashboard(self, request: Request):
        """Renders the main landing page."""
        return self.templates.TemplateResponse("main_dashboard.html", {"request": request})

