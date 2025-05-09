from fastapi import Request
from fastapi import FastAPI, Request, APIRouter
from fastapi.params import Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from fastapi import Response
from itsdangerous import TimestampSigner
from starlette.responses import RedirectResponse, JSONResponse

from controllers.account_controller import AccountController
from controllers.auth_middleware import AuthMiddleware
from controllers.display_custom_etf_controller import DisplayCustomETFController
from controllers.display_series_controller import DisplaySeriesController
from controllers.load_series_controller import LoadSeriesController
from controllers.routing_dashboard_controller import RoutingDashboardController
from controllers.simulate_indicator_strategy_controller import SimulateIndicatorStrategy
from controllers.stripe_ACH_POC_controller import StripeAchDemoController
from data_access_layer.account_manager import AccountManager
from data_access_layer.user_manager import UserManager
from framework.common.logger.message_type import MessageType
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
class MainDashboardController:
    def __init__(self, logger, config_settings):
        self.logger = logger

        self.valid_users = {"admin": "Test123123"}


        self.signer = TimestampSigner("my_super_secret")

        # ✅ Create the main FastAPI instance
        self.app = FastAPI()

        ib_prod_ws = config_settings["IB_PROD_WS"]
        primary_prod_ws = config_settings["PRIMARY_PROD_WS"]
        ib_dev_ws = config_settings["IB_DEV_WS"]
        fund_mgmt_dashboard_cs= config_settings["fund_mgmt_dashboard_cs"]
        secret_key = "my_super_secret"  # El secreto utilizado para firmar tokens

        # Inicializamos el UserManager
        self.user_manager = UserManager(fund_mgmt_dashboard_cs, secret_key)

        # ✅ Instantiate RoutingDashboardController (WITHOUT creating another FastAPI app)
        self.routing_dashboard = RoutingDashboardController(logger, ib_prod_ws, primary_prod_ws, ib_dev_ws,fund_mgmt_dashboard_cs)
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

        # 📌 Register Account Controller
        self.account_controller = AccountController(AccountManager(fund_mgmt_dashboard_cs))
        self.app.include_router(self.account_controller.router)


        # 📌 Stripe ACH POC
        self.stripe_ACH_POC_controller = StripeAchDemoController(config_settings, logger)
        self.app.include_router(self.stripe_ACH_POC_controller.router, prefix="/stripe_ACH_POC")

        # ✅ Set up the templates directory
        templates_path = Path(__file__).parent.parent / "templates"
        self.templates = Jinja2Templates(directory=templates_path)

        # ✅ Define the main route
        self.app.get("/", response_class=HTMLResponse)(self.main_dashboard)

        # ✅ Serve static files (CSS, JS, etc.)
        self.app.mount("/static", StaticFiles(directory="static"), name="static")

        self.app.post("/login")(self.login)

        self.app.get("/login", response_class=HTMLResponse)(self.login_form)
        self.app.get("/logout")(self.logout)

        self.app.add_event_handler("startup", self.startup_event)

        self.app.add_middleware(
            AuthMiddleware,
            secret_key="my_super_secret",
            exempt_paths=["/login", "/login/", "/static", "/static/", "/favicon.ico"]
        )

    async def startup_event(self):
        await self.routing_dashboard.initialize()
        self.app.include_router(self.routing_dashboard.router)

    async def login_form(self, request: Request):
        return self.templates.TemplateResponse("login.html", {"request": request})

    async def main_dashboard(self, request: Request):
        return self.templates.TemplateResponse("main_dashboard.html", {"request": request})

    def display(self, port=8000):
        """Starts the main dashboard server with all integrated dashboards."""

        def run():
            self.logger.do_log(f"Starting Main Dashboard on port {port}...", MessageType.INFO)
            import uvicorn
            uvicorn.run(self.app, host="0.0.0.0", port=port)

        import threading
        threading.Thread(target=run, daemon=True).start()

    from fastapi import Request

    async def login(self, request: Request, username: str = Form(...), password: str = Form(...)):
        """
        Handles the login logic, checking user credentials and providing feedback
        """
        if self.user_manager.authenticate_user(username, password):
            token = self.signer.sign("session").decode()
            response = RedirectResponse(url="/", status_code=302)
            response.set_cookie(key="session", value=token, httponly=True, max_age=3600)
            return response
        else:
            error_message = "Invalid credentials: The username or password you entered is incorrect. Please check your input and try again."
            return self.templates.TemplateResponse("login.html", {"request": request, "error_message": error_message})

    async def logout(self, request: Request):
        response = RedirectResponse(url="/login", status_code=302)
        response.delete_cookie("session")
        return response


