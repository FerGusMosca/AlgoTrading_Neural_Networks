import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from fastapi import Request
from starlette.responses import HTMLResponse

from stripe_ACH_POC_controller import StripeAchDemoController
from stripe_USDC_POC_controller import StripeUSDCDemoController
from common.util.logger import Logger
from common.util.ml_settings_loader import MLSettingsLoader

# Init
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configs
loader = MLSettingsLoader()
logger = Logger()
config_settings = loader.load_settings("./configs/commands_mgr.ini")

# Controllers
ach_controller = StripeAchDemoController(config_settings, logger)
usdc_controller = StripeUSDCDemoController(config_settings, logger)

app.include_router(ach_controller.router, prefix="/stripe_ACH_POC")
app.include_router(usdc_controller.router, prefix="/stripe_USDC_POC")

# Templates
templates_path = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=templates_path)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8081, reload=True)
