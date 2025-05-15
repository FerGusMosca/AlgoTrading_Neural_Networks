from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import stripe
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from framework.common.logger.message_type import MessageType
from controllers.base_controller import BaseController

class SetupIntentRequest(BaseModel):
    secret_key: str
    email: str
    name: str

class StripeAchDemoController(BaseController):
    def __init__(self, config_settings, logger):
        super().__init__()
        self.logger = logger
        self.router = APIRouter()

        templates_path = Path(__file__).parent.parent / "templates"
        self.templates = Jinja2Templates(directory=templates_path)

        self.router.get("/", response_class=HTMLResponse)(self.display_page)
        self.router.post("/create_setup_intent")(self.create_setup_intent_endpoint)

    async def display_page(self, request: Request):
        return self.templates.TemplateResponse("stripe_ACH_POC.html", {"request": request})

    async def create_setup_intent_endpoint(self, payload: SetupIntentRequest):
        try:
            stripe.api_key = payload.secret_key

            # Create customer first
            customer = stripe.Customer.create(
                email=payload.email,
                name=payload.name
            )

            # Then create SetupIntent with the customer
            setup_intent = stripe.SetupIntent.create(
                customer=customer.id,
                payment_method_types=["us_bank_account"],
                usage="off_session"
            )

            self.logger.do_log(f"✅ SetupIntent created: {setup_intent.id}", MessageType.INFO)

            return JSONResponse(content={
                "client_secret": setup_intent.client_secret,
                "setup_intent_id": setup_intent.id,
                "customer_id": customer.id
            })

        except Exception as e:
            self.logger.do_log(f"❌ Error creating SetupIntent: {str(e)}", MessageType.ERROR)
            return JSONResponse(status_code=500, content={"error": str(e)})
