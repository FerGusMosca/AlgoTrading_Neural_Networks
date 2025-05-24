from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import stripe
from controllers.base_controller import BaseController
from framework.common.logger.message_type import MessageType
from pathlib import Path
from fastapi.templating import Jinja2Templates
from fastapi import Request
from starlette.responses import HTMLResponse

class UsdcPaymentIntentRequest(BaseModel):
    secret_key: str
    amount: int  # in cents

class StripeUSDCDemoController(BaseController):
    def __init__(self, config_settings, logger):
        super().__init__()
        self.logger = logger
        self.router = APIRouter()

        templates_path = Path(__file__).parent.parent / "templates"
        self.templates = Jinja2Templates(directory=templates_path)

        self.router.get("/", response_class=HTMLResponse)(self.display_page)
        self.router.post("/create_usdc_payment_intent")(self.create_usdc_payment_intent)

    async def display_page(self, request: Request):
        return self.templates.TemplateResponse("stripe_USDC_POC.html", {"request": request})

    async def create_usdc_payment_intent(self, payload: UsdcPaymentIntentRequest):
        try:
            stripe.api_key = payload.secret_key

            # Crear un customer temporal (en producción, deberías usar uno real o existente)
            customer = stripe.Customer.create(description="Temp customer for crypto test")

            intent = stripe.PaymentIntent.create(
                amount=payload.amount,
                currency='usd',
                description='USDC test payment',
                customer=customer.id,
                automatic_payment_methods={"enabled": True}
            )

            # Refrescamos para que incluya el next_action si aplica
            intent = stripe.PaymentIntent.retrieve(intent.id)

            self.logger.do_log(f"✅ Created USDC PaymentIntent: {intent.id}", MessageType.INFO)
            print(">> Stripe intent status:", intent.status)
            print(">> Stripe next_action:", intent.next_action)

            hosted_url = None
            if intent.next_action and intent.next_action.get("type") == "verify_with_crypto":
                hosted_url = intent.next_action["verify_with_crypto"]["hosted_url"]

            return JSONResponse(content={
                "client_secret": intent.client_secret,
                "payment_intent_id": intent.id,
                "payment_intent_url": hosted_url
            })

        except Exception as e:
            self.logger.do_log(f"❌ Error creating USDC PaymentIntent: {str(e)}", MessageType.ERROR)
            return JSONResponse(status_code=500, content={"error": str(e)})

