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

    def create_usdc_payment_intent_manual(self, payload: UsdcPaymentIntentRequest):


        # Your test secret key
        stripe.api_key = payload.secret_key

        # 1. Create a temporary customer
        customer = stripe.Customer.create(
            email="test-crypto@example.com",
            description="Manual test customer for USDC"
        )

        # 2. Create the PaymentIntent with payment_method_types=["crypto"]
        intent = stripe.PaymentIntent.create(
            amount=1000,  # $10 USD
            currency="usd",
            customer=customer.id,
            payment_method_types=["crypto"],
            description="Manual USDC test intent"
        )

        print(">> Created PaymentIntent:", intent.id)
        print(">> Initial status:", intent.status)

        # 3. Manually confirm the intent
        confirmed_intent = stripe.PaymentIntent.confirm(
            intent.id,
            payment_method="crypto"
        )

        print(">> Confirmed")
        print(">> Status:", confirmed_intent.status)
        print(">> next_action:", confirmed_intent.next_action)

        if confirmed_intent.next_action and confirmed_intent.next_action["type"] == "verify_with_crypto":
            print("✅ Hosted checkout URL:", confirmed_intent.next_action["verify_with_crypto"]["hosted_url"])
        else:
            print("❌ No hosted_url generated")

    async def create_usdc_payment_intent(self, payload: UsdcPaymentIntentRequest):
        try:

            #self.create_usdc_payment_intent_manual(payload)
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

