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

        templates_path = Path(__file__).parent / "templates"
        self.templates = Jinja2Templates(directory=templates_path)

        # Registrar handlers
        self.router.get("/", response_class=HTMLResponse)(self.display_page)
        self.router.post("/create_usdc_payment_intent")(self.create_usdc_payment_intent)

    async def display_page(self, request: Request):
        return self.templates.TemplateResponse("stripe_USDC_POC.html", {"request": request})
    def create_usdc_payment_intent_autom(self, payload: UsdcPaymentIntentRequest):
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

    def create_usdc_payment_intent_manual(self, payload: UsdcPaymentIntentRequest):
        import stripe
        stripe.api_key = payload.secret_key

        # 1. Create customer
        customer = stripe.Customer.create(
            email="test-crypto@example.com",
            description="Manual test customer for USDC"
        )

        # 2. Create PaymentIntent (manual flow)
        intent = stripe.PaymentIntent.create(
            amount=payload.amount,
            currency="usd",
            customer=customer.id,
            payment_method_types=["crypto"],
            description="USDC test intent"
        )

        print("✅ Created PaymentIntent:", intent.id)
        print("📄 Status:", intent.status)

        hosted_url = None
        if intent.next_action and intent.next_action.get("type") == "verify_with_crypto":
            hosted_url = intent.next_action["verify_with_crypto"]["hosted_url"]
            print("🔗 Hosted checkout URL:", hosted_url)
        else:
            print("ℹ️ No hosted_url available at this stage (expected in manual flow)")

        # Optional: Stripe debug link
        if intent.last_response and hasattr(intent.last_response, "request_id"):
            print("🔍 Stripe log URL:")
            print(f"https://dashboard.stripe.com/test/logs/{intent.last_response.request_id}")

        return JSONResponse(content={
            "client_secret": intent.client_secret,
            "payment_intent_id": intent.id,
            "payment_intent_url": hosted_url  # May be None initially
        })

    async def create_usdc_payment_intent(self, payload: UsdcPaymentIntentRequest):

        return self.create_usdc_payment_intent_manual(payload)
        #self.create_usdc_payment_intent_autom(payload)


