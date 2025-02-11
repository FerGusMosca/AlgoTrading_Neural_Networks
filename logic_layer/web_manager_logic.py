from fastapi import FastAPI, WebSocket, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import threading
import json
from framework.common.logger.message_type import MessageType

class OrderDTO(BaseModel):
    symbol: str
    side: str
    cash_qty: float | None
    nom_qty: int | None
    broker: str
    currency: str
    exchange: str
    account: str

class WebManagerLogic:
    def __init__(self, logger, ib_prod_ws, primary_prod_ws, ib_dev_ws):
        self.logger = logger
        self.ib_prod_ws = ib_prod_ws
        self.primary_prod_ws = primary_prod_ws
        self.ib_dev_ws = ib_dev_ws

        self.app = FastAPI()
        self.templates = Jinja2Templates(directory="templates")

        # Rutas en FastAPI
        self.app.get("/", response_class=HTMLResponse)(self.read_root)
        self.app.post("/submit_order")(self.submit_order)

    async def read_root(self, request: Request):
        """Renderiza la página principal"""
        return self.templates.TemplateResponse("order_routing_template.html", {"request": request})

    async def submit_order(self, order: OrderDTO = Body(...)):
        """Recibe la orden y la procesa correctamente como JSON"""
        self.logger.do_log(f"New Order Received: {order}", MessageType.INFO)
        return {"status": "success", "message": "Order received successfully!"}

    def display_order_routing_screen(self, port=8000):
        """Levanta el servidor en un hilo separado"""
        def run():
            self.logger.do_log(f"Starting Order Routing Screen on port {port}...", MessageType.INFO)
            uvicorn.run(self.app, host="127.0.0.1", port=port)

        threading.Thread(target=run, daemon=True).start()
