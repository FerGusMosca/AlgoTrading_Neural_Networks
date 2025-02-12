import asyncio

from fastapi import FastAPI, WebSocket, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import threading
import json

from starlette.responses import JSONResponse

from common.dto.websocket_conn.market_data_dto import MarketDataDTO
from framework.common.logger.message_type import MessageType
from service_layer.websocket_client import WebSocketClient


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

        self.ws_prod_client=None
        self.ws_dev_client=None
        self.ws_primary_client=None

        self.market_data = {}

        # Start evaluating connections in a separate thread
        threading.Thread(target=self._run_async_evaluation, daemon=True).start()

        self.app = FastAPI()
        self.templates = Jinja2Templates(directory="templates")

        # Rutas en FastAPI
        self.app.get("/", response_class=HTMLResponse)(self.read_root)
        self.app.post("/submit_order")(self.submit_order)
        self.app.get("/get_connection_status")(self.get_connection_status)  # Register route
        self.app.get("/get_market_data")(self.get_market_data)

    def _run_async_evaluation(self):
        """Runs evaluate_connections in an independent event loop."""
        asyncio.run(self.evaluate_connections())

    async def evaluate_connections(self):
        self.ws_prod_client = WebSocketClient(self.ib_prod_ws, self.logger,self.store_market_data)
        self.ws_dev_client = WebSocketClient(self.ib_dev_ws, self.logger,self.store_market_data)
        self.ws_primary_client = WebSocketClient(self.primary_prod_ws, self.logger,self.store_market_data)

        # Dictionary to store connection results
        self.connection_status = {
            "IB_PROD": False,
            "IB_DEV": False,
            "PRIMARY_PROD": False
        }

        # Attempt to connect to each WebSocket
        await asyncio.gather(
            self._connect_and_store_status("IB_PROD", self.ws_prod_client),
            self._connect_and_store_status("IB_DEV", self.ws_dev_client),
            self._connect_and_store_status("PRIMARY_PROD", self.ws_primary_client)
        )

        # Log results
        self.logger.do_log("WebSocket connection evaluation completed.", MessageType.INFO)

    async def _connect_and_store_status(self, name, ws_client):
        """Attempts to connect a WebSocket client and stores its status."""
        await ws_client.connect()
        self.connection_status[name] = ws_client.get_status()
        self.logger.do_log(f"{name} Connection Status: {self.connection_status[name]}", MessageType.INFO)

    async def read_root(self, request: Request):
        """Renderiza la página principal"""
        return self.templates.TemplateResponse("order_routing_template.html", {"request": request})

    def store_market_data(self, market_data: MarketDataDTO):
        """Stores the latest market data for each symbol."""
        self.market_data[market_data.symbol] = market_data.dict()

    def get_connection_status(self):
        """Returns the connection status as JSON."""
        return JSONResponse(content=self.connection_status)

    def get_market_data(self):
        """Returns the latest market data as JSON."""
        return JSONResponse(content=list(self.market_data.values()))

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
