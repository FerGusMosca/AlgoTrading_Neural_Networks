import asyncio
from typing import Dict

from fastapi import FastAPI, WebSocket, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import threading
import json

from starlette.responses import JSONResponse

from common.dto.websocket_conn.client_messages.new_order_req import NewOrderReq
from common.dto.websocket_conn.execution_report_dto import ExecutionReportDTO
from common.dto.websocket_conn.market_data_dto import MarketDataDTO
from common.dto.websocket_conn.order_dto import OrderDTO
from common.enums.brokers import Brokers
from framework.common.logger.message_type import MessageType
from service_layer.websocket_client import WebSocketClient



class WebManagerLogic:
    def __init__(self, logger, ib_prod_ws, primary_prod_ws, ib_dev_ws):
        self.logger = logger
        self.ib_prod_ws = ib_prod_ws
        self.primary_prod_ws = primary_prod_ws
        self.ib_dev_ws = ib_dev_ws

        self.ws_ib_prod_client=None
        self.ws_ib_dev_client=None
        self.ws_primary_client=None

        self.market_data = {}
        self.execution_reports={}

        # Start evaluating connections in a separate thread
        threading.Thread(target=self._run_async_evaluation, daemon=True).start()

        self.app = FastAPI()
        self.templates = Jinja2Templates(directory="templates")

        # Rutas en FastAPI
        self.app.get("/", response_class=HTMLResponse)(self.read_root)
        self.app.post("/submit_order")(self.submit_order)
        self.app.get("/get_connection_status")(self.get_connection_status)  # Register route
        self.app.get("/get_market_data")(self.get_market_data)
        self.app.get("/get_execution_reports")(self.get_execution_reports)

    def _run_async_evaluation(self):
        """Runs evaluate_connections in an independent event loop."""
        asyncio.run(self.evaluate_connections())

    async def evaluate_connections(self):
        self.ws_ib_prod_client = WebSocketClient(self.ib_prod_ws, self.logger, self.store_market_data,self.store_execution_report)
        self.ws_ib_dev_client = WebSocketClient(self.ib_dev_ws, self.logger, self.store_market_data,self.store_execution_report)
        self.ws_primary_client = WebSocketClient(self.primary_prod_ws, self.logger, self.store_market_data,self.store_execution_report)

        # Dictionary to store connection results
        self.connection_status = {
            Brokers.IB_PROD.value: False,
            Brokers.IB_DEV.value: False,
            Brokers.PRIMARY_PROD.value: False
        }

        # Attempt to connect to each WebSocket
        await asyncio.gather(
            self._connect_and_store_status(Brokers.IB_PROD.value, self.ws_ib_prod_client),
            self._connect_and_store_status(Brokers.IB_DEV.value, self.ws_ib_dev_client),
            self._connect_and_store_status(Brokers.PRIMARY_PROD.value, self.ws_primary_client)
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

    def store_execution_report(self, execution_report: ExecutionReportDTO):
        """Stores the latest execution report for each ClOrdId."""
        key=execution_report.cl_ord_id[:-8]
        self.execution_reports[key] = execution_report.dict()

    def store_market_data(self, market_data: MarketDataDTO):
        """Stores the latest market data for each symbol."""
        self.market_data[market_data.symbol] = market_data.dict()

    from typing import List

    def get_execution_reports(self) -> List[dict]:
        """Returns all stored execution reports as a list."""
        return list(self.execution_reports.values())

    def get_connection_status(self):
        """Returns the connection status as JSON."""
        return JSONResponse(content=self.connection_status)

    def get_market_data(self):
        """Returns the latest market data as JSON."""
        return JSONResponse(content=list(self.market_data.values()))

    async def submit_order(self, order: "OrderDTO" = Body(...)):
        """Receives an order and processes it correctly as JSON"""

        self.logger.do_log(f"New Order Received: {order}", MessageType.INFO)
        new_order = NewOrderReq.from_order_dto(order)
        # If the broker is IB_PROD_WS, transform the order into NewOrderReq
        if order.broker == Brokers.IB_PROD.value:

            self.logger.do_log(f"Transformed Order for IB_PROD_WS: {new_order.json()}", MessageType.INFO)
            # Here, you would send the JSON message via WebSocket to IB_PROD_WS
            await self.ws_ib_prod_client.send_message(new_order.json())

            return {"status": "success", "message": "Order transformed and sent successfully to IB_PROD!!"}

        elif order.broker == Brokers.IB_DEV.value:
            self.logger.do_log(f"Transformed Order for IB_DEV_WS: {new_order.json()}", MessageType.INFO)
            # Here, you would send the JSON message via WebSocket to IB_DEV_WS
            await self.ws_ib_dev_client.send_message(new_order.json())

            return {"status": "success", "message": "Order transformed and sent successfully to IB_DEV!!"}
        elif order.broker == Brokers.PRIMARY_PROD.value:
            self.logger.do_log(f"Transformed Order for PRIMARY_PROD_WS: {new_order.json()}", MessageType.INFO)
            # Here, you would send the JSON message via WebSocket to IB_DEV_WS
            await self.ws_primary_client.send_message(new_order.json())

            return {"status": "success", "message": "Order transformed and sent successfully to PRIMARY_PROD!!"}

        return {"status": "success", "message": "Order received but not transformed."}

    def display_order_routing_screen(self, port=8000):
        """Levanta el servidor en un hilo separado"""
        def run():
            self.logger.do_log(f"Starting Order Routing Screen on port {port}...", MessageType.INFO)
            uvicorn.run(self.app, host="0.0.0.0", port=port)

        threading.Thread(target=run, daemon=True).start()
