import asyncio
import time
from http.client import HTTPException
from typing import Dict

from fastapi import FastAPI, WebSocket, Request, Body, APIRouter, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List
import uvicorn
import threading
import json

from starlette.responses import JSONResponse

from common.dto.account_info_dto import AccountInfo
from common.dto.websocket_conn.cancel_order_dto import CancelOrderDTO
from common.dto.websocket_conn.client_messages.cancel_order_req import CancelOrderReq
from common.dto.websocket_conn.client_messages.new_order_req import NewOrderReq
from common.dto.websocket_conn.client_messages.portfolio_req import PortfolioReq
from common.dto.websocket_conn.execution_report_dto import ExecutionReportDTO
from common.dto.websocket_conn.market_data_dto import MarketDataDTO
from common.dto.websocket_conn.order_dto import OrderDTO
from common.enums.brokers import Brokers
from data_access_layer.account_manager import AccountManager
from framework.common.logger.message_type import MessageType
from service_layer.websocket_client import WebSocketClient



class RoutingDashboardController:
    def __init__(self, logger, ib_prod_ws, primary_prod_ws, ib_dev_ws,fund_mgmt_dashboard_cs):
        self.logger = logger
        self.ib_prod_ws = ib_prod_ws
        self.primary_prod_ws = primary_prod_ws
        self.ib_dev_ws = ib_dev_ws

        self.ws_ib_prod_client = None
        self.ws_ib_dev_client = None
        self.ws_primary_client = None

        self.market_data = {}
        self.execution_reports = {}
        self.order_broker = {}
        self.portfolios = {}  # Keyed by AccountNumber

        self.account_manager = AccountManager(fund_mgmt_dashboard_cs)

        # Start evaluating connections in a separate thread
        threading.Thread(target=self._run_async_evaluation, daemon=True).start()

        # ✅ Use APIRouter instead of FastAPI
        self.router = APIRouter()

        self.templates = Jinja2Templates(directory="templates")

        # ✅ Register routes properly in the router
        self.router.get("/", response_class=HTMLResponse)(self.read_root)
        self.router.post("/submit_order")(self.submit_order)
        self.router.post("/cancel_order")(self.cancel_order)
        self.router.get("/get_connection_status")(self.get_connection_status)
        self.router.get("/get_market_data")(self.get_market_data)
        self.router.get("/get_execution_reports")(self.get_execution_reports)
        self.router.get("/get_accounts")(self.get_accounts)
        self.router.get("/get_portfolio")(self.get_portfolio)

    async def read_root(self, request: Request):
        """Serves the Routing Dashboard HTML page."""
        return self.templates.TemplateResponse("order_routing_template.html", {"request": request})
    def _run_async_evaluation(self):
        """Runs evaluate_connections in an independent event loop."""
        asyncio.run(self.evaluate_connections())

    async def evaluate_connections(self):
        self.ws_ib_prod_client = WebSocketClient(self.ib_prod_ws, self.logger, self.store_market_data,
                                                 self.store_execution_report,self.store_portfolio,Brokers.IB_PROD.value)
        self.ws_ib_dev_client = WebSocketClient(self.ib_dev_ws, self.logger, self.store_market_data,
                                                self.store_execution_report,self.store_portfolio,Brokers.IB_DEV.value)
        self.ws_primary_client = WebSocketClient(self.primary_prod_ws, self.logger, self.store_market_data,
                                                 self.store_execution_report,self.store_portfolio,Brokers.BYMA_PROD.value)

        # Dictionary to store connection results
        self.connection_status = {
            Brokers.IB_PROD.value: False,
            Brokers.IB_DEV.value: False,
            Brokers.BYMA_PROD.value: False
        }

        # Attempt to connect to each WebSocket
        await asyncio.gather(
            self._connect_and_store_status(Brokers.IB_PROD.value, self.ws_ib_prod_client),
            self._connect_and_store_status(Brokers.IB_DEV.value, self.ws_ib_dev_client),
            self._connect_and_store_status(Brokers.BYMA_PROD.value, self.ws_primary_client)
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
        key=ExecutionReportDTO.get_execution_report_key(execution_report.cl_ord_id)
        self.execution_reports[key] = execution_report.dict()

    def store_portfolio(self, portfolio):
        self.portfolios[portfolio.AccountNumber] = portfolio

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

    async def send_to_broker(self,json,broker):
        if broker == Brokers.IB_PROD.value:
            await self.ws_ib_prod_client.send_message(json)
        elif broker==Brokers.IB_DEV.value:
            await self.ws_ib_dev_client.send_message(json)
        elif broker==Brokers.BYMA_PROD.value:
            await self.ws_primary_client.send_message(json)
        else:
            raise Exception(f"Unknown broker {broker} to send message!")

    async def submit_order(self, order: "OrderDTO" = Body(...)):
        """Receives an order and processes it correctly as JSON"""

        self.logger.do_log(f"New Order Received: {order}", MessageType.INFO)
        new_order = NewOrderReq.from_order_dto(order)
        # If the broker is IB_PROD_WS, transform the order into NewOrderReq
        if order.broker == Brokers.IB_PROD.value:

            self.logger.do_log(f"Transformed Order for IB_PROD_WS: {new_order.json()}", MessageType.INFO)
            self.order_broker[new_order.ClOrdId]=Brokers.IB_PROD.value
            await self.send_to_broker(new_order.json(),Brokers.IB_PROD.value)
            return {"status": "success", "message": "Order transformed and sent successfully to IB_PROD!!"}

        elif order.broker == Brokers.IB_DEV.value:
            self.logger.do_log(f"Transformed Order for IB_DEV_WS: {new_order.json()}", MessageType.INFO)
            # Here, you would send the JSON message via WebSocket to IB_DEV_WS
            self.order_broker[new_order.ClOrdId] = Brokers.IB_DEV.value
            await self.send_to_broker(new_order.json(), Brokers.IB_DEV.value)
            return {"status": "success", "message": "Order transformed and sent successfully to IB_DEV!!"}

        elif order.broker == Brokers.BYMA_PROD.value:
            self.logger.do_log(f"Transformed Order for PRIMARY_PROD_WS: {new_order.json()}", MessageType.INFO)
            # Here, you would send the JSON message via WebSocket to IB_DEV_WS
            self.order_broker[new_order.ClOrdId] = Brokers.BYMA_PROD.value
            await self.send_to_broker(new_order.json(), Brokers.BYMA_PROD.value)
            return {"status": "success", "message": "Order transformed and sent successfully to BYMA_PROD!!"}

        return {"status": "success", "message": "Order received but not transformed."}


    async def cancel_order(self, cancel_request: CancelOrderDTO):
        """Cancela una orden si existe en execution_reports."""
        cl_ord_id = cancel_request.cl_ord_id
        key = ExecutionReportDTO.get_execution_report_key(cl_ord_id)
        if key in self.execution_reports :
            exec_report =self.execution_reports[key]
            cancel_req = CancelOrderReq.from_cl_ord_id(cl_ord_id)
            await self.send_to_broker( cancel_req.model_dump_json(),exec_report["broker"])
            self.logger.do_log(f"Order {cl_ord_id} canceled.", MessageType.INFO)
            return {"status": "success", "message": f"Order {cl_ord_id} cancelation sent!."}
        else:
            raise HTTPException(status_code=404, detail="Order not found.")

    def display_order_routing_screen(self, port=8000):
        """Levanta el servidor en un hilo separado"""
        def run():
            self.logger.do_log(f"Starting Order Routing Screen on port {port}...", MessageType.INFO)
            uvicorn.run(self.app, host="0.0.0.0", port=port)

        threading.Thread(target=run, daemon=True).start()

    def get_accounts(self, broker: str = Query(...)) -> List[AccountInfo]:
        """Returns account options for a specific broker from the database."""
        accounts = self.account_manager.get_all_accounts()
        return [
            AccountInfo(account_id=acc.account_number, client_name=acc.account_name)
            for acc in accounts
            if acc.broker == broker
        ]

    def _get_broker_for_account(self, account_number: str) -> str:
        for acc in self.account_manager.get_all_accounts():
            if acc.account_number == account_number:
                return acc.broker
        raise ValueError(f"Broker not found for account {account_number}")

    async def get_portfolio(self, account_id: str = Query(...)):
        account_number = account_id  # mismo valor
        broker = self._get_broker_for_account(account_number)

        req = PortfolioReq(AccountNumber=account_number)

        await self.send_to_broker(req.model_dump_json(), broker)

        for _ in range(10):  # Espera máx 5 segs
            if account_number in self.portfolios:
                portfolio = self.portfolios[account_number]
                return JSONResponse(content={
                    "securities": [
                        {
                            "symbol": pos.Symbol,
                            "type": pos.Type,
                            "currency": pos.Currency,
                            "qty": pos.Qty,
                            "avg_px": pos.AvgPx
                        } for pos in portfolio.SecurityPositions
                    ],
                    "currencies": [
                        {
                            "amount": pos.Qty,
                            "currency": pos.Currency
                        } for pos in portfolio.LiquidPositions
                    ]
                })
            time.sleep(0.5)
        else:
            return JSONResponse(content={"error": "Portfolio not received"}, status_code=504)
