from pydantic import BaseModel
from typing import Optional

from framework.common.enums.OrdStatus import OrdStatus
from framework.common.enums.OrdType import OrdType
from framework.common.enums.Side import Side


class ExecutionReportDTO(BaseModel):
    cl_ord_id: str
    short_cl_ord_id: str
    orig_cl_ord_id: str
    order_id: str
    symbol:str
    side:str
    ord_type:str
    status: str
    transact_time: str
    exec_type: int
    ord_status: str
    order_qty: float
    price:float
    avg_px: float
    cum_qty: float
    leaves_qty: float
    last_px: float
    currency: str
    broker: str


    @staticmethod
    def get_execution_report_key(cl_ord_id):
        key = cl_ord_id[:-8]
        return key

    @staticmethod
    def get_short_cl_ord_id(cl_ord_id):
        key = ExecutionReportDTO.get_execution_report_key(cl_ord_id)
        key = f"...{key[-16:]}"

        return key


    @staticmethod
    def parse_side(value):
        """Castea el valor de Side del JSON al Enum Side."""
        try:
            if isinstance(value, int):  # Si el valor es un número (como 49)
                value = chr(value)  # Convierte el número ASCII al carácter correspondiente

            side= Side(str(value))
            return side.name
        except ValueError:
            return Side.Unknown.name


    @staticmethod
    def parse_ord_type(value):
        """Castea el valor de Side del JSON al Enum Side."""
        try:
            if isinstance(value, int):  # Si el valor es un número (como 49)
                value = chr(value)  # Convierte el número ASCII al carácter correspondiente

            ord_type= OrdType(str(value))
            return ord_type.name
        except ValueError:
            return OrdType.Market.name


    @staticmethod
    def parse_ord_status(value):
        """Castea el valor de Side del JSON al Enum Side."""
        try:
            if isinstance(value, int):  # Si el valor es un número (como 49)
                value = chr(value)  # Convierte el número ASCII al carácter correspondiente

            ord_status= OrdStatus(str(value))
            return ord_status.name
        except ValueError:
            return OrdStatus.Undefined.name


    @staticmethod
    def from_execution_report(data: dict, broker_id: str) -> "ExecutionReportDTO":
        """
        Converts the received WebSocket execution report message into an ExecutionReportDTO.
        """
        order_data = data.get("Order", {})

        return ExecutionReportDTO(
            cl_ord_id=data.get("ClOrdId", "UNKNOWN"),
            short_cl_ord_id=ExecutionReportDTO.get_short_cl_ord_id(data.get("ClOrdId", "UNKNOWN")),
            orig_cl_ord_id=data.get("OrigClOrdId", "UNKNOWN"),
            order_id=data.get("OrderId", "UNKNOWN"),
            symbol=order_data.get("Symbol","??"),
            side=ExecutionReportDTO.parse_side( order_data.get("Side","UNK")),
            ord_type=ExecutionReportDTO.parse_ord_type(order_data.get("OrdType","??")),
            order_qty=order_data.get("OrderQty",0),
            price=order_data.get("Price"),
            status=data.get("Status", "UNKNOWN"),
            transact_time=data.get("TransactTime", "UNKNOWN"),
            exec_type=data.get("ExecType", 0),
            ord_status=ExecutionReportDTO.parse_ord_status(data.get("OrdStatus", 0)),
            avg_px=data.get("AvgPx", 0.0),
            cum_qty=data.get("CumQty", 0.0),
            leaves_qty=data.get("LeavesQty", 0.0),
            last_px=data.get("LastPx", 0.0),
            currency=order_data.get("Currency", "UNKNOWN"),
            broker=broker_id
        )
