import uuid
import datetime
from pydantic import BaseModel
from typing import Optional

class NewOrderReq(BaseModel):
    Msg: str = "NewOrderReq"  # Static message type
    ReqId: str
    UUID: str
    ClOrdId: str
    Symbol: str
    Side: str
    Qty:  Optional[float]= None
    CashQty:  Optional[float]= None
    Account: str
    Type: str
    Price: Optional[float] = None
    Currency: str
    Exchange: str
    TimeInForce: str
    CreationTime: str  # String to match the JSON format

    @staticmethod
    def from_order_dto(order: "OrderDTO") -> "NewOrderReq":
        """
        Converts an OrderDTO into a NewOrderReq format.
        """
        return NewOrderReq(
            ReqId=str(uuid.uuid4()),  # Generate unique request ID
            UUID=str(uuid.uuid4()),  # Generate unique UUID
            ClOrdId=str(uuid.uuid4()),  # Generate a unique Client Order ID
            Symbol=order.symbol,
            Side="BUY" if order.side.lower() == "buy" else "SELL",
            Qty=order.nom_qty if order.nom_qty is not None else None,  # Use whichever is provided
            CashQty= order.cash_qty if order.cash_qty is not None else None,
            Account=order.account,
            Type="MKT",  # Assuming market order for now
            Price=None,  # No price specified for market order
            Currency=order.currency,
            Exchange=order.exchange,
            TimeInForce="DAY",  # Assuming DAY order for now
            CreationTime=datetime.datetime.utcnow().isoformat()  # ISO 8601 timestamp
        )
