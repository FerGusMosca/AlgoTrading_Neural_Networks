import uuid
import datetime
from pydantic import BaseModel
from typing import Optional

from common.enums.platform_ord_type import PlatformOrdType
from common.enums.platform_side import PlatformSide
from common.enums.platform_tif import PlatformTif


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
            Side=PlatformSide.BUY.value if order.side.upper() == PlatformSide.BUY.value else PlatformSide.SELL.value,
            Qty=order.nom_qty if order.nom_qty is not None else  None,  # Use whichever is provided
            CashQty= order.cash_qty if order.cash_qty is not None else None,
            Account=order.account,
            Type=PlatformOrdType.MARKET.value,  # Assuming market order for now
            Price=None,  # No price specified for market order
            Currency=order.currency,
            Exchange=order.exchange,
            TimeInForce=PlatformTif.DAY.value,  # Assuming DAY order for now
            CreationTime=datetime.datetime.utcnow().isoformat()  # ISO 8601 timestamp
        )
