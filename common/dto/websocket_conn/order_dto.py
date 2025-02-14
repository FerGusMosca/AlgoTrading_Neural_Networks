from pydantic import BaseModel

class OrderDTO(BaseModel):
    symbol: str
    side: str
    cash_qty: float | None
    nom_qty: int | None
    broker: str
    currency: str
    exchange: str
    account: str
