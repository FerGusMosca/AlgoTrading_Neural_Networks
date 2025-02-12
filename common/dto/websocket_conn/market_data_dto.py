from pydantic import BaseModel
from typing import Optional

class MarketDataDTO(BaseModel):
    symbol: str
    exchange: Optional[str] = "UNKNOWN"  # Si falta, ponemos "UNKNOWN"
    opening_price: Optional[float] = 0.0
    high_price: Optional[float] = 0.0
    low_price: Optional[float] = 0.0
    closing_price: Optional[float] = 0.0
    last_trade_price: Optional[float] = 0.0
    trade_volume: Optional[float] = 0.0
    timestamp: str
