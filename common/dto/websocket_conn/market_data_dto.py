from pydantic import BaseModel

class MarketDataDTO(BaseModel):
    symbol: str
    exchange: str
    opening_price: float
    high_price: float
    low_price: float
    closing_price: float
    last_trade_price: float
    timestamp: str
