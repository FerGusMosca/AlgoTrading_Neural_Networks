from typing import Optional

from pydantic import BaseModel


class PositionDTO(BaseModel):
    Symbol: str
    Qty: Optional[float]
    AvgPx: Optional[float]
    Currency: Optional[str]
    Type: Optional[str] = "Stock"  # Podés cambiarlo según `SecurityType`