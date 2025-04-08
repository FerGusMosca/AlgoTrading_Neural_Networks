from typing import List

from pydantic import BaseModel

from common.dto.websocket_conn.position_dto import PositionDTO


class PortfolioDTO(BaseModel):
    Msg: str
    AccountNumber: str
    SecurityPositions: List[PositionDTO]
    LiquidPositions: List[PositionDTO]

    @staticmethod
    def from_raw_msg(data: dict) -> "PortfolioDTO":
        """Builds PortfolioDTO from raw WebSocket JSON structure"""

        def parse_position(p: dict) -> PositionDTO:
            return PositionDTO(
                Symbol=p.get("Symbol") or p.get("Security", {}).get("Symbol", "??"),
                Qty=p.get("Qty", 0),
                AvgPx=p.get("AvgPx", 0.0),
                Currency=p.get("Security", {}).get("Currency", "UNKNOWN"),
                Type="Stock" if p.get("QuantityType") == 1 else "Currency"  # ajustar según lógica real
            )

        securities = [parse_position(p) for p in data.get("SecurityPositions", [])]
        currencies = [parse_position(p) for p in data.get("LiquidPositions", [])]

        return PortfolioDTO(
            Msg=data.get("Msg", "PortfolioMsg"),
            AccountNumber=data.get("AccountNumber", ""),
            SecurityPositions=securities,
            LiquidPositions=currencies
        )