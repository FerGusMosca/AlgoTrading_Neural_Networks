from pydantic import BaseModel


class PortfolioReq(BaseModel):
    Msg: str = "PortfolioRequest"
    AccountNumber: str