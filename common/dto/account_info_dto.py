from pydantic import BaseModel


class AccountInfo(BaseModel):
    account_id: str
    client_name: str