from pydantic import BaseModel


class CancelOrderDTO(BaseModel):
    cl_ord_id: str