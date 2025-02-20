from pydantic import BaseModel
import uuid

class CancelOrderReq(BaseModel):
    Msg: str = "CancelOrderReq"
    OrigClOrderId: str
    ClOrderId: str

    @staticmethod
    def from_cl_ord_id(orig_cl_ord_id: str) -> "CancelOrderReq":
        """Creates a CancelOrderReq from an existing ClOrdId"""
        return CancelOrderReq(
            OrigClOrderId=orig_cl_ord_id,
            ClOrderId=orig_cl_ord_id
        )

