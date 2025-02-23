import pandas as pd
from starlette.responses import JSONResponse


class BaseController():

    def __init__(self):
        self.detailed_MTMS=[]
    async def get_chart_data(self):

        if not self.detailed_MTMS:
            return JSONResponse({
                "dates": [],
                "values": [],
                "message": "No data available. Please upload a file first."
            }, status_code=200)

        df = pd.DataFrame([{"date": obj.date, "MTM": obj.MTM} for obj in self.detailed_MTMS])

        if df.empty:
            return JSONResponse({
                "dates": [],
                "values": [],
                "message": "data is empty after processing. Check the uploaded file/key."
            }, status_code=200)

        df.sort_values(by="date", inplace=True)
        self.detailed_MTMS=[]#just one display
        return JSONResponse({
            "dates": df["date"].astype(str).tolist(),
            "values": df["MTM"].tolist()
        })

