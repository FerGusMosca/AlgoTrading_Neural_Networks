import pandas as pd
from starlette.responses import JSONResponse


class BaseController():

    def __init__(self):
        self.detailed_MTMS=[]

    async def get_chart_data(self):
        """Returns the chart data if available."""

        if not self.detailed_MTMS:
            return JSONResponse({
                "dates": [],
                "values": [],
                "message": "No data available. Please load data first."
            }, status_code=200)

        df = pd.DataFrame([{"date": obj.date, "MTM": obj.MTM} for obj in self.detailed_MTMS])

        if df.empty:
            return JSONResponse({
                "dates": [],
                "values": [],
                "message": "Data is empty after processing. Check the uploaded file/key."
            }, status_code=200)

        df.sort_values(by="date", inplace=True)

        return JSONResponse({
            "dates": df["date"].astype(str).tolist(),
            "values": df["MTM"].tolist()
        })


