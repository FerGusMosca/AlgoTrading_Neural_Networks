import pandas as pd
from fastapi import HTTPException
from starlette.responses import JSONResponse


class BaseController():

    def __init__(self):
        self.detailed_MTMS=[]

    async def get_chart_data_w_mmov(self):
        try:
            if not hasattr(self, 'detailed_MTMS'):
                raise HTTPException(status_code=400, detail="No ETF data loaded yet.")

            # ✅ Convert dates to ISO 8601 string format for JSON
            dates = [
                item.date.isoformat() if hasattr(item.date, "isoformat") else str(item.date)
                for item in self.detailed_MTMS
            ]

            values = [item.MTM for item in self.detailed_MTMS]
            ma_values = getattr(self, "moving_avg_values", [])

            return JSONResponse(content={
                "dates": dates,
                "values": values,
                "moving_avg": ma_values
            })

        except Exception as e:
            #self.logger.do_log(f"❌ Error in get_chart_data: {str(e)}", MessageType.ERROR)
            raise HTTPException(status_code=500, detail=f"Error retrieving chart data: {str(e)}")
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


