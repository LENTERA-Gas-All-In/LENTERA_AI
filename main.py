from fastapi import FastAPI, HTTPException, status
from service.market_insight import MarketInsightService

app = FastAPI()
market_insight_service = MarketInsightService()

@app.get("/")
async def index():
    return "Hello world"

@app.get("/market/insight")
async def get_market_insight(product: str, period:str="2023,2022,2021", n:int=5):
    try:
        market_insight = market_insight_service.get_market_insight(product,period,n) 
        return {"status":status.HTTP_200_OK, "data": market_insight}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))