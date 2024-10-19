from fastapi import FastAPI, HTTPException
app = FastAPI()

@app.get("/")
async def index():
    return "Hello world"