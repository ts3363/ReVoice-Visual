import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="AMAC Test")

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root():
    return {"message": "AMAC Therapy System is working!"}

@app.get("/test")
async def test():
    return HTMLResponse("<h1>Test Page</h1><p>Server is running!</p>")

if __name__ == "__main__":
    print("Starting test server on http://localhost:8001")
    print("Press Ctrl+C to stop")
    uvicorn.run(app, host="0.0.0.0", port=8001)
