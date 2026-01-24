from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
import os

# Create necessary directories
os.makedirs("app/static/css", exist_ok=True)
os.makedirs("app/static/js", exist_ok=True)
os.makedirs("app/templates", exist_ok=True)

app = FastAPI(title="AMAC Therapy", version="1.0")

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("therapy.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "ok", "service": "AMAC Therapy"}

if __name__ == "__main__":
    print("🚀 AMAC Therapy System Starting...")
    print("🌐 Open your browser and go to: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
