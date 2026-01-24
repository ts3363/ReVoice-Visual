from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
import os

app = FastAPI(title="AMAC Therapy")

# Create directories if they don't exist
os.makedirs("app/static/css", exist_ok=True)
os.makedirs("app/static/js", exist_ok=True)
os.makedirs("app/templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("therapy.html", {"request": request})

@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "AMAC Therapy"}

if __name__ == "__main__":
    print("Starting AMAC Therapy System...")
    print("Open: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
