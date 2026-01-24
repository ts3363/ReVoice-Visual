from fastapi import FastAPI
import uvicorn
import time

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "AMAC Therapy", "status": "running"}

@app.get("/test")
def test():
    return {"message": "Server is alive", "timestamp": time.time()}

if __name__ == "__main__":
    print("=" * 60)
    print("AMAC THERAPY TEST SERVER")
    print("=" * 60)
    print("Server is starting on: http://127.0.0.1:8002")
    print("Keep this window open!")
    print("Open a web browser and go to: http://localhost:8002")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # This will keep the server running
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
