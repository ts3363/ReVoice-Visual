from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World", "AMAC": "Therapy"}

if __name__ == "__main__":
    print("=" * 50)
    print("Starting simple test server...")
    print("Open: http://127.0.0.1:8002")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8002)
