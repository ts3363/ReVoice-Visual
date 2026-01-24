import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse("""
    <html>
        <head><title>AMAC Test</title></head>
        <body>
            <h1>AMAC Therapy Test</h1>
            <p>If you can see this, the server is running!</p>
            <p>Go to <a href="http://localhost:8000">main app</a></p>
        </body>
    </html>
    """)

if __name__ == "__main__":
    print("=" * 50)
    print("TEST SERVER STARTING")
    print("=" * 50)
    print("Open your browser and go to: http://localhost:8001")
    print("If you see the test page, the server is working!")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8001)
