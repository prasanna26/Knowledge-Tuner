from fastapi import FastAPI
import uvicorn
import uvicorn.server

app = FastAPI(title="My FastAPI App", version="1.0.0")
@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0", port=8080, reload=True)