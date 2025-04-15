from fastApi import FastAPI


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    app = FastAPI()