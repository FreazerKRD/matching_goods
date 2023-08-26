import uvicorn
from fastapi import FastAPI
from typing import Union

app = FastAPI()
dim = 72

def parse_string(vec: str) -> list[float]:
    l = vec.split(",")
    if len(l) != dim:
        return None
    return [float(el) for el in l]

@app.get("/")
def main() -> dict:
    return {"status": "OK", "Message": "Hello world!"}

@app.get("/knn")
def match(item: Union[str, None] = None) -> dict:
    if item is None:
        return {"status": "Fail", "Message": "No data recieved"}
    
    vec = parse_string(item)
    return {"status": "OK", "data": vec}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8031)