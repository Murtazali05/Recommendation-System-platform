from typing import Optional

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.get("/recommendations/item/{item_id}/user/{user_id}/model/")
def get_recommendation(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}