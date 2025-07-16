from fastapi import APIRouter, Query, Body, HTTPException
from typing import Annotated
from pydantic import BaseModel, Field

items_router = APIRouter(prefix="/items", tags=["items"])

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


class FilterParams(BaseModel):
    limit: int = Field(default=10, ge=1, le=100)
    skip: int = Field(default=0, ge=0)


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


class User(BaseModel):
    username: str
    full_name: str | None = None


@items_router.get("/")
async def read_item(params: FilterParams):
    return fake_items_db[params.skip : params.skip + params.limit]


@items_router.post("/create")
async def create_item(
    item: Annotated[Item, Body()], params: Annotated[FilterParams, Query()]
):
    item_dict = item.model_dump()
    params_dict = params.model_dump()
    print(item_dict)
    print(item.name)
    print(params_dict)
    return item


@items_router.put("/{item_id}")
async def update_item(
    item_id: int, item: Item, user: User, params: Annotated[FilterParams, Query()]
):
    if item_id < 0:
        raise HTTPException(status_code=501, detail="Item ID must be positive")
    results = {"item_id": item_id, "item": item, "user": user}
    if params:
        results.update({"params": params})
    print(params)
    return results
