from fastapi import FastAPI
from routers.models import models_router
from routers.items import items_router

app = FastAPI()

# Include routers
app.include_router(models_router)
app.include_router(items_router)
