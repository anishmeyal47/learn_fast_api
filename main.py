from fastapi import FastAPI
from routers.models import models_router
from routers.items import items_router
from routers.lazy_dependencies import lazy_dependencies_router

app = FastAPI()

# Include routers
app.include_router(models_router)
app.include_router(items_router)
app.include_router(lazy_dependencies_router)
