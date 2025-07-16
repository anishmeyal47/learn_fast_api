from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated, Optional, Callable
import asyncio
import time
from functools import lru_cache
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lazy_dependencies_router = APIRouter(prefix="/lazy", tags=["lazy-dependencies"])


# Pydantic models
class User(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool = True


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "myapp"
    username: str = "user"
    password: str = "password"


class ExternalServiceConfig(BaseModel):
    api_key: str
    base_url: str
    timeout: int = 30


# Simulated database connection
class DatabaseConnection:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._connection = None
        self._is_connected = False

    async def connect(self):
        if not self._is_connected:
            logger.info(
                f"Connecting to database at {self.config.host}:{self.config.port}"
            )
            # Simulate connection delay
            await asyncio.sleep(0.1)
            self._connection = f"postgresql://{self.config.username}@{self.config.host}:{self.config.port}/{self.config.database}"
            self._is_connected = True
            logger.info("Database connected successfully")
        return self._connection

    async def disconnect(self):
        if self._is_connected:
            logger.info("Disconnecting from database")
            self._connection = None
            self._is_connected = False

    async def execute_query(self, query: str):
        await self.connect()
        logger.info(f"Executing query: {query}")
        # Simulate query execution
        await asyncio.sleep(0.05)
        return {"result": f"Query '{query}' executed successfully"}


# Simulated external service
class ExternalService:
    def __init__(self, config: ExternalServiceConfig):
        self.config = config
        self._client = None
        self._is_initialized = False

    async def initialize(self):
        if not self._is_initialized:
            logger.info(
                f"Initializing external service client for {self.config.base_url}"
            )
            # Simulate initialization delay
            await asyncio.sleep(0.2)
            self._client = f"HTTPClient({self.config.base_url})"
            self._is_initialized = True
            logger.info("External service client initialized")
        return self._client

    async def make_request(self, endpoint: str, data: Optional[dict] = None):
        await self.initialize()
        logger.info(f"Making request to {endpoint}")
        # Simulate API call
        await asyncio.sleep(0.1)
        return {"endpoint": endpoint, "data": data, "status": "success"}


# Configuration management with lazy loading
class ConfigManager:
    def __init__(self):
        self._db_config = None
        self._external_config = None
        self._cache = {}

    def get_db_config(self) -> DatabaseConfig:
        if self._db_config is None:
            logger.info("Loading database configuration...")
            # Simulate loading from environment or file
            self._db_config = DatabaseConfig(
                host="localhost",
                port=5432,
                database="myapp",
                username="user",
                password="password",
            )
            logger.info("Database configuration loaded")
        return self._db_config

    def get_external_config(self) -> ExternalServiceConfig:
        if self._external_config is None:
            logger.info("Loading external service configuration...")
            # Simulate loading from environment or file
            self._external_config = ExternalServiceConfig(
                api_key="your-api-key-here",
                base_url="https://api.external-service.com",
                timeout=30,
            )
            logger.info("External service configuration loaded")
        return self._external_config

    def get_cached_value(self, key: str, factory: Callable):
        if key not in self._cache:
            logger.info(f"Computing value for key: {key}")
            self._cache[key] = factory()
        return self._cache[key]


# Global instances (lazy loaded)
_config_manager = None
_db_connection = None
_external_service = None


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        logger.info("Creating new ConfigManager instance")
        _config_manager = ConfigManager()
    return _config_manager


def get_db_connection() -> DatabaseConnection:
    global _db_connection
    if _db_connection is None:
        logger.info("Creating new DatabaseConnection instance")
        config_manager = get_config_manager()
        db_config = config_manager.get_db_config()
        _db_connection = DatabaseConnection(db_config)
    return _db_connection


def get_external_service() -> ExternalService:
    global _external_service
    if _external_service is None:
        logger.info("Creating new ExternalService instance")
        config_manager = get_config_manager()
        external_config = config_manager.get_external_config()
        _external_service = ExternalService(external_config)
    return _external_service


# Dependency injection with lazy loading
async def get_db_dependency() -> DatabaseConnection:
    """Lazy loading database dependency"""
    db = get_db_connection()
    return db


async def get_external_service_dependency() -> ExternalService:
    """Lazy loading external service dependency"""
    service = get_external_service()
    return service


# LRU Cache example for expensive computations
@lru_cache(maxsize=128)
def expensive_computation(n: int) -> int:
    """Simulate expensive computation with caching"""
    logger.info(f"Computing expensive operation for n={n}")
    time.sleep(0.1)  # Simulate computation time
    return n * n * n


# Async lazy loading with caching
class AsyncCache:
    def __init__(self):
        self._cache = {}

    async def get_or_compute(self, key: str, factory: Callable, ttl: int = 300):
        """Get cached value or compute and cache it"""
        current_time = time.time()

        if key in self._cache:
            value, timestamp = self._cache[key]
            if current_time - timestamp < ttl:
                logger.info(f"Cache hit for key: {key}")
                return value
            else:
                logger.info(f"Cache expired for key: {key}")
                del self._cache[key]

        logger.info(f"Computing new value for key: {key}")
        value = await factory()
        self._cache[key] = (value, current_time)
        return value


# Global async cache instance
_async_cache = AsyncCache()


# API endpoints demonstrating lazy loading
@lazy_dependencies_router.get("/database/query")
async def execute_database_query(
    db: Annotated[DatabaseConnection, Depends(get_db_dependency)],
    query: str = "SELECT * FROM users",
):
    """Execute a database query with lazy-loaded database connection"""
    result = await db.execute_query(query)
    return {"message": "Database query executed", "result": result}


@lazy_dependencies_router.get("/external/api-call")
async def make_external_api_call(
    service: Annotated[ExternalService, Depends(get_external_service_dependency)],
    endpoint: str = "/users",
):
    """Make an external API call with lazy-loaded service"""
    result = await service.make_request(endpoint)
    return {"message": "External API call made", "result": result}


@lazy_dependencies_router.get("/config/database")
async def get_database_config(
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)]
):
    """Get database configuration (lazy loaded)"""
    config = config_manager.get_db_config()
    return {"message": "Database configuration retrieved", "config": config}


@lazy_dependencies_router.get("/config/external")
async def get_external_config(
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)]
):
    """Get external service configuration (lazy loaded)"""
    config = config_manager.get_external_config()
    return {"message": "External service configuration retrieved", "config": config}


@lazy_dependencies_router.get("/compute/{n}")
async def compute_with_cache(n: int):
    """Demonstrate LRU cache for expensive computations"""
    result = expensive_computation(n)
    return {"input": n, "result": result, "message": "Computation completed"}


@lazy_dependencies_router.get("/async-cache/{key}")
async def async_cached_computation(key: str):
    """Demonstrate async caching with lazy loading"""

    async def expensive_async_operation():
        logger.info(f"Performing expensive async operation for key: {key}")
        await asyncio.sleep(0.5)  # Simulate async work
        return f"Computed result for {key}: {hash(key) % 1000}"

    result = await _async_cache.get_or_compute(key, expensive_async_operation)
    return {"key": key, "result": result, "message": "Async computation completed"}


@lazy_dependencies_router.get("/users/{user_id}")
async def get_user_with_dependencies(
    user_id: int,
    db: Annotated[DatabaseConnection, Depends(get_db_dependency)],
    external_service: Annotated[
        ExternalService, Depends(get_external_service_dependency)
    ],
):
    """Get user data using multiple lazy-loaded dependencies"""

    # Simulate fetching user from database
    user_query = f"SELECT * FROM users WHERE id = {user_id}"
    db_result = await db.execute_query(user_query)

    # Simulate fetching additional data from external service
    external_result = await external_service.make_request(f"/users/{user_id}/profile")

    # Simulate user data
    user = User(
        id=user_id,
        username=f"user{user_id}",
        email=f"user{user_id}@example.com",
        is_active=True,
    )

    return {
        "user": user,
        "database_result": db_result,
        "external_service_result": external_result,
        "message": "User data retrieved with lazy-loaded dependencies",
    }


@lazy_dependencies_router.post("/reset-cache")
async def reset_cache():
    """Reset all caches (useful for testing)"""
    global _async_cache
    _async_cache = AsyncCache()
    expensive_computation.cache_clear()
    logger.info("All caches reset")
    return {"message": "All caches have been reset"}


@lazy_dependencies_router.get("/health")
async def health_check():
    """Health check endpoint to verify lazy loading is working"""
    return {
        "status": "healthy",
        "lazy_loading": "enabled",
        "message": "Lazy dependencies are working correctly",
    }
