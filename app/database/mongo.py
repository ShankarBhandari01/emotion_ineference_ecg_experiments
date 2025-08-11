from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import os

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "my_mongo_db")

client: AsyncIOMotorClient = AsyncIOMotorClient(MONGO_URL)

def get_mongo_db(config) -> AsyncIOMotorDatabase:
    """Return the Motor AsyncIOMotorDatabase instance."""
    return client[MONGO_DB_NAME]
