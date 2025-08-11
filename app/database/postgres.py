from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager
import logging

DATABASE_URL = "postgresql://postgres:@localhost:5432/mydb"

# Make engine creation lazy to avoid import-time failures
_engine = None
_SessionLocal = None
Base = declarative_base()

logger = logging.getLogger(__name__)

def get_engine():
    """Lazy initialization of database engine"""
    global _engine
    if _engine is None:
        try:
            _engine = create_engine(DATABASE_URL)
            logger.info("PostgreSQL engine created successfully")
        except ImportError as e:
            logger.error(f"PostgreSQL driver not available: {e}")
            raise RuntimeError(
                "PostgreSQL driver (psycopg2) is not properly installed. "
                "Please install PostgreSQL client libraries or use an alternative database."
            )
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise
    return _engine

def get_session_local():
    """Lazy initialization of session factory"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal

@contextmanager
def get_postgres_db():
    """Context manager for database sessions"""
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()

def get_postgres_db_session():
    """Returns a database session for dependency injection"""
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()