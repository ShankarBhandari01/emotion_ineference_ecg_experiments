import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.logger import logger

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} completed in {duration:.4f}s")
        return response
