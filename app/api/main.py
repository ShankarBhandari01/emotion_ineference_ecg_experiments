from fastapi import FastAPI
from app.core.exception_handler import global_exception_handler
from app.routes.routes import router
from app.middlewares.LoggingMiddleware import LoggingMiddleware
from app.containers.containers import Container

# container
container = Container()
# app instance
app = FastAPI(title="FastAPI Inference API")
# add middlewares
app.add_middleware(LoggingMiddleware)
# add container
app.container = container
# add routers
app.include_router(router)

# Attach global exception handler
app.add_exception_handler(Exception, global_exception_handler)
