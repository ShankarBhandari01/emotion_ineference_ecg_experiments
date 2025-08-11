from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.exceptions import UserNotFound, ForbiddenAccess
import logging
import traceback


async def global_exception_handler(request: Request, exc: Exception):
    # Log all exceptions
    logging.error(f"Unhandled exception: {str(exc)}")
    traceback.print_exc()

    if isinstance(exc, UserNotFound):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": exc.message},
        )

    elif isinstance(exc, StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "message": exc.detail},
        )

    elif isinstance(exc, RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "message": "Validation error",
                "details": exc.errors(),
            },
        )

    elif isinstance(exc, ForbiddenAccess):
        return JSONResponse(status_code=403, content={"status": "error", "message": exc.message})

    else:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal Server Error"},
        )
