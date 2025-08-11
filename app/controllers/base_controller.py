from fastapi import HTTPException
from typing import Any


class BaseController:
    def success(self, data: Any, message: str = "Success") -> dict:
        return {
            "code": 200,
            "status": "success",
            "message": message,
            "data": data
        }

    def error(self, message: str = "An error occurred", status_code: int = 400):
        raise HTTPException(status_code=status_code, detail=message)
