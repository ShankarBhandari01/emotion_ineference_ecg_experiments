from fastapi import APIRouter
from app.routes.prediction_router import router as prediction_router
from app.routes.user_routes import router as user_router


router = APIRouter()

# Register all routers
router.include_router(prediction_router)
router.include_router(user_router)


