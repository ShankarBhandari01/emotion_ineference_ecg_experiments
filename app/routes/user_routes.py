from fastapi import APIRouter, Depends
from dependency_injector.wiring import inject, Provide
from app.controllers.user_controller import UserController
from app.containers.containers import Container

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/{user_id}")
@inject
async def get_user(
        user_id: int,
        controller: UserController = Depends(Provide[Container.user_controller])

):
    user = await controller.get_user_profile(user_id)
    return user
