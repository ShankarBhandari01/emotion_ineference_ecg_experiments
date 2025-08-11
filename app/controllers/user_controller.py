from app.services.user_service import UserService
from app.controllers.base_controller import BaseController
from app.core.exceptions import UserNotFound


class UserController(BaseController):
    def __init__(self, service: UserService):
        self.service = service

    async def get_user_profile(self, user_id: int):
        user = await self.service.get_profile(user_id)
        if not user:
            raise UserNotFound(user_id)
        return self.success(user, message="User profile fetched")
