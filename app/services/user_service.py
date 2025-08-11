from app.repositories import UserRepository


class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    async def get_profile(self, user_id: int):
        return await self.repo.get_user(user_id)
