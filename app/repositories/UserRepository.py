class UserRepository:
    def __init__(self, db):
        self.db_session = db

    async def get_user(self, user_id: int):
        return {"id": user_id, "name": "Neo"}
