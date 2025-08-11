class UserNotFound(Exception):
    def __init__(self, user_id: int):
        self.message = f"User with ID {user_id} not found"
        super().__init__(self.message)


class ForbiddenAccess(Exception):
    def __init__(self, message="Forbidden"):
        self.message = message
        super().__init__(self.message)


class BadRequest(Exception):
    def __init__(self, message="Bad request"):
        self.message = message
        super().__init__(self.message)
