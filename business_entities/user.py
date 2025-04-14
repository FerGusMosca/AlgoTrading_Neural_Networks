# business_entities/user.py

class User:
    def __init__(self, username: str, hashed_password: str):
        self.username = username
        self.hashed_password = hashed_password

    def __repr__(self):
        return f"User(username={self.username})"
