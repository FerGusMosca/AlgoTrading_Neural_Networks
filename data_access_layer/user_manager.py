import hashlib

import pyodbc
from business_entities.user import User
from itsdangerous import TimestampSigner

class UserManager:
    def __init__(self, connection_string: str, secret_key: str):
        """
        Initializes the UserManager instance with a database connection and secret key.

        :param connection_string: Connection string to the SQL Server database.
        :param secret_key: Secret key used for signing tokens.
        """
        self.connection = pyodbc.connect(connection_string)
        self.signer = TimestampSigner(secret_key)

    def authenticate_user(self, username: str, password: str) -> bool:
        """
        Verifies the provided username and password against the database.

        :param username: The username of the user attempting to log in.
        :param password: The password of the user attempting to log in (plaintext).
        :return: True if the credentials are valid (user exists and password matches), otherwise False.
        """
        with self.connection.cursor() as cursor:
            # Calls the stored procedure to verify the username and password
            params = (username, hashlib.sha256(password.encode('utf-8')).hexdigest())
            cursor.execute("{CALL AuthenticateUser (?,?)}", params)
            result = cursor.fetchone()  # Fetches the result from the stored procedure

            # If the result contains a value (1 means success)
            if result and result[0] == 1:
                return True  # Credentials are valid
            return False  # Credentials are invalid

    def get_user_by_username(self, username: str) -> User:
        """
        Retrieves a user from the database by their username.

        :param username: The username of the user to retrieve.
        :return: A User object if the user exists, otherwise None.
        """
        with self.connection.cursor() as cursor:
            # Queries the database to retrieve the user's hashed password by username
            cursor.execute("SELECT username, hashed_password FROM dbo.users WHERE username = ?", (username,))
            result = cursor.fetchone()  # Fetches the result (username and hashed password)

            if result:
                # Create a User object if the user exists
                user = User(username=result[0], hashed_password=result[1])
                return user  # Returns the User object
            return None  # Returns None if no user was found with the given username
