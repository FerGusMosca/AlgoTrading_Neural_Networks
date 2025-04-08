import pyodbc
from business_entities.account import Account

_ACCOUNT_NUMBER_IDX = 0
_ACCOUNT_NAME_IDX = 1
_BROKER_IDX = 2

class AccountManager:

    def __init__(self, connection_string: str):
        self.connection = pyodbc.connect(connection_string)

    def get_all_accounts(self) -> list[Account]:
        """Fetches all accounts from the database via stored procedure."""
        accounts = []
        with self.connection.cursor() as cursor:
            cursor.execute("{CALL GetAccounts}")
            for row in cursor:
                account = Account(
                    account_number=str(row[_ACCOUNT_NUMBER_IDX]),
                    account_name=str(row[_ACCOUNT_NAME_IDX]),
                    broker=str(row[_BROKER_IDX])
                )
                accounts.append(account)
        return accounts
