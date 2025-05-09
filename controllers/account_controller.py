from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from business_entities.account import Account
from data_access_layer.account_manager import AccountManager


class AccountController:
    def __init__(self, account_manager: AccountManager):
        self.router = APIRouter()
        self.templates = Jinja2Templates(directory="templates")
        self.account_manager = account_manager

        # Endpoint to render the account management UI
        self.router.get("/accounts", response_class=HTMLResponse)(self.account_list_view)

        # Endpoint to persist (insert/update) an account
        self.router.post("/accounts/save")(self.save_account)
        self.router.post("/accounts/delete")(self.delete_account)


    async def delete_account(self, account_number: str = Form(...)):
        self.account_manager.delete_account(account_number)
        return RedirectResponse(url="/accounts", status_code=303)

    async def account_list_view(self, request: Request):
        """Displays all accounts and a form to insert/update."""
        accounts = self.account_manager.get_all_accounts()
        return self.templates.TemplateResponse("manage_accounts.html", {
            "request": request,
            "accounts": accounts
        })

    async def save_account(self,
                           account_number: str = Form(...),
                           account_name: str = Form(...),
                           broker: str = Form(...)):
        """Handles form submission to insert or update an account."""
        account = Account(account_number=account_number,
                          account_name=account_name,
                          broker=broker)

        self.account_manager.persist_account(account)

        return RedirectResponse(url="/accounts", status_code=303)
