# controllers/auth_middleware.py

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse
from itsdangerous import TimestampSigner, BadSignature


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, secret_key: str, exempt_paths=None):  # 👈 aceptar exempt_paths
        super().__init__(app)
        self.signer = TimestampSigner(secret_key)
        self.exempt_paths = exempt_paths or []

    async def dispatch(self, request: Request, call_next):
        print(f"[MIDDLEWARE] Incoming path: {request.url.path}")  # 👈 log para debug

        for path in self.exempt_paths:
            if request.url.path.startswith(path):
                return await call_next(request)

        token = request.cookies.get("session")
        if token:
            try:
                self.signer.unsign(token, max_age=3600)
                return await call_next(request)
            except BadSignature:
                pass

        return RedirectResponse(url="/login")
