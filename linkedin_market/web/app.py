"""Application FastAPI : API locale + frontend React compilé.

L'API peut postuler et envoyer des emails au nom de l'utilisateur : elle n'écoute que sur
127.0.0.1, vérifie ``Host``/``Origin`` et exige un jeton local (en-tête ``X-Local-Token`` ou
paramètre ``token``). Le jeton est injecté dans ``index.html`` : une page tierce ne peut pas le lire.
"""

from __future__ import annotations

import hmac
import re
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .tasks import TaskManager

FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"
_HOST = re.compile(r"^(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$")
DEV_ORIGINS = ("http://127.0.0.1:5174", "http://localhost:5174")


def create_app(*, tasks: TaskManager | None = None, token: str | None = "auto", dev: bool = False) -> FastAPI:
    """``token=None`` désactive le jeton (tests, développement avec le proxy Vite)."""
    app = FastAPI(title="linkedin_market", docs_url="/api/docs" if dev else None, redoc_url=None, openapi_url="/api/openapi.json" if dev else None)
    app.state.tasks = tasks or TaskManager()
    app.state.token = secrets.token_urlsafe(24) if token == "auto" else token
    app.state.dev = dev

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if not _HOST.match(request.headers.get("host", "")):
            return JSONResponse({"detail": "Hôte non autorisé."}, status_code=403)
        origin = request.headers.get("origin")
        if origin and request.method not in ("GET", "HEAD", "OPTIONS"):
            same = origin.split("://", 1)[-1] == request.headers.get("host", "")
            if not same and not (dev and origin in DEV_ORIGINS):
                return JSONResponse({"detail": "Origine non autorisée."}, status_code=403)
        expected = request.app.state.token
        if expected and request.url.path.startswith("/api/") and request.url.path != "/api/health":
            given = request.headers.get("x-local-token") or request.query_params.get("token") or ""
            if not hmac.compare_digest(given, expected):
                return JSONResponse({"detail": "Jeton local manquant ou invalide."}, status_code=401)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    from .routers import applications, core, jobs

    for module in (core, jobs, applications):
        app.include_router(module.router, prefix="/api")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str, request: Request):
        if path.startswith("api/"):
            return JSONResponse({"detail": "Introuvable."}, status_code=404)
        if not FRONTEND_DIST.is_dir():
            return HTMLResponse(
                "<h1>Interface non compilée</h1><p>Exécute <code>npm install &amp;&amp; npm run build</code> "
                "dans <code>linkedin_market/web/frontend</code>, ou lance <code>npm run dev</code>.</p>", status_code=503)
        candidate = (FRONTEND_DIST / path).resolve()
        if path and FRONTEND_DIST.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        page = (FRONTEND_DIST / "index.html").read_text(encoding="utf-8")
        token = request.app.state.token or ""
        page = page.replace("<!--LM_TOKEN-->", f'<meta name="lm-token" content="{token}">')
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    return app
