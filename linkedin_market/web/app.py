"""Application FastAPI : API locale + frontend React compilé.

L'API peut postuler et envoyer des emails au nom de l'utilisateur : elle n'écoute que sur
127.0.0.1, vérifie ``Host``/``Origin`` et exige un jeton local (en-tête ``X-Local-Token`` ou
paramètre ``token``). Le jeton est injecté dans ``index.html`` : une page tierce ne peut pas le lire.
"""

from __future__ import annotations

import hmac
import re
import secrets
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .. import consent
from .tasks import TaskManager

FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"
_HOST = re.compile(r"^(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$")
DEV_ORIGINS = ("http://127.0.0.1:5174", "http://localhost:5174")
_LOCATIONS = {"body", "query", "path", "header", "cookie"}
_TYPE_MESSAGES = {
    "missing": "champ requis",
    "int_parsing": "doit être un entier",
    "int_type": "doit être un entier",
    "int_from_float": "doit être un entier",
    "float_parsing": "doit être un nombre",
    "float_type": "doit être un nombre",
    "bool_parsing": "doit valoir true ou false",
    "bool_type": "doit valoir true ou false",
    "string_type": "doit être un texte",
    "dict_type": "doit être un objet JSON",
}


def _field(loc: Sequence[Any]) -> str:
    """Dernier nom de champ de ``loc`` (hors emplacement ``body``/``query``/``path`` en tête et indices de liste).

    Un champ peut lui-même s'appeler ``body`` (``MessageBody.body``) : seul le premier élément est un emplacement.
    """
    fields = list(loc[1:]) if loc and loc[0] in _LOCATIONS else list(loc)
    for part in reversed(fields):
        if isinstance(part, str):
            return part
    return "corps de la requête"


def _choices(ctx: dict[str, Any]) -> str:
    """``"'email' or 'linkedin'"`` -> ``email, linkedin``."""
    expected = str(ctx.get("expected", ""))
    values = re.findall(r"'([^']*)'", expected)
    return ", ".join(values) if values else expected


def _message(error: dict[str, Any]) -> str:
    """Message français d'une erreur Pydantic (G-1) ; les messages de validateurs gardent leur ponctuation."""
    kind, ctx = error.get("type", ""), error.get("ctx") or {}
    if kind in _TYPE_MESSAGES:
        return _TYPE_MESSAGES[kind]
    if kind in ("literal_error", "enum"):
        return f"doit être l'un de : {_choices(ctx)}"
    if kind == "greater_than_equal":
        return f"doit être ≥ {ctx.get('ge')}"
    if kind == "less_than_equal":
        return f"doit être ≤ {ctx.get('le')}"
    if kind == "value_error":
        return str(error.get("msg", "")).removeprefix("Value error, ") or "valeur invalide"
    return "valeur invalide"


def validation_detail(errors: Sequence[dict[str, Any]]) -> str:
    """``<champ> : <message>`` joints par « ; » ; JSON mal formé : ``Corps JSON invalide.`` (contrat G-1, E-10)."""
    if any(error.get("type") == "json_invalid" for error in errors):
        return "Corps JSON invalide."
    return " ; ".join(f"{_field(error.get('loc', ()))} : {_message(error)}" for error in errors)


def create_app(*, tasks: TaskManager | None = None, token: str | None = "auto", dev: bool = False) -> FastAPI:
    """``token=None`` désactive le jeton (tests, développement avec le proxy Vite)."""
    app = FastAPI(
        title="linkedin_market",
        docs_url="/api/docs" if dev else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if dev else None,
    )
    app.state.tasks = tasks or TaskManager()
    app.state.token = secrets.token_urlsafe(24) if token == "auto" else token
    app.state.dev = dev

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"detail": validation_detail(exc.errors())}, status_code=422)

    @app.exception_handler(consent.SourceConsentRequired)
    async def consent_required(request: Request, exc: consent.SourceConsentRequired) -> JSONResponse:
        """Action navigateur sans consentement actif à la version courante des risques (I-16)."""
        return JSONResponse(
            {"detail": str(exc), "source": exc.source_code, "risk_version": exc.risk_version}, status_code=428
        )

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

    from .routers import applications, core, jobs, sources

    for module in (core, jobs, applications, sources):
        app.include_router(module.router, prefix="/api")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str, request: Request):
        if path.startswith("api/"):
            return JSONResponse({"detail": "Introuvable."}, status_code=404)
        if not FRONTEND_DIST.is_dir():
            return HTMLResponse(
                "<h1>Interface non compilée</h1><p>Exécute <code>npm install &amp;&amp; npm run build</code> "
                "dans <code>linkedin_market/web/frontend</code>, ou lance <code>npm run dev</code>.</p>",
                status_code=503,
            )
        candidate = (FRONTEND_DIST / path).resolve()
        if path and FRONTEND_DIST.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        page = (FRONTEND_DIST / "index.html").read_text(encoding="utf-8")
        token = request.app.state.token or ""
        page = page.replace("<!--LM_TOKEN-->", f'<meta name="lm-token" content="{token}">')
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    return app
