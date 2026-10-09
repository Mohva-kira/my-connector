"""``python -m linkedin_market.web`` : lance l'interface de gestion sur http://127.0.0.1:8765."""

from __future__ import annotations

import argparse
import logging
import os
import webbrowser

import uvicorn

from .. import config
from ..db.bootstrap import prepare_database
from .app import create_app

PORT = 8765


def main() -> None:
    parser = argparse.ArgumentParser(prog="linkedin_market.web", description=__doc__)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--dev", action="store_true", help="sans jeton local, pour le serveur Vite (npm run dev) ; docs API activées"
    )
    parser.add_argument("--no-open", action="store_true", help="ne pas ouvrir le navigateur")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config.ensure_data_dirs()
    prepare_database()  # avant create_app : les tâches interrompues sont déjà en échec
    token = None if args.dev else os.environ.get("LINKEDIN_UI_TOKEN") or "auto"
    app = create_app(token=token, dev=args.dev)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\nInterface : {url}\nAPI uniquement sur 127.0.0.1. Ctrl+C pour arrêter.\n")
    if not args.no_open and not args.dev:
        webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
