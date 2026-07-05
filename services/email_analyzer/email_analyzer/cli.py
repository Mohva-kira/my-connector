"""Entrée CLI JSON pour pont Node ou scripts (stdout = rapport structuré)."""

import argparse
import json
import sys

from .analyzer import EmailProcessor
from .config import (
    ASSISTANT_PROVIDER_NONE,
    ASSISTANT_PROVIDER_OPENAI,
    DEFAULT_GEMINI_MODEL,
    VALID_ASSISTANT_PROVIDERS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse emails projet -> JSON sur stdout")
    parser.add_argument("--project", required=True, help="Nom du projet (filtre)")
    parser.add_argument(
        "--period",
        default=None,
        help="today, yesterday, 3, 7, 11 (optionnel)",
    )
    parser.add_argument("--days", type=int, default=30, help="Fenêtre IMAP si pas de --period")
    parser.add_argument(
        "--assistant-provider",
        choices=sorted(VALID_ASSISTANT_PROVIDERS),
        default=ASSISTANT_PROVIDER_OPENAI,
    )
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--no-llm", action="store_true", help="Équivalent assistant none")
    args = parser.parse_args()

    provider = ASSISTANT_PROVIDER_NONE if args.no_llm else args.assistant_provider

    proc = EmailProcessor()
    result = proc.process_latest_emails(
        args.project,
        period=args.period,
        days=args.days,
        assistant_provider=provider,
        openai_model=args.openai_model,
        gemini_model=args.gemini_model,
    )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
