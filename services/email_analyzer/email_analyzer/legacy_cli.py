"""CLI complète rétrocompatible avec l'ancien script monolithique."""

import argparse
import json
import logging
import sys
from time import perf_counter

from dotenv import load_dotenv

from .config import (
    ASSISTANT_PROVIDER_NONE,
    ASSISTANT_PROVIDER_OPENAI,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_BODY_CHARS,
    DEFAULT_OPENAI_MAX_INPUT_CHARS,
    DEFAULT_OPENAI_MAX_TOKENS,
    ECONOMY_OPENAI_BODY_CHARS,
    ECONOMY_OPENAI_MAX_INPUT_CHARS,
    ECONOMY_OPENAI_MAX_TOKENS,
    VALID_ASSISTANT_PROVIDERS,
    VALID_PERIODS,
)
from .period import imap_days_back_for_period
from .project_mail import EmailProjectAnalyzer


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyse intelligente des emails de projets avec IA")
    parser.add_argument("--email", required=True, help="Adresse email")
    parser.add_argument("--password", required=True, help="Mot de passe")
    parser.add_argument("--projects", required=True, nargs="+", help="Liste des projets à analyser")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Nombre de jours pour la fenêtre IMAP SINCE (ignoré si --period est défini)",
    )
    parser.add_argument(
        "--period",
        choices=sorted(VALID_PERIODS),
        default=None,
        metavar="PÉRIODE",
        help=(
            "Filtre temporel sur la date des messages: today, yesterday, "
            "3 (3 derniers jours), 7, 11. Remplace --days pour la recherche IMAP."
        ),
    )
    parser.add_argument(
        "--assistant-provider",
        choices=sorted(VALID_ASSISTANT_PROVIDERS),
        default=ASSISTANT_PROVIDER_OPENAI,
        help=(
            "Résumé assistant : openai (OPENAI_API_KEY), gemini (GEMINI_API_KEY ou GOOGLE_API_KEY, "
            "https://aistudio.google.com), none pour désactivé. Ignoré si --no-openai."
        ),
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help=f"Modèle Gemini pour --assistant-provider gemini (défaut: {DEFAULT_GEMINI_MODEL})",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help=(
            "Modèle OpenAI pour le résumé assistant (OPENAI_API_KEY). Comptes à crédit limité : "
            "préférez --openai-economy et/ou un modèle moins coûteux si votre compte le permet."
        ),
    )
    parser.add_argument(
        "--openai-economy",
        action="store_true",
        help=(
            "Réduit fortement tokens entrée/sortie (corpus court, max_tokens bas) pour limiter "
            "le coût et mieux tenir dans un quota faible ; ignore les options --openai-max-* ci-dessous."
        ),
    )
    parser.add_argument(
        "--openai-max-tokens",
        type=int,
        default=DEFAULT_OPENAI_MAX_TOKENS,
        metavar="N",
        help=f"Plafond de tokens générés par le modèle (défaut: {DEFAULT_OPENAI_MAX_TOKENS}, ignoré avec --openai-economy)",
    )
    parser.add_argument(
        "--openai-max-input-chars",
        type=int,
        default=DEFAULT_OPENAI_MAX_INPUT_CHARS,
        metavar="N",
        help=(
            f"Taille max du texte agrégé envoyé au modèle en caractères (défaut: "
            f"{DEFAULT_OPENAI_MAX_INPUT_CHARS}, ignoré avec --openai-economy)"
        ),
    )
    parser.add_argument(
        "--openai-body-chars",
        type=int,
        default=DEFAULT_OPENAI_BODY_CHARS,
        metavar="N",
        help=(
            f"Max de caractères du corps par email dans le corpus (défaut: "
            f"{DEFAULT_OPENAI_BODY_CHARS}, ignoré avec --openai-economy)"
        ),
    )
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Ne pas appeler l'API OpenAI pour le résumé assistant",
    )
    parser.add_argument("--server", default="mail.mediasoftci.net", help="Serveur IMAP")
    parser.add_argument("--port", type=int, default=993, help="Port IMAP")
    parser.add_argument("--no-ssl", action="store_true", help="Désactiver SSL")
    parser.add_argument("--output", help="Fichier de sortie JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
    parser.add_argument(
        "--max-deep-emails",
        type=int,
        default=20,
        help="Nombre max d'emails analysés en profondeur par projet",
    )
    parser.add_argument(
        "--cache-file",
        default="email_analysis_cache.json",
        help="Fichier de cache local pour éviter les retraitements",
    )

    args = parser.parse_args()

    if args.openai_economy:
        openai_max_tokens = ECONOMY_OPENAI_MAX_TOKENS
        openai_max_input_chars = ECONOMY_OPENAI_MAX_INPUT_CHARS
        openai_body_chars = ECONOMY_OPENAI_BODY_CHARS
        logging.info(
            "Mode OpenAI économique: max_tokens=%s, max_input_chars=%s, body_chars=%s",
            openai_max_tokens,
            openai_max_input_chars,
            openai_body_chars,
        )
    else:
        openai_max_tokens = args.openai_max_tokens
        openai_max_input_chars = args.openai_max_input_chars
        openai_body_chars = args.openai_body_chars

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    analyzer = EmailProjectAnalyzer(
        args.email,
        args.password,
        args.server,
        args.port,
        max_deep_emails=args.max_deep_emails,
        cache_file=args.cache_file,
    )

    if args.no_ssl:
        logging.warning("--no-ssl : non appliqué dans cette version (connexion inchangée).")

    try:
        print("🤖 Démarrage de l'analyse intelligente des emails...")

        if not analyzer.connect():
            print("❌ Échec de la connexion")
            sys.exit(1)

        imap_days = imap_days_back_for_period(args.period) if args.period is not None else args.days
        if args.period:
            logging.info("Période: %s (IMAP days_back=%s)", args.period, imap_days)

        search_start = perf_counter()
        project_data = analyzer.search_project_emails(args.projects, imap_days)
        analyzer.record_timing("phase_recherche_imap", perf_counter() - search_start)

        if not project_data:
            print("❌ Aucun email trouvé")
            sys.exit(1)

        if args.period:
            project_data = analyzer.apply_period_filter(project_data, args.period)
            total_in_period = sum(len(d.get("emails", [])) for d in project_data.values())
            if total_in_period == 0:
                print(f"❌ Aucun email dans la période sélectionnée ({args.period})")
                sys.exit(1)

        print("🧠 Analyse IA en cours...")
        ai_start = perf_counter()
        assistant_provider = ASSISTANT_PROVIDER_NONE if args.no_openai else args.assistant_provider

        intelligent_summary = analyzer.generate_intelligent_summary(
            project_data,
            assistant_provider=assistant_provider,
            openai_model=args.openai_model,
            gemini_model=args.gemini_model,
            openai_max_tokens=openai_max_tokens,
            openai_max_input_chars=openai_max_input_chars,
            openai_body_chars=openai_body_chars,
        )
        analyzer.record_timing("phase_analyse_ia", perf_counter() - ai_start)

        print("\n" + "=" * 80)
        print("🤖 RAPPORT INTELLIGENT D'ANALYSE DES PROJETS")
        print("=" * 80)

        for project_name, analysis in intelligent_summary.items():
            for line in analyzer.format_project_report(project_name, analysis):
                print(line)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(intelligent_summary, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 Rapport sauvegardé: {args.output}")

        print("\n⏱️ BILAN PERFORMANCE")
        print("-" * 80)
        for step in analyzer.get_top_timing_steps(3):
            print(
                f"   • {step['étape']}: {step['total_s']}s "
                f"(runs: {step['runs']}, moyenne: {step['moyenne_s']}s)"
            )

        print("\n✅ Analyse intelligente terminée!")

    except Exception as e:
        logging.error("Erreur: %s", e)
        sys.exit(1)
    finally:
        analyzer.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
