"""
Analyse d'emails de projets — point d'entrée historique.
La logique vit dans le package : services/email_analyzer/email_analyzer/
"""

import sys
from pathlib import Path

_pkg = Path(__file__).resolve().parent / "services" / "email_analyzer"
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from email_analyzer.legacy_cli import main

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
