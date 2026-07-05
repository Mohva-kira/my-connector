"""python -m email_analyzer → CLI JSON."""

import logging

from .cli import main

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    main()
