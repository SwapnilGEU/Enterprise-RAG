"""Put the project root on sys.path so `from src... import ...` works when a
script is run directly (`python scripts/run_index.py`) without installing the
package. Import this first in every script."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
