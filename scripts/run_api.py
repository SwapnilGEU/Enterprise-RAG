"""Start the API.

    python scripts/run_api.py                  # 127.0.0.1:8000, reload off
    python scripts/run_api.py --reload         # development
    python scripts/run_api.py --host 0.0.0.0   # reachable from the network

Equivalent to `uvicorn api.main:app`, with the flags spelled out. Use whichever
you prefer; Docker will call uvicorn directly.

Workers: leave it at 1 for now. Each worker is a separate process with its own
Ollama client, its own Qdrant client and its own compiled agent, all competing
for one 6GB card — more workers make generation slower, not faster. Raise it
only once the generator is hosted somewhere else.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        # Our own JSON formatter is installed by api.main; uvicorn's default
        # dictConfig would replace it and we would lose request_id and trace_id.
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
