"""``py -3 run.py`` == ``py -3 -m flybrain`` (SPEC section c.28 / i.5).

Adds ``backend/`` to ``sys.path`` so the server starts without ``pip install -e backend``, then hands
over to ``flybrain.__main__.main``. Nothing here depends on the current working directory (SPEC 0.1).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("PYTHONUTF8", "1")

from flybrain.__main__ import main  # noqa: E402 - sys.path must be set first

if __name__ == "__main__":
    raise SystemExit(main())
