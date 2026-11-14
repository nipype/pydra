"""Per-item overhead of a trivial task; kept runnable, the code is in ``benchmark.py``.

Usage: ``python pydra_overhead.py [kind:worker:n ...]`` (``kind`` is ``split`` or ``chain``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import benchmark  # noqa: E402

if __name__ == "__main__":
    sys.exit(benchmark.main(["overhead", *sys.argv[1:]]))
