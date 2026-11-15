"""One failed split element and its siblings; kept runnable, the code is in ``benchmark.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import benchmark  # noqa: E402

if __name__ == "__main__":
    sys.exit(benchmark.main(["partial-failure"]))
