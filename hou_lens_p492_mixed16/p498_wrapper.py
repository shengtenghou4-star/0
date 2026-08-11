#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hou_lens_p492_mixed16 import exact_replay

# P4.9.8 coordinate-byte recovery: only one exposed-development control centroid is
# corrected to the exact ESO KiDS DR5 RAJ2000/DECJ2000 row; historical imaging logic is unchanged.
exact_replay.MANIFEST_SHA = "46fce794408c2f3519acc4aa8ca61c581a69c42b643c98d096b377db3ad6f936"
exact_replay.RESULT_CERT_SHA = "3074cc6af3830060533d58d0f3b6aefc4c2d1dfaac4f257e2ab0590a8033b7e5"

if __name__ == "__main__":
    exact_replay.main()
