#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hou_lens_p492_mixed16 import exact_replay

# P4.9.8 transport-only supersession. Historical replay logic and source blobs remain unchanged.
exact_replay.MANIFEST_SHA = "f55c4796b20a363702a05045fc835daebc81072f9c3b299ef461ea77fea35d5b"
exact_replay.RESULT_CERT_SHA = "3074cc6af3830060533d58d0f3b6aefc4c2d1dfaac4f257e2ab0590a8033b7e5"

if __name__ == "__main__":
    exact_replay.main()
