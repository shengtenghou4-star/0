#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from hou_lens_gradea_prospective16 import exact_replay

exact_replay.MANIFEST_SHA="67ea4a82fab65955dd3d0efbcc86b159c46eed0787a2963a3fc38dff63252553"
exact_replay.RESULT_CERT_SHA="3074cc6af3830060533d58d0f3b6aefc4c2d1dfaac4f257e2ab0590a8033b7e5"

if __name__=="__main__":
    exact_replay.main()
