#!/usr/bin/env python3
"""Create a harmless synthetic capsule for first-run validation."""

import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

CAPSULE_ID = "0123456789abcdef0123456789abcdef"
root = Path("sample-capsule")
root.mkdir(exist_ok=True)
stream = io.BytesIO()
script = b'#!/bin/bash\nset -euo pipefail\nout="$1"\nmkdir -p "$out"\nprintf "synthetic relay ok\\n" > "$out/result.txt"\n'
with tarfile.open(fileobj=stream, mode="w:gz") as archive:
    info = tarfile.TarInfo("run.sh")
    info.size = len(script)
    info.mode = 0o700
    archive.addfile(info, io.BytesIO(script))
payload = stream.getvalue()
(root / "payload.tar.gz.b64").write_text(base64.b64encode(payload).decode("ascii") + "\n", encoding="ascii")
(root / "manifest.json").write_text(json.dumps({
    "schema": 1,
    "capsule_id": CAPSULE_ID,
    "payload_sha256": hashlib.sha256(payload).hexdigest(),
    "timeout_minutes": 5,
}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(root)
