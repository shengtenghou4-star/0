from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

URL = "https://osf.io/download/evhrg/"
FILENAME = "AML310_moving.tar.gz"
EXPECTED_BYTES = 348_444_164
EXPECTED_SHA256 = "144126ee9a49d311c3393deea434e1a0963d55de35318e25d98d48f9c175250a"
FORBIDDEN = {"AML310_transition.tar.gz", "AML32_chip.tar.gz"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


root = Path("s40_artifacts")
raw = root / "raw"
raw.mkdir(parents=True, exist_ok=True)
assert not any(name in URL for name in FORBIDDEN)
target = raw / FILENAME
urllib.request.urlretrieve(URL, target)
observed_bytes = target.stat().st_size
observed_sha = sha256(target)
if observed_bytes != EXPECTED_BYTES or observed_sha != EXPECTED_SHA256:
    raise RuntimeError((observed_bytes, observed_sha))
manifest = {
    "schema": "bio-001-s40-source-manifest-v1",
    "target": {"filename": FILENAME, "url": URL, "bytes": observed_bytes, "sha256": observed_sha},
    "development_source_accessed": False,
    "sealed_holdout_requested": False,
    "sealed_holdout_listed": False,
    "sealed_holdout_downloaded": False,
    "sealed_holdout_opened": False,
}
(root / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"status": "PASS", "bytes": observed_bytes, "sha256": observed_sha, "holdout_opened": False}))
