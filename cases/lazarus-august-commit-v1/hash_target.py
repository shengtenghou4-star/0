from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

PAGE = "https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/download_db.comp"
ARCHIVE = (
    "https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?"
    "/scratch/NEUTRALIZATION/archive/CATNAP_2026_08_01.tar.gz"
)
EXPECTED_RELEASE_TOKEN = "CATNAP_2026_08_01.tar.gz"
UA = "Mozilla/5.0 LAZARUS-prospective-hash-only/2026-08-01"


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).parent / "artifacts"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    diagnostics = root / "transport-diagnostics"
    diagnostics.mkdir()
    cookie = diagnostics / "cookies.txt"
    page = diagnostics / "download-page.html"
    archive = diagnostics / EXPECTED_RELEASE_TOKEN
    meta = diagnostics / "curl-meta.txt"

    page_result = run([
        "curl", "-fsSL", "--max-redirs", "8", "--user-agent", UA,
        "--cookie-jar", str(cookie), PAGE, "-o", str(page),
    ])
    (diagnostics / "page.log").write_text(page_result.stdout, encoding="utf-8")
    if page_result.returncode:
        raise RuntimeError(f"download page failed with curl code {page_result.returncode}")

    archive_result = run([
        "curl", "-fsSL", "--max-redirs", "12", "--user-agent", UA,
        "--referer", PAGE, "--cookie", str(cookie), "--cookie-jar", str(cookie),
        "--write-out", "%{url_effective}\n%{http_code}\n%{size_download}\n",
        ARCHIVE, "-o", str(archive),
    ])
    meta.write_text(archive_result.stdout, encoding="utf-8")
    if archive_result.returncode:
        raise RuntimeError(f"archive transport failed with curl code {archive_result.returncode}")
    lines = archive_result.stdout.strip().splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"curl metadata incomplete: {archive_result.stdout!r}")
    final_url, status_text, downloaded_text = lines[-3:]
    if status_text != "200":
        raise RuntimeError(f"unexpected HTTP status: {status_text}")
    if EXPECTED_RELEASE_TOKEN not in ARCHIVE:
        raise AssertionError("source release token drifted")
    if not archive.is_file() or archive.stat().st_size <= 0:
        raise RuntimeError("empty target archive")
    downloaded = int(float(downloaded_text))
    if downloaded != archive.stat().st_size:
        raise RuntimeError((downloaded, archive.stat().st_size))

    receipt = {
        "schema_version": "1.0",
        "status": "TARGET_ARCHIVE_HASH_COMMITTED",
        "project": "LAZARUS Genesis",
        "release": "2026-08-01",
        "source": "LANL HIV CATNAP official monthly archive",
        "requested_url": ARCHIVE,
        "effective_url": final_url,
        "http_status": 200,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_stream(archive),
        "archive_members_listed": 0,
        "archive_members_extracted": 0,
        "target_rows_parsed": 0,
        "target_values_read": 0,
        "model_fits": 0,
        "private_bytes_read": 0,
        "confirmation_targets_read_by_scientific_evaluator": 0,
        "substitution_used": False,
    }
    (root / "TARGET_ARCHIVE_HASH_COMMITMENT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    archive.unlink()
    page.unlink(missing_ok=True)
    cookie.unlink(missing_ok=True)
    meta.unlink(missing_ok=True)
    for log in diagnostics.glob("*.log"):
        log.unlink()
    diagnostics.rmdir()
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
