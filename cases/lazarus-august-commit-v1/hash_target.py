from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urljoin

PAGE = "https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/download_db.comp"
NOMINATED_ARCHIVE = (
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


def write_receipt(root: Path, receipt: dict[str, object]) -> None:
    (root / "TARGET_ARCHIVE_HASH_COMMITMENT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


def base_receipt() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "project": "LAZARUS Genesis",
        "release": "2026-08-01",
        "source": "LANL HIV CATNAP official monthly archive",
        "nominated_url": NOMINATED_ARCHIVE,
        "archive_members_listed": 0,
        "archive_members_extracted": 0,
        "target_rows_parsed": 0,
        "target_values_read": 0,
        "model_fits": 0,
        "private_bytes_read": 0,
        "confirmation_targets_read_by_scientific_evaluator": 0,
        "substitution_used": False,
    }


def main() -> None:
    root = Path(__file__).parent / "artifacts"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    diagnostics = root / "transport-diagnostics"
    diagnostics.mkdir()
    cookie = diagnostics / "cookies.txt"
    page = diagnostics / "download-page.html"
    archive = diagnostics / EXPECTED_RELEASE_TOKEN

    page_result = run([
        "curl", "-fsSL", "--max-redirs", "8", "--user-agent", UA,
        "--cookie-jar", str(cookie), PAGE, "-o", str(page),
    ])
    if page_result.returncode:
        raise RuntimeError(f"download page failed with curl code {page_result.returncode}")

    page_text = page.read_text(encoding="utf-8", errors="replace")
    page_mentions_release = EXPECTED_RELEASE_TOKEN in page_text
    hrefs = [
        html.unescape(match)
        for match in re.findall(r'href=["\']([^"\']+)["\']', page_text, flags=re.IGNORECASE)
        if EXPECTED_RELEASE_TOKEN in html.unescape(match)
    ]
    receipt = base_receipt()
    receipt["page_mentions_release"] = page_mentions_release
    receipt["matching_official_links"] = len(hrefs)

    if not page_mentions_release or not hrefs:
        receipt.update({
            "status": "ABORT_TARGET_UNAVAILABLE",
            "reason": "official LANL download page does not publish the nominated 2026-08-01 archive",
            "archive_bytes": None,
            "archive_sha256": None,
            "effective_url": None,
            "http_status": None,
        })
        write_receipt(root, receipt)
    else:
        official_url = urljoin(PAGE, hrefs[0])
        if EXPECTED_RELEASE_TOKEN not in official_url:
            raise AssertionError("official page link release token drifted")
        archive_result = run([
            "curl", "-sS", "-L", "--fail", "--max-redirs", "20",
            "--user-agent", UA, "--referer", PAGE,
            "--cookie", str(cookie), "--cookie-jar", str(cookie),
            "--write-out", "\n%{url_effective}\n%{http_code}\n%{size_download}\n",
            official_url, "-o", str(archive),
        ])
        lines = archive_result.stdout.strip().splitlines()
        final_url = lines[-3] if len(lines) >= 3 else None
        status_text = lines[-2] if len(lines) >= 2 else None
        downloaded_text = lines[-1] if lines else None
        if archive_result.returncode or status_text != "200" or not archive.is_file() or archive.stat().st_size <= 0:
            receipt.update({
                "status": "ABORT_TARGET_UNAVAILABLE",
                "reason": f"official page lists the archive but transport did not yield one finite HTTP 200 object; curl={archive_result.returncode}",
                "official_page_url": official_url,
                "effective_url": final_url,
                "http_status": int(status_text) if status_text and status_text.isdigit() else None,
                "archive_bytes": archive.stat().st_size if archive.exists() else None,
                "archive_sha256": None,
            })
            write_receipt(root, receipt)
        else:
            downloaded = int(float(downloaded_text))
            if downloaded != archive.stat().st_size:
                raise RuntimeError((downloaded, archive.stat().st_size))
            receipt.update({
                "status": "TARGET_ARCHIVE_HASH_COMMITTED",
                "official_page_url": official_url,
                "effective_url": final_url,
                "http_status": 200,
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": sha256_stream(archive),
            })
            write_receipt(root, receipt)

    archive.unlink(missing_ok=True)
    page.unlink(missing_ok=True)
    cookie.unlink(missing_ok=True)
    diagnostics.rmdir()


if __name__ == "__main__":
    main()
