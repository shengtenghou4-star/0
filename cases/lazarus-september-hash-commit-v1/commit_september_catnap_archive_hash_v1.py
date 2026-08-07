from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urljoin, urlparse

OFFICIAL_PAGE = "https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/download_db.comp"
TARGET_RELEASE = "2026-09-01"
TARGET_ARCHIVE_NAME = "CATNAP_2026_09_01.tar.gz"
USER_AGENT = "Mozilla/5.0 LAZARUS-september-hash-only/2026-09-01"
ALLOWED_HOST = "www.hiv.lanl.gov"


def canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_external(path: Path) -> str:
    result = subprocess.run(
        ["sha256sum", str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sha256sum failed: {result.stdout[-500:]}")
    token = result.stdout.strip().split()[0]
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise RuntimeError("sha256sum returned an invalid digest")
    return token


def exact_official_links(page_text: str) -> list[str]:
    hrefs = [
        html.unescape(match)
        for match in re.findall(r'href=["\']([^"\']+)["\']', page_text, flags=re.IGNORECASE)
        if TARGET_ARCHIVE_NAME in html.unescape(match)
    ]
    resolved: list[str] = []
    for href in hrefs:
        url = urljoin(OFFICIAL_PAGE, href)
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
            raise ValueError(f"matching archive link escaped official host: {url}")
        if TARGET_ARCHIVE_NAME not in url:
            raise ValueError("resolved official archive link lost exact release token")
        resolved.append(url)
    return sorted(set(resolved))


def base_receipt() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "project_id": "LAZ-001",
        "transaction": "september_catnap_archive_hash_only_commitment_v1",
        "release": TARGET_RELEASE,
        "archive_name": TARGET_ARCHIVE_NAME,
        "source_page": OFFICIAL_PAGE,
        "date_substitution_used": False,
        "archive_members_listed": 0,
        "archive_members_extracted": 0,
        "target_rows_parsed": 0,
        "target_values_read": 0,
        "scientific_supervised_fits": 0,
        "scientific_execution_authorized": False,
    }


def unavailable_receipt(page_text: str, reason: str) -> dict[str, object]:
    links = exact_official_links(page_text)
    receipt = base_receipt()
    receipt.update(
        {
            "status": "ABORT_TARGET_UNAVAILABLE",
            "reason": reason,
            "page_sha256": hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
            "page_mentions_exact_archive": TARGET_ARCHIVE_NAME in page_text,
            "matching_official_links": len(links),
            "archive_bytes": None,
            "archive_sha256": None,
            "independent_archive_sha256": None,
            "hash_implementations_agree": None,
            "target_archive_bytes_hashed": 0,
        }
    )
    return receipt


def committed_receipt(
    page_text: str,
    archive: Path,
    official_url: str,
    effective_url: str,
) -> dict[str, object]:
    links = exact_official_links(page_text)
    if links != [official_url]:
        raise ValueError("official link set changed before commitment")
    if archive.name != TARGET_ARCHIVE_NAME:
        raise ValueError("archive filename drifted")
    if not archive.is_file() or archive.stat().st_size <= 0:
        raise ValueError("archive object missing or empty")
    effective = urlparse(effective_url)
    if effective.scheme != "https" or effective.hostname != ALLOWED_HOST:
        raise ValueError("effective archive URL escaped official LANL host")
    first = sha256_stream(archive)
    second = sha256_external(archive)
    if first != second:
        raise RuntimeError("independent SHA-256 implementations disagree")
    receipt = base_receipt()
    receipt.update(
        {
            "status": "TARGET_ARCHIVE_HASH_COMMITTED",
            "page_sha256": hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
            "page_mentions_exact_archive": True,
            "matching_official_links": 1,
            "official_archive_url": official_url,
            "effective_url": effective_url,
            "http_status": 200,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": first,
            "independent_archive_sha256": second,
            "hash_implementations_agree": True,
            "target_archive_bytes_hashed": archive.stat().st_size,
        }
    )
    return receipt


def run_curl(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["curl", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def execute(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"duplicate commitment output: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(f"stale commitment staging path: {staging}")
    staging.mkdir(parents=True)

    cookie = staging / "cookies.txt"
    page_path = staging / "download-page.html"
    archive_path = staging / TARGET_ARCHIVE_NAME
    receipt_path = staging / "SEPTEMBER_TARGET_ARCHIVE_HASH_COMMITMENT.json"

    try:
        page_result = run_curl(
            [
                "-fsSL",
                "--max-redirs",
                "8",
                "--user-agent",
                USER_AGENT,
                "--cookie-jar",
                str(cookie),
                OFFICIAL_PAGE,
                "-o",
                str(page_path),
            ]
        )
        if page_result.returncode != 0:
            raise RuntimeError(f"official page transport failed: curl={page_result.returncode}")

        page_text = page_path.read_text(encoding="utf-8", errors="replace")
        links = exact_official_links(page_text)

        if TARGET_ARCHIVE_NAME not in page_text or len(links) == 0:
            receipt = unavailable_receipt(
                page_text,
                "official LANL download page does not publish the exact nominated September archive",
            )
        elif len(links) != 1:
            raise RuntimeError(f"expected exactly one official exact-release link, observed {len(links)}")
        else:
            official_url = links[0]
            archive_result = run_curl(
                [
                    "-sS",
                    "-L",
                    "--fail",
                    "--max-redirs",
                    "20",
                    "--user-agent",
                    USER_AGENT,
                    "--referer",
                    OFFICIAL_PAGE,
                    "--cookie",
                    str(cookie),
                    "--cookie-jar",
                    str(cookie),
                    "--write-out",
                    "\n%{url_effective}\n%{http_code}\n%{size_download}\n",
                    official_url,
                    "-o",
                    str(archive_path),
                ]
            )
            lines = archive_result.stdout.strip().splitlines()
            effective_url = lines[-3] if len(lines) >= 3 else ""
            http_status = lines[-2] if len(lines) >= 2 else ""
            downloaded = lines[-1] if len(lines) >= 1 else ""
            if (
                archive_result.returncode != 0
                or http_status != "200"
                or not archive_path.is_file()
                or archive_path.stat().st_size <= 0
            ):
                receipt = unavailable_receipt(
                    page_text,
                    "exact official link exists but transport did not yield one finite HTTP 200 archive; "
                    f"curl={archive_result.returncode}",
                )
                receipt.update(
                    {
                        "official_archive_url": official_url,
                        "effective_url": effective_url or None,
                        "http_status": int(http_status) if http_status.isdigit() else None,
                    }
                )
            else:
                try:
                    downloaded_bytes = int(float(downloaded))
                except ValueError as exc:
                    raise RuntimeError("curl size_download was not numeric") from exc
                if downloaded_bytes != archive_path.stat().st_size:
                    raise RuntimeError("curl download byte count disagrees with filesystem size")
                receipt = committed_receipt(page_text, archive_path, official_url, effective_url)

        archive_path.unlink(missing_ok=True)
        page_path.unlink(missing_ok=True)
        cookie.unlink(missing_ok=True)

        receipt_path.write_text(canonical_json(receipt), encoding="utf-8")
        sealed_files = [path.name for path in staging.iterdir() if path.is_file()]
        if sealed_files != [receipt_path.name]:
            raise RuntimeError(f"unexpected sealed files: {sealed_files}")
        staging.rename(output_dir)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    execute(Path("september-hash-commitment-v1"))
