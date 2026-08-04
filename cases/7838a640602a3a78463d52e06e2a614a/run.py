from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

CASE_ID = "7838a640602a3a78463d52e06e2a614a"
TOKEN_NAMES = (
    "LAZARUS_REPO_TOKEN",
    "PRIVATE_REPO_TOKEN",
    "GH_PAT",
    "REPO_TOKEN",
)
RELEASE = "2026-07-01"
PAGE_URL = "https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/download_db.comp"
ARCHIVE_URL = (
    "https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?"
    "%2Fscratch%2FNEUTRALIZATION%2Farchive%2FCATNAP_2026_07_01.tar.gz="
)
DIRECT_BASE = "https://www.hiv.lanl.gov/scratch/NEUTRALIZATION/latest"
USER_AGENT = f"Mozilla/5.0 LAZARUS-public-recovery/{CASE_ID}"
REQUIRED = (
    "assay_2026-07-01.txt",
    "abs_2026-07-01.txt",
    "heavy_seqs_aa_2026-07-01.fasta",
    "light_seqs_aa_2026-07-01.fasta",
    "virseqs_aa_O_2026-07-01.fasta",
)
EXPECTED = {
    "assay_2026-07-01.txt": "e6b575e0785b43c5dbd84b7ef3eb08e2d072fa32117d1ad0bdd31eb5e243bdd2",
    "virseqs_aa_O_2026-07-01.fasta": "9c83b868d7159ad6148f628878027ccbaf2e07131baf4efd3221f1aa8ec568cf",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curl(args: list[str], log: Path) -> None:
    completed = subprocess.run(
        ["curl", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"curl exit {completed.returncode}: {completed.stdout[-1000:]}")


def probe_private_repo(token: str) -> tuple[bool, int | None, str]:
    req = urllib.request.Request(
        "https://api.github.com/repos/shengtenghou4-star/19",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": f"fixed-case-{CASE_ID}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return (
                response.status == 200 and payload.get("full_name") == "shengtenghou4-star/19",
                response.status,
                "repository-visible" if payload.get("private") is True else "unexpected-payload",
            )
    except urllib.error.HTTPError as exc:
        return False, exc.code, "http-error"
    except Exception as exc:
        return False, None, type(exc).__name__


def recover_public_inputs(artifact_dir: Path) -> dict[str, object]:
    public_dir = artifact_dir / "catnap-2026-07-01"
    diagnostics = artifact_dir / "transport-diagnostics"
    public_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    cookie = diagnostics / "cookies.txt"
    page = diagnostics / "download-page.html"
    methods: dict[str, str] = {}
    errors: dict[str, str] = {}

    curl(
        [
            "-fsSL",
            "--max-redirs", "8",
            "--user-agent", USER_AGENT,
            "--cookie-jar", str(cookie),
            PAGE_URL,
            "-o", str(page),
        ],
        diagnostics / "page.log",
    )

    for name in REQUIRED:
        try:
            curl(
                [
                    "-fsSL",
                    "--max-redirs", "12",
                    "--user-agent", USER_AGENT,
                    "--referer", PAGE_URL,
                    "--cookie", str(cookie),
                    "--cookie-jar", str(cookie),
                    f"{DIRECT_BASE}/{name}",
                    "-o", str(public_dir / name),
                ],
                diagnostics / f"{name}.log",
            )
            methods[name] = "cookie-aware-direct-latest-url"
        except Exception as exc:
            errors[name] = f"direct:{type(exc).__name__}:{exc}"
            (public_dir / name).unlink(missing_ok=True)

    missing = [name for name in REQUIRED if not (public_dir / name).is_file()]
    if missing:
        archive = artifact_dir / "CATNAP_2026_07_01.tar.gz"
        try:
            curl(
                [
                    "-fsSL",
                    "--max-redirs", "12",
                    "--user-agent", USER_AGENT,
                    "--referer", PAGE_URL,
                    "--cookie", str(cookie),
                    "--cookie-jar", str(cookie),
                    ARCHIVE_URL,
                    "-o", str(archive),
                ],
                diagnostics / "archive.log",
            )
            with tarfile.open(archive, "r:gz") as handle:
                members = {
                    Path(member.name).name: member
                    for member in handle.getmembers()
                    if member.isfile()
                }
                for name in missing:
                    member = members.get(name)
                    if member is None:
                        continue
                    source = handle.extractfile(member)
                    if source is None:
                        continue
                    with source, (public_dir / name).open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                    methods[name] = "cookie-aware-official-archive"
        except Exception as exc:
            errors["archive"] = f"{type(exc).__name__}:{exc}"
        finally:
            archive.unlink(missing_ok=True)

    still_missing = [name for name in REQUIRED if not (public_dir / name).is_file()]
    files = {
        name: {
            "bytes": (public_dir / name).stat().st_size,
            "sha256": sha256(public_dir / name),
            "method": methods.get(name, "unknown"),
        }
        for name in REQUIRED
        if (public_dir / name).is_file()
    }
    mismatches = {
        name: {"expected": expected, "actual": files.get(name, {}).get("sha256")}
        for name, expected in EXPECTED.items()
        if files.get(name, {}).get("sha256") != expected
    }
    result = {
        "release": RELEASE,
        "source": "LANL HIV CATNAP official release",
        "files": files,
        "missing": still_missing,
        "known_hashes_match": not mismatches and not still_missing,
        "mismatches": mismatches,
        "transport_errors": errors,
    }
    (public_dir / "PUBLIC_INPUT_RECEIPT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if still_missing or mismatches:
        raise RuntimeError(
            f"official frozen input closure failed: missing={still_missing}, mismatches={mismatches}"
        )
    return result


def main() -> None:
    artifact_dir = Path(__file__).parent / "artifacts"
    shutil.rmtree(artifact_dir, ignore_errors=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    token_presence = {name: bool(os.environ.get(name, "")) for name in TOKEN_NAMES}
    private_access = False
    selected_token_name: str | None = None
    http_status: int | None = None
    probe_note = "no-token-present"
    for name in TOKEN_NAMES:
        token = os.environ.get(name, "")
        if not token:
            continue
        ok, status, note = probe_private_repo(token)
        if ok:
            private_access = True
            selected_token_name = name
            http_status = status
            probe_note = note
            break
        if http_status is None:
            http_status = status
            probe_note = note

    public_result = recover_public_inputs(artifact_dir)
    result = {
        "schema_version": "1.2",
        "case_id": CASE_ID,
        "runner_started": True,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "token_presence": token_presence,
        "private_repository_access": private_access,
        "selected_token_name": selected_token_name,
        "private_probe_http_status": http_status,
        "private_probe_note": probe_note,
        "private_bytes_read": 0,
        "public_inputs_recovered": True,
        "public_known_hashes_match": public_result["known_hashes_match"],
        "model_fits": 0,
        "confirmation_targets_read": 0,
    }
    (artifact_dir / "probe.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
