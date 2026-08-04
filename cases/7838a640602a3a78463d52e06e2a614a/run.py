from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
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
ARCHIVE_URL = (
    "https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?"
    "%2Fscratch%2FNEUTRALIZATION%2Farchive%2FCATNAP_2026_07_01.tar.gz="
)
DIRECT_BASE = "https://www.hiv.lanl.gov/scratch/NEUTRALIZATION/latest"
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


def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={"User-Agent": f"LAZARUS-public-recovery/{CASE_ID}"},
    )


def download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(request(url), timeout=90) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


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
    public_dir.mkdir(parents=True, exist_ok=True)
    methods: dict[str, str] = {}
    errors: dict[str, str] = {}

    for name in REQUIRED:
        try:
            download(f"{DIRECT_BASE}/{name}", public_dir / name)
            methods[name] = "direct-latest-url"
        except Exception as exc:
            errors[name] = f"direct:{type(exc).__name__}:{exc}"

    missing = [name for name in REQUIRED if not (public_dir / name).is_file()]
    if missing:
        archive = artifact_dir / "CATNAP_2026_07_01.tar.gz"
        download(ARCHIVE_URL, archive)
        with tarfile.open(archive, "r:gz") as handle:
            members = {Path(member.name).name: member for member in handle.getmembers() if member.isfile()}
            for name in missing:
                member = members.get(name)
                if member is None:
                    continue
                source = handle.extractfile(member)
                if source is None:
                    continue
                with source, (public_dir / name).open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                methods[name] = "official-archive"
        archive.unlink(missing_ok=True)

    still_missing = [name for name in REQUIRED if not (public_dir / name).is_file()]
    if still_missing:
        raise RuntimeError(f"missing official CATNAP files: {still_missing}; errors={errors}")

    files = {
        name: {
            "bytes": (public_dir / name).stat().st_size,
            "sha256": sha256(public_dir / name),
            "method": methods.get(name, "unknown"),
        }
        for name in REQUIRED
    }
    mismatches = {
        name: {"expected": expected, "actual": files[name]["sha256"]}
        for name, expected in EXPECTED.items()
        if files[name]["sha256"] != expected
    }
    result = {
        "release": RELEASE,
        "source": "LANL HIV CATNAP official release",
        "files": files,
        "known_hashes_match": not mismatches,
        "mismatches": mismatches,
        "direct_download_errors": errors,
    }
    (public_dir / "PUBLIC_INPUT_RECEIPT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if mismatches:
        raise RuntimeError(f"official frozen hash mismatch: {mismatches}")
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
        "schema_version": "1.1",
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
