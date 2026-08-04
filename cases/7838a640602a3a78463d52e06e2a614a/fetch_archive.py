from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

PAGE = "https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/download_db.comp"
ARCHIVE = (
    "https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?"
    "/scratch/NEUTRALIZATION/archive/CATNAP_2026_07_01.tar.gz"
)
UA = "Mozilla/5.0 LAZARUS-public-recovery/7838a640602a3a78463d52e06e2a614a"
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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_curl(args: list[str], log: Path) -> None:
    result = subprocess.run(
        ["curl", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"curl {result.returncode}: {result.stdout[-1000:]}")


def main() -> None:
    root = Path(__file__).parent / "artifacts"
    shutil.rmtree(root, ignore_errors=True)
    data = root / "catnap-2026-07-01"
    diag = root / "transport-diagnostics"
    data.mkdir(parents=True)
    diag.mkdir(parents=True)
    cookie = diag / "cookies.txt"
    page = diag / "download-page.html"
    archive = root / "CATNAP_2026_07_01.tar.gz"

    try:
        run_curl(
            ["-fsSL", "--max-redirs", "8", "--user-agent", UA,
             "--cookie-jar", str(cookie), PAGE, "-o", str(page)],
            diag / "page.log",
        )
        run_curl(
            ["-fsSL", "--max-redirs", "12", "--user-agent", UA,
             "--referer", PAGE, "--cookie", str(cookie),
             "--cookie-jar", str(cookie), ARCHIVE, "-o", str(archive)],
            diag / "archive.log",
        )
        with tarfile.open(archive, "r:gz") as handle:
            members = {
                Path(member.name).name: member
                for member in handle.getmembers()
                if member.isfile()
            }
            (diag / "archive-members.txt").write_text(
                "\n".join(sorted(members)) + "\n", encoding="utf-8"
            )
            for name in REQUIRED:
                member = members.get(name)
                if member is None:
                    raise RuntimeError(f"archive member missing: {name}")
                source = handle.extractfile(member)
                if source is None:
                    raise RuntimeError(f"archive member unreadable: {name}")
                with source, (data / name).open("wb") as destination:
                    shutil.copyfileobj(source, destination)

        files = {
            name: {"bytes": (data / name).stat().st_size, "sha256": digest(data / name)}
            for name in REQUIRED
        }
        mismatches = {
            name: {"expected": expected, "actual": files[name]["sha256"]}
            for name, expected in EXPECTED.items()
            if files[name]["sha256"] != expected
        }
        receipt = {
            "schema_version": "1.0",
            "release": "2026-07-01",
            "source": "LANL HIV CATNAP official archive",
            "archive_transport": ARCHIVE,
            "files": files,
            "known_hashes_match": not mismatches,
            "mismatches": mismatches,
            "private_bytes_read": 0,
            "model_fits": 0,
            "confirmation_targets_read": 0,
        }
        (data / "PUBLIC_INPUT_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if mismatches:
            raise RuntimeError(f"frozen public hash mismatch: {mismatches}")
        print(json.dumps(receipt, sort_keys=True))
    except Exception as exc:
        (root / "FAILURE.json").write_text(
            json.dumps({"error": type(exc).__name__, "detail": str(exc)}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        archive.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
