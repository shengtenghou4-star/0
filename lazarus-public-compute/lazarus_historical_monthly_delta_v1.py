#!/usr/bin/env python3
"""Three-window historical CATNAP release-delta audit for the frozen LAZARUS family."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

WINDOWS = (
    ("2026-04-01", "2026-05-01"),
    ("2026-05-01", "2026-06-01"),
    ("2026-06-01", "2026-07-01"),
)
MINIMUMS = {
    "entries": 50,
    "potent_entries": 10,
    "resistant_entries": 10,
    "unique_antibodies": 5,
    "unique_viruses": 5,
    "antibody_families": 2,
}
EXPECTED_RUNNER_SHA256 = "3e27e43dc33ba8deac14d4b981cd8fe9d7d220cd01e908f31fee51fb4e9a200c"
EXPECTED_LEGACY_SHA256 = "7429d615a4fcbcff529270ed25dbfb6e1395b07883dda59d4f9317724bfd6a2c"
EXPECTED_ORTH_SHA256 = "42c284c9f06f713a59443fae04987292b6fc39ff1de1f85e903207c6857a8c33"
EXPECTED_WRAPPER_SHA256 = "5d354d5482c59be02fe7775849e6841b333d35617262e91c46fae633efd8cdf1"
PAGE = "https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/download_db.comp"
UA = "Mozilla/5.0 LAZARUS-historical-monthly-delta-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_curl(args: list[str]) -> None:
    result = subprocess.run(
        ["curl", *args], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"curl {result.returncode}: {result.stdout[-1200:]}")


def archive_url(release: str) -> str:
    stamp = release.replace("-", "_")
    return (
        "https://www.hiv.lanl.gov/cgi-bin/common_code/download.cgi?"
        f"/scratch/NEUTRALIZATION/archive/CATNAP_{stamp}.tar.gz"
    )


def names_for(release: str) -> tuple[str, ...]:
    return (
        f"assay_{release}.txt",
        f"abs_{release}.txt",
        f"heavy_seqs_aa_{release}.fasta",
        f"light_seqs_aa_{release}.fasta",
        f"virseqs_aa_O_{release}.fasta",
    )


def fetch_release(release: str, root: Path) -> dict[str, Any]:
    destination = root / release
    destination.mkdir(parents=True)
    archive = root / f"CATNAP_{release.replace('-', '_')}.tar.gz"
    cookie = root / f"cookies-{release}.txt"
    page = root / f"page-{release}.html"
    run_curl([
        "-fsSL", "--max-redirs", "8", "--user-agent", UA,
        "--cookie-jar", str(cookie), PAGE, "-o", str(page),
    ])
    run_curl([
        "-fsSL", "--max-redirs", "12", "--user-agent", UA,
        "--referer", PAGE, "--cookie", str(cookie), "--cookie-jar", str(cookie),
        archive_url(release), "-o", str(archive),
    ])
    if not archive.is_file() or archive.stat().st_size <= 0:
        raise RuntimeError(f"empty archive for {release}")
    required = names_for(release)
    with tarfile.open(archive, "r:gz") as handle:
        members = {
            Path(member.name).name: member
            for member in handle.getmembers() if member.isfile()
        }
        for name in required:
            member = members.get(name)
            if member is None:
                raise RuntimeError(f"{release} missing {name}")
            source = handle.extractfile(member)
            if source is None:
                raise RuntimeError(f"{release} unreadable {name}")
            with source, (destination / name).open("wb") as output:
                shutil.copyfileobj(source, output)
    receipt = {
        "release": release,
        "source": "LANL HIV CATNAP official archive",
        "archive_url": archive_url(release),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "files": {
            name: {"bytes": (destination / name).stat().st_size, "sha256": sha256(destination / name)}
            for name in required
        },
        "real_august_rows_read": 0,
        "private_bytes_read": 0,
    }
    (destination / "PUBLIC_INPUT_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    archive.unlink()
    cookie.unlink(missing_ok=True)
    page.unlink(missing_ok=True)
    return receipt


def canonical_training(source: Path, release: str, destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True)
    mapping = {
        "assay_2026-07-01.txt": f"assay_{release}.txt",
        "abs_2026-07-01.txt": f"abs_{release}.txt",
        "heavy_seqs_aa_2026-07-01.fasta": f"heavy_seqs_aa_{release}.fasta",
        "light_seqs_aa_2026-07-01.fasta": f"light_seqs_aa_{release}.fasta",
        "virseqs_aa_O_2026-07-01.fasta": f"virseqs_aa_O_{release}.fasta",
    }
    for canonical, original in mapping.items():
        shutil.copyfile(source / original, destination / canonical)
    return {name: sha256(destination / name) for name in mapping}


def canonical_target(source: Path, release: str, destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True)
    mapping = {
        "assay_2026-08-01.txt": f"assay_{release}.txt",
        "virseqs_aa_O_2026-08-01.fasta": f"virseqs_aa_O_{release}.fasta",
    }
    for canonical, original in mapping.items():
        shutil.copyfile(source / original, destination / canonical)
    return {name: sha256(destination / name) for name in mapping}


def hash_files(directory: Path) -> dict[str, dict[str, int | str]]:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(directory.iterdir()) if path.is_file()
    }


def read_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {"count", "rmse", "bias", "pearson", "spearman", "prediction_truth_std_ratio"}
    for row in rows:
        for key in numeric:
            if key in row:
                row[key] = float(row[key])
    return rows


def seal_insufficient(
    staging: Path,
    training_release: str,
    target_release: str,
    counts: dict[str, int],
    failures: dict[str, dict[str, int]],
    training_hashes: dict[str, str],
    target_hashes: dict[str, str],
) -> None:
    staging.mkdir(parents=True)
    manifest = {
        "training_release": training_release,
        "target_release": target_release,
        "counts": counts,
        "minimum_failures": failures,
        "training_overlap_pairs": 0,
    }
    (staging / "historical_target_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    adjudication = {
        "status": "TEMPORAL_INSUFFICIENT",
        "supervised_fits": 0,
        "minimum_failures": failures,
    }
    (staging / "historical_adjudication.json").write_text(
        json.dumps(adjudication, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema_version": "1.0",
        "status": "TEMPORAL_INSUFFICIENT",
        "training_release": training_release,
        "target_release": target_release,
        "supervised_fits": 0,
        "candidate_count": 1,
        "training_hashes": training_hashes,
        "target_hashes": target_hashes,
        "real_august_rows_read": 0,
        "real_august_values_read": 0,
        "private_bytes_read": 0,
    }
    receipt["files"] = hash_files(staging)
    (staging / "HISTORICAL_WINDOW_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def execute_window(
    runner: Any,
    legacy: Any,
    orth: Any,
    wrapper: Any,
    families: Path,
    releases_root: Path,
    training_release: str,
    target_release: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    staging = output.parent / f".{output.name}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        training_data = temp / "training"
        target_data = temp / "target"
        training_hashes = canonical_training(
            releases_root / training_release, training_release, training_data
        )
        target_hashes = canonical_target(
            releases_root / target_release, target_release, target_data
        )
        runner.EXPECTED_TRAINING_HASHES = dict(training_hashes)
        runner.MINIMUMS = dict(MINIMUMS)
        runner.generic_matrix = lambda legacy_arg, assay, fasta: wrapper.prospective_generic_matrix(
            runner, legacy_arg, assay, fasta
        )
        family_by, _ = legacy.load_family_map(families)
        train, antibody_sequences, training_viruses, training_keys = runner.training_pairs(
            legacy, training_data, family_by
        )
        target, _, _, counts = runner.target_pairs(
            legacy, target_data, family_by, training_keys,
            antibody_sequences, training_viruses,
        )
        overlap = sum((row.antibody, row.virus) in training_keys for row in target)
        if overlap:
            raise AssertionError(f"{training_release}->{target_release} overlap {overlap}")
        failures = {
            name: {"required": required, "observed": int(counts[name])}
            for name, required in MINIMUMS.items() if int(counts[name]) < required
        }
        if failures:
            seal_insufficient(
                staging, training_release, target_release,
                {key: int(value) for key, value in counts.items()}, failures,
                training_hashes, target_hashes,
            )
            staging.rename(output)
            return {
                "training_release": training_release,
                "target_release": target_release,
                "status": "TEMPORAL_INSUFFICIENT",
                "supervised_fits": 0,
                "counts": counts,
            }

        commitment = temp / "commitment.json"
        commitment.write_text(json.dumps({
            "status": "TARGET_ARCHIVE_HASH_COMMITTED",
            "release": "2026-08-01",
            "archive_sha256": hashlib.sha256(
                (training_release + "->" + target_release).encode()
            ).hexdigest(),
            "historical_window": True,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        fit_count = 0
        original_fit = runner.Ridge.fit
        def counted_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal fit_count
            fit_count += 1
            return original_fit(self, *args, **kwargs)
        runner.Ridge.fit = counted_fit
        try:
            with tempfile.TemporaryFile(mode="w+") as capture:
                original_stdout = sys.stdout
                sys.stdout = capture
                try:
                    runner.execute(
                        legacy, orth, training_data, target_data,
                        families, commitment, staging,
                    )
                finally:
                    sys.stdout = original_stdout
        finally:
            runner.Ridge.fit = original_fit
        if fit_count != 2:
            raise AssertionError(f"fit budget drift {training_release}->{target_release}: {fit_count}")

        target_manifest_path = staging / "prospective_target_manifest.json"
        target_manifest = json.loads(target_manifest_path.read_text())
        target_manifest.update({
            "historical_training_release": training_release,
            "historical_target_release": target_release,
            "canonical_runtime_release_alias": "2026-08-01",
            "training_hashes": training_hashes,
            "target_hashes": target_hashes,
        })
        target_manifest_path.write_text(
            json.dumps(target_manifest, indent=2, sort_keys=True) + "\n"
        )
        adjudication_path = staging / "prospective_adjudication.json"
        adjudication = json.loads(adjudication_path.read_text())
        prospective_status = adjudication["status"]
        adjudication.update({
            "status": "TEMPORAL_WINDOW_COMPLETE",
            "historical_training_release": training_release,
            "historical_target_release": target_release,
            "prospective_gate_projection": prospective_status,
            "claim_class": "post-selection retrospective temporal stress test",
        })
        adjudication_path.write_text(
            json.dumps(adjudication, indent=2, sort_keys=True) + "\n"
        )
        original_receipt = staging / "PROSPECTIVE_RELAY_RECEIPT.json"
        original_receipt.unlink()
        receipt = {
            "schema_version": "1.0",
            "status": "TEMPORAL_WINDOW_COMPLETE",
            "training_release": training_release,
            "target_release": target_release,
            "supervised_fits": fit_count,
            "candidate_count": 1,
            "training_hashes": training_hashes,
            "target_hashes": target_hashes,
            "prospective_gate_projection": prospective_status,
            "real_august_rows_read": 0,
            "real_august_values_read": 0,
            "private_bytes_read": 0,
        }
        receipt["files"] = hash_files(staging)
        (staging / "HISTORICAL_WINDOW_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(output)
        metrics = read_metrics(output / "prospective_metrics.csv")
        indexed = {(row["model"], row["regime"]): row for row in metrics}
        return {
            "training_release": training_release,
            "target_release": target_release,
            "status": "TEMPORAL_WINDOW_COMPLETE",
            "supervised_fits": fit_count,
            "counts": counts,
            "overall_raw_rmse": indexed[(runner.REFERENCE, "overall")]["rmse"],
            "overall_candidate_rmse": indexed[(runner.CANDIDATE, "overall")]["rmse"],
            "potent_raw_rmse": indexed[(runner.REFERENCE, "potent")]["rmse"],
            "potent_candidate_rmse": indexed[(runner.CANDIDATE, "potent")]["rmse"],
            "prospective_gate_projection": prospective_status,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--orth", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {
        args.runner: EXPECTED_RUNNER_SHA256,
        args.legacy: EXPECTED_LEGACY_SHA256,
        args.orth: EXPECTED_ORTH_SHA256,
        args.wrapper: EXPECTED_WRAPPER_SHA256,
    }
    for path, digest in expected.items():
        actual = sha256(path)
        if actual != digest:
            raise ValueError(f"payload hash drift {path}: {actual}")
    if args.output.exists():
        raise FileExistsError(args.output)
    staging = args.output.parent / f".{args.output.name}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    releases_root = staging / "inputs"
    windows_root = staging / "windows"
    releases_root.mkdir()
    windows_root.mkdir()

    input_receipts = {
        release: fetch_release(release, releases_root)
        for release in ("2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01")
    }
    runner = load_module(args.runner, "lazarus_historical_runner")
    legacy = load_module(args.legacy, "lazarus_historical_legacy")
    orth = load_module(args.orth, "lazarus_historical_orth")
    wrapper = load_module(args.wrapper, "lazarus_historical_wrapper")

    summaries = []
    total_fits = 0
    for training_release, target_release in WINDOWS:
        result = execute_window(
            runner, legacy, orth, wrapper, args.families, releases_root,
            training_release, target_release,
            windows_root / f"{training_release}_to_{target_release}",
        )
        total_fits += int(result["supervised_fits"])
        summaries.append(result)
    if total_fits > 6:
        raise AssertionError(f"total fit budget drifted: {total_fits}")

    fitted = [row for row in summaries if row["status"] == "TEMPORAL_WINDOW_COMPLETE"]
    overall_wins = sum(row["overall_candidate_rmse"] < row["overall_raw_rmse"] for row in fitted)
    potent_wins = sum(row["potent_candidate_rmse"] < row["potent_raw_rmse"] for row in fitted)
    joint_wins = sum(
        row["overall_candidate_rmse"] < row["overall_raw_rmse"]
        and row["potent_candidate_rmse"] < row["potent_raw_rmse"]
        for row in fitted
    )
    if not fitted:
        label = "TEMPORAL_INSUFFICIENT"
    elif joint_wins == len(fitted):
        label = "TEMPORAL_JOINT_REPLICATION"
    elif potent_wins == len(fitted):
        label = "TEMPORAL_POTENT_ONLY_REPLICATION"
    elif overall_wins or potent_wins:
        label = "TEMPORAL_MIXED"
    else:
        label = "TEMPORAL_FAILURE"
    summary = {
        "schema_version": "1.0",
        "status": label,
        "claim_class": "post-selection retrospective temporal stress test; not independent confirmation",
        "windows_declared": len(WINDOWS),
        "windows_fitted": len(fitted),
        "overall_rmse_win_windows": overall_wins,
        "potent_rmse_win_windows": potent_wins,
        "joint_win_windows": joint_wins,
        "historical_supervised_fits": total_fits,
        "maximum_historical_supervised_fits": 6,
        "real_august_rows_read": 0,
        "real_august_values_read": 0,
        "prospective_scientific_fits": 0,
        "private_bytes_read": 0,
        "windows": summaries,
        "input_receipts": input_receipts,
    }
    (staging / "HISTORICAL_TEMPORAL_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(releases_root)
    receipt = {
        "schema_version": "1.0",
        "status": label,
        "historical_supervised_fits": total_fits,
        "real_august_rows_read": 0,
        "real_august_values_read": 0,
        "prospective_scientific_fits": 0,
        "private_bytes_read": 0,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "legacy_sha256": EXPECTED_LEGACY_SHA256,
        "orth_sha256": EXPECTED_ORTH_SHA256,
        "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        "files": hash_files(staging),
    }
    (staging / "HISTORICAL_TEMPORAL_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.rename(args.output)
    print(json.dumps({
        "status": label,
        "windows_fitted": len(fitted),
        "historical_supervised_fits": total_fits,
        "real_august_rows_read": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
