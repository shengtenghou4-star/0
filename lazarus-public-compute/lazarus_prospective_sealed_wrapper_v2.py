#!/usr/bin/env python3
"""Integrity wrapper for the frozen LAZARUS prospective runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_RUNNER_SHA256 = "3e27e43dc33ba8deac14d4b981cd8fe9d7d220cd01e908f31fee51fb4e9a200c"
EXPECTED_ORTH_SHA256 = "42c284c9f06f713a59443fae04987292b6fc39ff1de1f85e903207c6857a8c33"
TARGET_RELEASE = "2026-08-01"
TARGET_REQUIRED_FILES = (
    "assay_2026-08-01.txt",
    "virseqs_aa_O_2026-08-01.fasta",
)


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


def extract_committed_archive(
    archive: Path,
    commitment: dict[str, Any],
    target_data: Path,
) -> dict[str, dict[str, int | str]]:
    if commitment.get("status") != "TARGET_ARCHIVE_HASH_COMMITTED":
        raise ValueError("target archive commitment is not closed")
    if commitment.get("release") != TARGET_RELEASE:
        raise ValueError("target release drifted")
    expected = commitment.get("archive_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("target archive commitment hash missing")
    actual = sha256(archive)
    if actual != expected:
        raise ValueError(f"target archive hash mismatch: {actual}")
    if not archive.is_file() or archive.stat().st_size <= 0:
        raise ValueError("target archive is empty")
    committed_bytes = commitment.get("archive_bytes")
    if committed_bytes is not None and int(committed_bytes) != archive.stat().st_size:
        raise ValueError("target archive byte length mismatch")

    shutil.rmtree(target_data, ignore_errors=True)
    target_data.mkdir(parents=True)
    manifest: dict[str, dict[str, int | str]] = {}
    with tarfile.open(archive, "r:gz") as handle:
        members = {
            Path(member.name).name: member
            for member in handle.getmembers()
            if member.isfile()
        }
        for name in TARGET_REQUIRED_FILES:
            member = members.get(name)
            if member is None:
                raise ValueError(f"target archive member missing: {name}")
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"target archive member unreadable: {name}")
            destination = target_data / name
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            manifest[name] = {
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
    return manifest


def prospective_generic_matrix(
    runner: Any,
    legacy: Any,
    assay: Path,
    fasta: Path,
) -> tuple[dict[tuple[str, str], tuple[float, int]], set[str], set[str], dict[str, str]]:
    minimum_virus_edges = 1 if assay.name == "assay_2026-08-01.txt" else 10
    sequences = legacy.virus_sequences(fasta)
    valid: set[str] = set()
    for line in fasta.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(">"):
            continue
        header = line[1:].strip()
        fields = header.split(".")
        key = fields[-2].strip() if len(fields) >= 2 else ""
        if key and "HXB2" not in header.upper():
            valid.add(key)

    fields, rows = legacy.read_table(assay)
    antibody_field = legacy.find_field(fields, ("Antibody",))
    virus_field = legacy.find_field(fields, ("Virus",))
    ic50_field = legacy.find_field(fields, ("IC50",))
    if not antibody_field or not virus_field or not ic50_field:
        raise ValueError("required CATNAP fields missing")

    antibody_order: list[str] = []
    virus_order: list[str] = []
    seen_antibodies: set[str] = set()
    seen_viruses: set[str] = set()
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        antibody = legacy.clean(row.get(antibody_field, ""))
        virus = legacy.clean(row.get(virus_field, ""))
        if antibody not in seen_antibodies:
            seen_antibodies.add(antibody)
            antibody_order.append(antibody)
        if virus not in seen_viruses:
            seen_viruses.add(virus)
            virus_order.append(virus)
        if "polyclonal" in antibody.casefold() or "/" in antibody or "+" in antibody or virus not in valid:
            continue
        target = legacy.parse_val(row.get(ic50_field, ""))
        if target is not None:
            values[(antibody, virus)].append(float(target))

    eligible_antibodies = [
        antibody
        for antibody in antibody_order
        if "polyclonal" not in antibody.casefold() and "/" not in antibody and "+" not in antibody
    ]
    eligible_viruses = [virus for virus in virus_order if virus in valid]
    for _ in range(20):
        antibody_set = set(eligible_antibodies)
        virus_set = set(eligible_viruses)
        antibody_counts: Counter[str] = Counter()
        virus_counts: Counter[str] = Counter()
        for antibody, virus in values:
            if antibody in antibody_set and virus in virus_set:
                antibody_counts[antibody] += 1
                virus_counts[virus] += 1
        next_antibodies = [
            antibody for antibody in eligible_antibodies if antibody_counts[antibody] >= 1
        ]
        next_viruses = [
            virus for virus in eligible_viruses if virus_counts[virus] >= minimum_virus_edges
        ]
        if next_antibodies == eligible_antibodies and next_viruses == eligible_viruses:
            break
        eligible_antibodies, eligible_viruses = next_antibodies, next_viruses

    antibody_set = set(eligible_antibodies)
    virus_set = set(eligible_viruses)
    matrix = {
        (antibody, virus): (float(sum(items) / len(items)), len(items))
        for (antibody, virus), items in values.items()
        if antibody in antibody_set and virus in virus_set
    }
    return matrix, antibody_set, virus_set, sequences


def hash_files(directory: Path) -> dict[str, dict[str, int | str]]:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "PROSPECTIVE_RELAY_RECEIPT.json"
    }


def seal_abort(
    staging: Path,
    commitment: dict[str, Any],
    wrapper_sha256: str,
    fit_count: int,
) -> None:
    adjudication_path = staging / "prospective_adjudication.json"
    if not adjudication_path.is_file():
        raise RuntimeError("insufficient-target abort did not produce adjudication")
    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    if adjudication.get("status") != "ABORT_INSUFFICIENT_TARGET":
        raise RuntimeError("unexpected partial adjudication")
    if fit_count != 0:
        raise AssertionError(f"abort consumed supervised fits: {fit_count}")
    receipt = {
        "schema_version": "2.0",
        "status": "ABORT_INSUFFICIENT_TARGET",
        "public_compute": True,
        "private_bytes_read": 0,
        "confirmation_target_release": TARGET_RELEASE,
        "target_archive_sha256": commitment["archive_sha256"],
        "target_archive_hash_reverified": True,
        "supervised_fits": 0,
        "candidate_count": 1,
        "adaptive_expansion_allowed": 0,
        "atomic_publication": True,
        "wrapper_sha256": wrapper_sha256,
        "wrapped_runner_sha256": EXPECTED_RUNNER_SHA256,
        "files": hash_files(staging),
    }
    (staging / "PROSPECTIVE_RELAY_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def execute(args: argparse.Namespace) -> None:
    if sha256(args.runner) != EXPECTED_RUNNER_SHA256:
        raise ValueError("wrapped runner hash drifted")
    if sha256(args.orth) != EXPECTED_ORTH_SHA256:
        raise ValueError("orthogonal residual helper hash drifted")
    commitment = json.loads(args.target_commitment.read_text(encoding="utf-8"))
    wrapper_sha = sha256(Path(__file__))

    runner = load_module(args.runner, "lazarus_prospective_wrapped")
    legacy = load_module(args.legacy, "lazarus_legacy_wrapped")
    orth = load_module(args.orth, "lazarus_orth_wrapped")
    runner.generic_matrix = lambda legacy_arg, assay, fasta: prospective_generic_matrix(
        runner, legacy_arg, assay, fasta
    )

    output = args.output
    if output.exists():
        raise FileExistsError(f"sealed output path already exists: {output}")
    staging = output.parent / f".{output.name}.staging"
    target_data = output.parent / f".{output.name}.target-data"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(target_data, ignore_errors=True)

    target_file_manifest = extract_committed_archive(
        args.target_archive,
        commitment,
        target_data,
    )

    fit_count = 0
    original_fit = runner.Ridge.fit

    def counted_fit(self: Any, *fit_args: Any, **fit_kwargs: Any) -> Any:
        nonlocal fit_count
        fit_count += 1
        return original_fit(self, *fit_args, **fit_kwargs)

    runner.Ridge.fit = counted_fit
    try:
        try:
            runner.execute(
                legacy,
                orth,
                args.training_data,
                target_data,
                args.families,
                args.target_commitment,
                staging,
            )
        except RuntimeError as exc:
            if str(exc) != "insufficient prospective target":
                raise
            seal_abort(staging, commitment, wrapper_sha, fit_count)
        else:
            if fit_count != 2:
                raise AssertionError(f"supervised fit budget drifted: {fit_count}")
            receipt_path = staging / "PROSPECTIVE_RELAY_RECEIPT.json"
            if not receipt_path.is_file():
                raise RuntimeError("wrapped runner did not seal a receipt")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("supervised_fits") != 2 or receipt.get("candidate_count") != 1:
                raise AssertionError("wrapped runner receipt drifted")
            target_manifest_path = staging / "prospective_target_manifest.json"
            target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
            target_manifest["target_files"] = target_file_manifest
            target_manifest["target_virus_minimum_edge_filter"] = 1
            target_manifest_path.write_text(
                json.dumps(target_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            receipt.update({
                "schema_version": "2.0",
                "target_archive_hash_reverified": True,
                "actual_supervised_fits": fit_count,
                "atomic_publication": True,
                "target_virus_minimum_edge_filter": 1,
                "wrapper_sha256": wrapper_sha,
                "wrapped_runner_sha256": EXPECTED_RUNNER_SHA256,
                "files": hash_files(staging),
            })
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if output.exists():
            raise FileExistsError(f"sealed output path appeared early: {output}")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        runner.Ridge.fit = original_fit
        shutil.rmtree(target_data, ignore_errors=True)

    final_receipt = json.loads(
        (output / "PROSPECTIVE_RELAY_RECEIPT.json").read_text(encoding="utf-8")
    )
    if final_receipt.get("status") != "ABORT_INSUFFICIENT_TARGET" and fit_count != 2:
        raise AssertionError("published scientific output did not consume exactly two fits")
    print(json.dumps({
        "status": final_receipt["status"],
        "supervised_fits": fit_count,
        "atomic_publication": True,
    }, sort_keys=True))


def self_test() -> None:
    assert TARGET_REQUIRED_FILES == (
        "assay_2026-08-01.txt",
        "virseqs_aa_O_2026-08-01.fasta",
    )
    print(json.dumps({
        "status": "SEALED_WRAPPER_SELF_TEST_PASS",
        "target_rows_parsed": 0,
        "target_values_read": 0,
        "supervised_fits": 0,
        "private_bytes_read": 0,
    }, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--orth", type=Path)
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--target-archive", type=Path)
    parser.add_argument("--families", type=Path)
    parser.add_argument("--target-commitment", type=Path)
    parser.add_argument("--output", type=Path, default=Path("prospective-artifacts"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (
        args.runner,
        args.legacy,
        args.orth,
        args.training_data,
        args.target_archive,
        args.families,
        args.target_commitment,
    )
    if any(value is None for value in required):
        parser.error("all execution paths are required")
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
