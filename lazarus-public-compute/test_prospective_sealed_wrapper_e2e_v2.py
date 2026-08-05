#!/usr/bin/env python3
"""Synthetic public-data-only end-to-end rehearsal of the sealed LAZARUS transaction."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def choose_synthetic_pairs(
    train: list[Any],
    family_by: dict[str, str],
    training_keys: set[tuple[str, str]],
) -> list[tuple[str, str, str]]:
    antibodies_by_family: dict[str, list[str]] = defaultdict(list)
    for antibody in sorted({row.antibody for row in train}):
        family = family_by.get(antibody)
        if family is not None:
            antibodies_by_family[family].append(antibody)
    usable_families = [
        family
        for family, antibodies in sorted(
            antibodies_by_family.items(), key=lambda item: (-len(item[1]), item[0])
        )
        if len(antibodies) >= 4
    ][:6]
    if len(usable_families) < 3:
        raise RuntimeError("fewer than three usable antibody families")
    selected_by_family = {
        family: antibodies_by_family[family][:8]
        for family in usable_families
    }
    viruses = sorted({row.virus for row in train})
    chosen: list[tuple[str, str, str]] = []
    chosen_viruses = 0
    for virus in viruses:
        available = {
            family: [
                antibody
                for antibody in selected_by_family[family]
                if (antibody, virus) not in training_keys
            ]
            for family in usable_families
        }
        picks: list[tuple[str, str, str]] = []
        cursors = {family: 0 for family in usable_families}
        while len(picks) < 10:
            progressed = False
            for family in usable_families:
                cursor = cursors[family]
                if cursor < len(available[family]):
                    antibody = available[family][cursor]
                    cursors[family] += 1
                    picks.append((antibody, family, virus))
                    progressed = True
                    if len(picks) == 10:
                        break
            if not progressed:
                break
        if len(picks) < 10:
            continue
        chosen.extend(picks)
        chosen_viruses += 1
        if chosen_viruses == 12:
            break
    if len(chosen) != 120:
        raise RuntimeError(f"could not construct 120 missing public-data pairs: {len(chosen)}")
    if len({antibody for antibody, _, _ in chosen}) < 8:
        raise RuntimeError("synthetic target has fewer than eight antibodies")
    if len({virus for _, _, virus in chosen}) < 10:
        raise RuntimeError("synthetic target has fewer than ten viruses")
    if len({family for _, family, _ in chosen}) < 3:
        raise RuntimeError("synthetic target has fewer than three families")
    if any((antibody, virus) in training_keys for antibody, _, virus in chosen):
        raise AssertionError("synthetic target overlaps July training")
    return chosen


def label_rows(pairs: list[tuple[str, str, str]]) -> list[tuple[str, str, str, str]]:
    labels = ["10" for _ in pairs]
    potent: set[int] = set()
    resistant: set[int] = set()
    by_virus: dict[str, list[int]] = defaultdict(list)
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, (_, family, virus) in enumerate(pairs):
        by_virus[virus].append(index)
        by_family[family].append(index)
    for indices in by_virus.values():
        potent.add(indices[0])
    for indices in by_family.values():
        potent.add(indices[0])
    for index in range(len(pairs)):
        if len(potent) >= 40:
            break
        potent.add(index)
    for indices in by_virus.values():
        for index in indices:
            if index not in potent:
                resistant.add(index)
                break
    for indices in by_family.values():
        for index in indices:
            if index not in potent:
                resistant.add(index)
                break
    for index in range(len(pairs)):
        if len(resistant) >= 40:
            break
        if index not in potent:
            resistant.add(index)
    if potent & resistant:
        raise AssertionError("synthetic target labels overlap")
    for index in potent:
        labels[index] = "0.1"
    for index in resistant:
        labels[index] = "100"
    return [
        (antibody, family, virus, labels[index])
        for index, (antibody, family, virus) in enumerate(pairs)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--orth", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shutil.rmtree(args.output, ignore_errors=True)
    args.output.mkdir(parents=True)
    workspace = args.output / "workspace"
    workspace.mkdir()

    legacy = load_module(args.legacy, "lazarus_synthetic_legacy")
    runner = load_module(args.runner, "lazarus_synthetic_runner")
    family_by, _ = legacy.load_family_map(args.families)
    train, _, _, training_keys = runner.training_pairs(
        legacy, args.training_data, family_by
    )
    selected = choose_synthetic_pairs(train, family_by, training_keys)
    labeled = label_rows(selected)

    assay = workspace / "assay_2026-08-01.txt"
    assay.write_text(
        "Antibody\tVirus\tIC50\n"
        + "".join(
            f"{antibody}\t{virus}\t{value}\n"
            for antibody, _, virus, value in labeled
        ),
        encoding="utf-8",
    )
    target_fasta = workspace / "virseqs_aa_O_2026-08-01.fasta"
    shutil.copyfile(
        args.training_data / "virseqs_aa_O_2026-07-01.fasta",
        target_fasta,
    )
    archive = workspace / "SYNTHETIC_CATNAP_2026_08_01.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(assay, arcname=f"synthetic/{assay.name}")
        handle.add(target_fasta, arcname=f"synthetic/{target_fasta.name}")
    commitment = {
        "schema_version": "synthetic-rehearsal-1.0",
        "status": "TARGET_ARCHIVE_HASH_COMMITTED",
        "release": "2026-08-01",
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "synthetic_only": True,
        "scientific_interpretation_forbidden": True,
    }
    commitment_path = workspace / "SYNTHETIC_TARGET_COMMITMENT.json"
    commitment_path.write_text(
        json.dumps(commitment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sealed_output = workspace / "sealed-output"
    command = [
        sys.executable,
        str(args.wrapper),
        "--runner", str(args.runner),
        "--legacy", str(args.legacy),
        "--orth", str(args.orth),
        "--training-data", str(args.training_data),
        "--target-archive", str(archive),
        "--families", str(args.families),
        "--target-commitment", str(commitment_path),
        "--output", str(sealed_output),
    ]
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"sealed synthetic rehearsal failed: {result.stdout[-4000:]}")
    if not sealed_output.is_dir():
        raise RuntimeError("atomic sealed output was not published")
    if (workspace / ".sealed-output.staging").exists():
        raise RuntimeError("staging directory survived atomic publication")
    if (workspace / ".sealed-output.target-data").exists():
        raise RuntimeError("temporary target directory survived execution")

    relay_receipt = json.loads(
        (sealed_output / "PROSPECTIVE_RELAY_RECEIPT.json").read_text(encoding="utf-8")
    )
    target_manifest = json.loads(
        (sealed_output / "prospective_target_manifest.json").read_text(encoding="utf-8")
    )
    adjudication = json.loads(
        (sealed_output / "prospective_adjudication.json").read_text(encoding="utf-8")
    )
    counts = target_manifest["counts"]
    if relay_receipt.get("target_archive_hash_reverified") is not True:
        raise AssertionError("actual archive hash was not reverified")
    if relay_receipt.get("actual_supervised_fits") != 2:
        raise AssertionError(relay_receipt)
    if relay_receipt.get("supervised_fits") != 2:
        raise AssertionError(relay_receipt)
    if relay_receipt.get("candidate_count") != 1:
        raise AssertionError(relay_receipt)
    if relay_receipt.get("atomic_publication") is not True:
        raise AssertionError(relay_receipt)
    if relay_receipt.get("target_virus_minimum_edge_filter") != 1:
        raise AssertionError(relay_receipt)
    if target_manifest.get("training_overlap_pairs") != 0:
        raise AssertionError(target_manifest)
    minimums = {
        "entries": 100,
        "potent_entries": 25,
        "resistant_entries": 25,
        "unique_antibodies": 8,
        "unique_viruses": 10,
        "antibody_families": 3,
    }
    for name, required in minimums.items():
        if int(counts[name]) < required:
            raise AssertionError((name, counts[name], required))
    if adjudication.get("supervised_fits") != 2:
        raise AssertionError(adjudication)

    output_hashes = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(sealed_output.iterdir())
        if path.is_file()
    }
    protocol_receipt = {
        "schema_version": "2.0",
        "status": "SYNTHETIC_SEALED_E2E_PASS",
        "synthetic_only": True,
        "scientific_interpretation_forbidden": True,
        "synthetic_source": "July 2026 public CATNAP sequences plus July-absent public entity pairs",
        "synthetic_target_rows": len(labeled),
        "synthetic_potent_rows": int(counts["potent_entries"]),
        "synthetic_resistant_rows": int(counts["resistant_entries"]),
        "synthetic_antibodies": int(counts["unique_antibodies"]),
        "synthetic_viruses": int(counts["unique_viruses"]),
        "synthetic_families": int(counts["antibody_families"]),
        "training_overlap_pairs": int(target_manifest["training_overlap_pairs"]),
        "actual_archive_hash_reverified": True,
        "actual_supervised_fits": 2,
        "prospective_scientific_fits": 0,
        "real_august_target_rows_read": 0,
        "real_august_target_values_read": 0,
        "private_bytes_read": 0,
        "atomic_publication_verified": True,
        "temporary_target_deleted": True,
        "staging_directory_deleted": True,
        "wrapper_sha256": sha256(args.wrapper),
        "wrapped_runner_sha256": sha256(args.runner),
        "orth_helper_sha256": sha256(args.orth),
        "synthetic_archive_sha256": commitment["archive_sha256"],
        "wrapped_output_status": adjudication["status"],
        "wrapped_output_files": output_hashes,
        "wrapper_stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
    }
    receipt_path = args.output / "SYNTHETIC_SEALED_E2E_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(protocol_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(workspace)
    print(json.dumps({
        "status": protocol_receipt["status"],
        "synthetic_target_rows": protocol_receipt["synthetic_target_rows"],
        "actual_supervised_fits": 2,
        "real_august_target_rows_read": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
