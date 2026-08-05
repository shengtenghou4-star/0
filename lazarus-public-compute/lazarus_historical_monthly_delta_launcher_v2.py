#!/usr/bin/env python3
"""Pre-output compatibility launcher for the frozen historical label-time audit."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("historical_delta_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module_path = Path(__file__).with_name("lazarus_historical_monthly_delta_v1.py")
    audit = load(module_path)

    def fixed_sequence_support_training(
        source: Path,
        release: str,
        destination: Path,
    ) -> dict[str, str]:
        july = source.parent / "2026-07-01"
        destination.mkdir(parents=True)
        mapping = {
            "assay_2026-07-01.txt": source / f"assay_{release}.txt",
            "abs_2026-07-01.txt": july / "abs_2026-07-01.txt",
            "heavy_seqs_aa_2026-07-01.fasta": july / "heavy_seqs_aa_2026-07-01.fasta",
            "light_seqs_aa_2026-07-01.fasta": july / "light_seqs_aa_2026-07-01.fasta",
            "virseqs_aa_O_2026-07-01.fasta": source / f"virseqs_aa_O_{release}.fasta",
        }
        for canonical, original in mapping.items():
            if not original.is_file():
                raise FileNotFoundError(original)
            shutil.copyfile(original, destination / canonical)
        return {name: audit.sha256(destination / name) for name in mapping}

    audit.canonical_training = fixed_sequence_support_training
    result = audit.main()

    output = Path(sys.argv[sys.argv.index("--output") + 1])
    summary_path = output / "HISTORICAL_TEMPORAL_SUMMARY.json"
    receipt_path = output / "HISTORICAL_TEMPORAL_RECEIPT.json"
    summary = json.loads(summary_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    summary["fixed_sequence_support"] = {
        "antibody_metadata_and_heavy_light_release": "2026-07-01",
        "antibody_family_graph_release": "2026-07-01",
        "neutralization_labels_from_july_used_in_earlier_windows": False,
        "interpretation": "label-time audit, not strict information-time deployment validation",
    }
    if int(summary["windows_fitted"]) < 2:
        summary["status"] = "TEMPORAL_INSUFFICIENT"
        receipt["status"] = "TEMPORAL_INSUFFICIENT"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    receipt["fixed_sequence_support"] = summary["fixed_sequence_support"]
    receipt["files"] = audit.hash_files(output)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": summary["status"],
        "windows_fitted": summary["windows_fitted"],
        "historical_supervised_fits": summary["historical_supervised_fits"],
        "real_august_rows_read": 0,
    }, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
