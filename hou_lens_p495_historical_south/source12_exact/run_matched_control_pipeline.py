#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_selected_target_pipeline as base  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    payload = json.loads(args.targets.read_text())
    matches = [row for row in payload["targets"] if row["target_id"] == args.target_id]
    if len(matches) != 1:
        raise RuntimeError("matched control is not uniquely defined")
    control = matches[0]
    if control.get("sample") != "matched_control" or control.get("split") != "development":
        raise RuntimeError("matched-control split contract violated")

    sys.argv = [
        "run_selected_target_pipeline.py",
        "--targets", str(args.targets),
        "--target-id", args.target_id,
        "--output", str(args.output),
        "--radius", str(args.radius),
        "--attempts", str(args.attempts),
        "--timeout", str(args.timeout),
    ]
    base.main()

    receipt_path = args.output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    object_meta = receipt.pop("published_reference")
    object_meta.update({
        "sample": "matched_control",
        "matched_positive_id": control["matched_positive_id"],
        "match_tier": control["match_tier"],
        "match_distance": control["match_distance"],
    })
    receipt["stage"] = "kids_dr5_phase4_matched_control_science_pipeline"
    receipt["object"] = object_meta
    receipt["claim_boundary"] = "Acquires and preprocesses one frozen matched development control only; no holdout or blind target is queried or scored."
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": receipt["status"],
        "target_id": object_meta["target_id"],
        "matched_positive_id": object_meta["matched_positive_id"],
        "cutout_count": receipt["cutout_count"],
        "raw_cube_shape": receipt["raw_cube_shape"],
    }, indent=2))


if __name__ == "__main__":
    main()
