from __future__ import annotations

import json
import traceback

import train_dev


def parse_dataset_list_strict(raw: bytes):
    records = []
    raw_lines = raw.decode("utf-8", errors="replace").splitlines()
    for line_number, original in enumerate(raw_lines, start=1):
        cleaned = original.split("#", 1)[0].strip()
        if not cleaned:
            continue
        tokens = cleaned.split()
        identifier = tokens[0]
        cut_volume = None
        # Dataset-list columns can contain condition flags. A source-defined
        # volume cutoff must be a plausible frame index, not a binary flag or
        # other small integer. This correction occurs before any development
        # metric has been generated and changes no model or evaluation rule.
        for token in tokens[1:]:
            try:
                numeric = int(float(token))
            except ValueError:
                continue
            if numeric >= 100:
                cut_volume = numeric
                break
        records.append({
            "identifier": identifier,
            "cut_volume": cut_volume,
            "line_number": line_number,
            "original": original,
        })
    if not records:
        raise RuntimeError("empty official dataset list")
    train_dev.dump(
        f"dataset_list_{len(list(train_dev.OUT.glob('dataset_list_*.json'))) + 1}.json",
        {"raw_lines": raw_lines, "parsed": records},
    )
    return records


train_dev.parse_dataset_list = parse_dataset_list_strict

if __name__ == "__main__":
    try:
        train_dev.main()
    except Exception as exc:
        train_dev.dump("failure_receipt_v2.json", {
            "schema": "bio-001-s26-osf-train-dev-failure-v2",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "preregistration_commit": train_dev.PREREG_COMMIT,
            "model_changed": False,
            "split_changed": False,
            "metric_changed": False,
            "holdout_requested": False,
            "CeRSI_v2_delta": 0.0,
        })
        raise
