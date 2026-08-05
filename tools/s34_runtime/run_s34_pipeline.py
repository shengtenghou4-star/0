from __future__ import annotations

"""S34 result-preceding source-structure adapter.

The frozen S27/S28 loader and pipeline remain byte-identical. This wrapper
applies the S34 MATLAB structured-scalar field-access erratum before importing
and running the frozen training/development pipeline. It does not transform,
squeeze, rescale, select, or inspect numeric field values.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import official_loader_port as loader

FIELD_ORDER = ("ethogram", "x_pos", "y_pos", "v", "pc1_2", "pc_3")


def extract_behavior_cell_source_native(behavior: np.ndarray) -> tuple[np.ndarray, ...]:
    cell = behavior[0][0]
    if isinstance(cell, np.void) and cell.dtype.names:
        observed = tuple(cell.dtype.names)
        if observed != FIELD_ORDER:
            raise ValueError(
                f"unexpected behavior struct fields: expected={FIELD_ORDER}, observed={observed}"
            )
        return tuple(np.asarray(cell[name]) for name in FIELD_ORDER)

    fields = tuple(cell.T)
    if len(fields) != 6:
        raise ValueError(f"expected six legacy behavior fields, found {len(fields)}")
    return tuple(np.asarray(field) for field in fields)


loader._extract_behavior_cell = extract_behavior_cell_source_native

import train_dev_pipeline as pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = pipeline.run(args.training_root, args.development_root, args.output)
    erratum_receipt = {
        "schema": "bio-001-s34-matlab-struct-adapter-receipt-v1",
        "private_erratum_commit": "935ce787ba3a302782ad3de2346912ab493ffeaf",
        "field_order": list(FIELD_ORDER),
        "numeric_transformation": False,
        "squeeze_or_reshape_fields": False,
        "frozen_loader_git_blob": "5d52037432c93ed8b5531faf4edb15533e7c5504",
        "frozen_pipeline_git_blob": "a8f5c47f63bc4a2dedd06dd4269327205c816ae9",
        "holdout_requested": False,
        "holdout_opened": False,
        "pipeline_status": receipt["status"],
        "CeRSI_v2_delta": 0.0,
    }
    pipeline.write_json(args.output / "source_structure_erratum_receipt.json", erratum_receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
