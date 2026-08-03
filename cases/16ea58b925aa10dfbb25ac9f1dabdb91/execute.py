from __future__ import annotations

import traceback

import run
import transport


def preloaded_acquire(role: str, name: str, file_id: int):
    if name in run.FORBIDDEN or name not in run.ALLOWED_NAMES or file_id not in run.ALLOWED_IDS:
        raise RuntimeError(f"preloaded acquisition rejected: {name}/{file_id}")
    path = run.RAW / name
    if not path.exists() or path.read_bytes()[:8] != run.SIG:
        raise RuntimeError(f"preloaded HDF5 missing or invalid: {name}")
    return path


if __name__ == "__main__":
    try:
        transport.download_all()
        run.acquire = preloaded_acquire
        run.main()
    except Exception as exc:
        run.dump("failure_receipt.json", {
            "type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "holdout_requested": False,
        })
        raise
