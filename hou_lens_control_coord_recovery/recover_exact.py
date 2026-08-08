#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import time

import requests

REQUEST_SHA = "4297bff7452b20aac3ab203821bed0bb7fabdb08b37aefe44500bd3052d0e99e"
RESULT_CERT_SHA = "16a6ffc5df2fc61289e506106dda702c157bc76261dd19a397efa6072b608625"
TAP_SYNC = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
TABLE = '"II/383/kids_dr5"'
BATCH = 8
PRIVATE_IN = pathlib.Path("private_input")
PRIVATE_OUT = pathlib.Path("private_output")


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def fail(message: str) -> None:
    # Never echo a private source identifier or coordinate into public logs.
    raise RuntimeError(message)


def esc(value: str) -> str:
    return value.replace("'", "''")


def query_batch(session: requests.Session, ids: list[str], batch_index: int) -> tuple[list[dict], dict]:
    quoted = ",".join("'" + esc(value) + "'" for value in ids)
    query = (
        f'SELECT "ID", "RAJ2000", "DEJ2000" FROM {TABLE} '
        f'WHERE "ID" IN ({quoted})'
    )
    started = time.monotonic()
    response = session.post(
        TAP_SYNC,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": query},
        timeout=180,
    )
    raw = response.content
    receipt = {
        "batch_index": batch_index,
        "requested_count": len(ids),
        "http_status": response.status_code,
        "elapsed_seconds": time.monotonic() - started,
        "response_bytes": len(raw),
        "response_sha256": sha_bytes(raw),
        "query_sha256": sha_bytes(query.encode()),
    }
    if response.status_code != 200:
        fail("VizieR TAP non-200 response")
    try:
        payload = json.loads(raw)
    except Exception:
        fail("VizieR TAP JSON parse failure")
    names = [str(item.get("name", "")) for item in payload.get("metadata", [])]
    if names != ["ID", "RAJ2000", "DEJ2000"]:
        fail("VizieR TAP response-column drift")
    rows = [dict(zip(names, row)) for row in payload.get("data", [])]
    requested = set(ids)
    seen: set[str] = set()
    output = []
    for row in rows:
        source_id = str(row["ID"])
        if source_id not in requested:
            fail("VizieR TAP scope drift")
        if source_id in seen:
            fail("VizieR TAP duplicate source ID")
        seen.add(source_id)
        ra = float(row["RAJ2000"])
        dec = float(row["DEJ2000"])
        if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
            fail("VizieR TAP invalid coordinate")
        output.append({"source_id": source_id, "ra_deg": ra, "dec_deg": dec})
    if seen != requested:
        fail("VizieR TAP source-ID shortfall")
    receipt["returned_count"] = len(output)
    return output, receipt


def main() -> None:
    request_path = PRIVATE_IN / "request.json"
    cert_path = PRIVATE_IN / "result-cert.pem"
    if sha_file(request_path) != REQUEST_SHA:
        fail("private request SHA gate failed")
    if sha_file(cert_path) != RESULT_CERT_SHA:
        fail("result certificate SHA gate failed")

    request = json.loads(request_path.read_text())
    rows = request.get("requests", [])
    if request.get("request_count") != 96 or len(rows) != 96:
        fail("request-count gate failed")
    ids = [str(row["source_id"]) for row in rows]
    if len(set(ids)) != 96:
        fail("source-ID uniqueness gate failed")
    if any(set(row) != {"family", "source_id", "tile_id"} for row in rows):
        fail("request-schema gate failed")

    PRIVATE_OUT.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "HOU-LENS-P4.9-control-coordinate-provenance-recovery/1.0"
    recovered_by_id: dict[str, dict] = {}
    batch_receipts = []
    for offset in range(0, len(ids), BATCH):
        batch_ids = ids[offset : offset + BATCH]
        recovered, receipt = query_batch(session, batch_ids, offset // BATCH + 1)
        batch_receipts.append(receipt)
        for item in recovered:
            recovered_by_id[item["source_id"]] = item
        print(
            f"batch {offset // BATCH + 1:02d}/{(len(ids) + BATCH - 1) // BATCH} exact-ID coordinates recovered; total={len(recovered_by_id)}/96",
            flush=True,
        )

    if set(recovered_by_id) != set(ids):
        fail("final coordinate-recovery set mismatch")
    recovered_rows = []
    for row in rows:
        c = recovered_by_id[row["source_id"]]
        recovered_rows.append(
            {
                "family": row["family"],
                "source_id": row["source_id"],
                "tile_id": row["tile_id"],
                "ra_deg": c["ra_deg"],
                "dec_deg": c["dec_deg"],
            }
        )

    result = {
        "schema_version": 1,
        "protocol": "HOU-LENS-P4.9-CONTROL-COORDINATE-PROVENANCE-RECOVERY-VIZIER",
        "status": "PASS_EXACT_96_OF_96_SOURCE_ID_COORDINATE_RECOVERY",
        "catalog": "VizieR II/383/kids_dr5 (ESO KiDS-DR5 multi-band source catalog mirror)",
        "request_sha256": REQUEST_SHA,
        "request_count": 96,
        "recovered_count": 96,
        "batch_count": len(batch_receipts),
        "batch_receipts": batch_receipts,
        "recovered": recovered_rows,
        "matching_performed": False,
        "selection_performed": False,
        "cube_sha_consulted": False,
        "scores_consulted": False,
        "claim_boundary": "Exact source-ID lookup of authoritative catalog coordinates for already-frozen P4.9 controls only.",
    }
    (PRIVATE_OUT / "coordinate_recovery.json").write_bytes(canonical(result))
    public = {
        "schema_version": 1,
        "protocol": result["protocol"],
        "status": result["status"],
        "request_sha256": REQUEST_SHA,
        "request_count": 96,
        "recovered_count": 96,
        "batch_count": len(batch_receipts),
        "private_result_sha256": sha_file(PRIVATE_OUT / "coordinate_recovery.json"),
        "matching_performed": False,
        "selection_performed": False,
        "cube_sha_consulted": False,
        "scores_consulted": False,
        "private_plaintext_emitted": False,
    }
    pathlib.Path("public_result").mkdir(exist_ok=True)
    pathlib.Path("public_result/public_summary.pre_encryption.json").write_bytes(canonical(public))
    print(json.dumps({k: public[k] for k in ["status", "request_count", "recovered_count", "batch_count", "private_result_sha256"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
