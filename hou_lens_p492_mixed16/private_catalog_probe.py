#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import requests

TAP = "https://archive.eso.org/tap_cat/sync"
PRIVATE_IN = Path("private_input")
PRIVATE_OUT = Path("private_output")


def query(session: requests.Session, adql: str) -> dict:
    r = session.get(TAP, params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql}, timeout=180)
    r.raise_for_status()
    p = r.json()
    names = [str(x.get("name", "")) for x in p.get("metadata", [])]
    return {"names": names, "rows": [dict(zip(names, row)) for row in p.get("data", [])]}


def main() -> None:
    manifest = json.loads((PRIVATE_IN / "P492_MIXED16_EXACT_REPLAY_MANIFEST.json").read_text())
    obj = manifest["objects"][3]
    source_id = obj["source_id"]
    session = requests.Session()
    session.headers["User-Agent"] = "HOU-LENS-P4.9.8-private-catalog-coordinate-recovery/1.0"

    tables = query(session, "SELECT table_name,description FROM TAP_SCHEMA.tables")
    candidates = []
    for row in tables["rows"]:
        name = str(row.get("table_name", ""))
        desc = str(row.get("description", ""))
        text = (name + " " + desc).lower()
        if "kids" in text or "kilo-degree" in text or "kilo degree" in text:
            candidates.append({"table_name": name, "description": desc})

    hits = []
    errors = []
    for c in candidates:
        table = c["table_name"]
        try:
            cols = query(session, "SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name='" + table.replace("'", "''") + "'")
            names = {str(r.get("column_name", "")) for r in cols["rows"]}
            if not {"ID", "RAJ2000", "DECJ2000"}.issubset(names):
                continue
            q = "SELECT TOP 5 ID,RAJ2000,DECJ2000 FROM " + table + " WHERE ID='" + source_id.replace("'", "''") + "'"
            res = query(session, q)
            for row in res["rows"]:
                hits.append({"table_name": table, "row": row})
        except Exception as exc:
            errors.append({"table_name": table, "error": type(exc).__name__ + ": " + str(exc)})

    PRIVATE_OUT.mkdir(exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": "HOU-LENS-P4.9.8-PRIVATE-KIDS-DR5-CATALOG-COORDINATE-PROBE",
        "opaque_index": 4,
        "manifest_ra_deg": obj["ra_deg"],
        "manifest_dec_deg": obj["dec_deg"],
        "source_id": source_id,
        "candidate_tables": candidates,
        "exact_id_hits": hits,
        "errors": errors,
        "scores_computed": False,
        "model_code_executed": False,
        "future_validation_touched": False,
    }
    (PRIVATE_OUT / "private_catalog_probe.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "PRIVATE_CATALOG_PROBE_WRITTEN", "candidate_table_count": len(candidates), "exact_id_hit_count": len(hits)}))


if __name__ == "__main__":
    main()
