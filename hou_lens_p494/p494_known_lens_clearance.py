#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

import requests

OUT = Path("p494_known_lens_clearance_artifact")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "HOU-LENS-P4.9.4-public-clearance/1"}
VIZIER = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"

CANDIDATES = [
    ("H24GOLD-CXCOJ100201+020330", "KiDSDR5 J095945.830+023117.48", 149.940961, 2.521523),
    ("H24GOLD-CXCOJ100201+020330", "KiDSDR5 J100210.175+015838.45", 150.542400, 1.977350),
    ("H24GOLD-CXCOJ100201+020330", "KiDSDR5 J100109.023+021350.95", 150.287599, 2.230822),
    ("H24GOLD-J0907+0003", "KiDSDR5 J090606.337+001142.57", 136.526406, 0.195159),
    ("H24GOLD-J0907+0003", "KiDSDR5 J090900.927+000021.95", 137.253865, 0.006098),
    ("H24GOLD-J0907+0003", "KiDSDR5 J090600.412+000153.86", 136.501717, 0.031630),
    ("H24GOLD-J1037+0018", "KiDSDR5 J103604.086+005403.00", 159.017029, 0.900835),
    ("H24GOLD-J1037+0018", "KiDSDR5 J103529.307+003155.99", 158.872114, 0.532221),
    ("H24GOLD-J1037+0018", "KiDSDR5 J103606.502+002558.78", 159.027094, 0.432995),
    ("H24GOLD-J1233-0227", "KiDSDR5 J123425.708-020012.41", 188.607118, -2.003450),
    ("H24GOLD-J1233-0227", "KiDSDR5 J123535.364-020755.56", 188.897353, -2.132101),
    ("H24GOLD-J1233-0227", "KiDSDR5 J123321.631-024832.04", 188.340131, -2.808902),
    ("H24GOLD-J1335+0118", "KiDSDR5 J133643.024+010220.96", 204.179268, 1.039156),
    ("H24GOLD-J1335+0118", "KiDSDR5 J133640.779+013003.20", 204.169915, 1.500889),
    ("H24GOLD-J1335+0118", "KiDSDR5 J133518.452+011247.55", 203.826886, 1.213209),
    ("H24GOLD-KIDS1042+0023", "KiDSDR5 J104300.049+003630.01", 160.750208, 0.608338),
    ("H24GOLD-KIDS1042+0023", "KiDSDR5 J104231.011+004813.58", 160.629215, 0.803773),
    ("H24GOLD-KIDS1042+0023", "KiDSDR5 J104228.693+000239.10", 160.619555, 0.044197),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request_bytes(url: str, *, params=None, timeout=180):
    r = requests.get(url, params=params, timeout=timeout, headers=UA)
    r.raise_for_status()
    return r


def find_col(fieldnames, candidates):
    lookup = {str(x).lower(): x for x in fieldnames or []}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    raise RuntimeError(f"missing columns {candidates}; got {fieldnames}")


def normalize_csv(payload: bytes, source: str):
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise RuntimeError(f"empty source {source}")
    ra_col = find_col(reader.fieldnames, ["RAJ2000", "RAdeg", "RA", "ra", "RA_ICRS"])
    dec_col = find_col(reader.fieldnames, ["DECJ2000", "DEJ2000", "DEdeg", "DEC", "dec", "DE_ICRS"])
    id_col = None
    lookup = {str(x).lower(): x for x in reader.fieldnames or []}
    for c in ["KIDS_ID", "ID", "Name", "name", "Lens", "Source", "recno"]:
        if c.lower() in lookup:
            id_col = lookup[c.lower()]
            break
    out = []
    for i, row in enumerate(rows):
        try:
            ra = float(row[ra_col])
            dec = float(row[dec_col])
        except (TypeError, ValueError):
            continue
        if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
            continue
        sid = str(row.get(id_col, i)) if id_col else str(i)
        out.append((source, sid, ra, dec))
    return out


def tap_table(table: str, source: str):
    q = f'SELECT * FROM "{table}"'
    r = request_bytes(VIZIER, params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": q})
    if r.text.lstrip().startswith("<?xml") or "QUERY_STATUS" in r.text[:1000]:
        raise RuntimeError(f"VizieR error for {table}: {r.text[:500]}")
    return normalize_csv(r.content, source), {
        "source": source,
        "url": r.url,
        "bytes": len(r.content),
        "sha256": sha256(r.content),
        "table": table,
    }


def sep_arcsec(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(math.radians, [ra1, dec1, ra2, dec2])
    sd = math.sin((d2-d1)/2.0)
    sr = math.sin((r2-r1)/2.0)
    a = sd*sd + math.cos(d1)*math.cos(d2)*sr*sr
    a = min(1.0, max(0.0, a))
    return math.degrees(2.0*math.asin(math.sqrt(a))) * 3600.0


def main():
    all_lenses = []
    receipts = []

    leiden_url = "https://kids.strw.leidenuniv.nl/DR4/data_files/lens_catalog.csv"
    r = request_bytes(leiden_url, timeout=120)
    leiden = normalize_csv(r.content, "kids_high_quality_268")
    if len(leiden) < 268:
        raise RuntimeError(f"Leiden rows {len(leiden)} < 268")
    all_lenses.extend(leiden)
    receipts.append({"source": "kids_high_quality_268", "url": r.url, "bytes": len(r.content), "sha256": sha256(r.content), "rows": len(leiden)})

    for table, source, minimum in [
        ("J/A+A/710/A366/cand", "liu2026_obscured", 2000),
        ("J/A+A/688/A34/tablea1", "teglie2024_grade1", 71),
        ("J/A+A/688/A34/tablec1", "teglie2024_grade2", 193),
    ]:
        rows, receipt = tap_table(table, source)
        if len(rows) < minimum:
            raise RuntimeError(f"{source} rows {len(rows)} < {minimum}")
        all_lenses.extend(rows)
        receipt["rows"] = len(rows)
        receipts.append(receipt)

    results = []
    for group, sid, ra, dec in CANDIDATES:
        nearest = None
        for source, lid, lra, ldec in all_lenses:
            s = sep_arcsec(ra, dec, lra, ldec)
            if nearest is None or s < nearest[0]:
                nearest = (s, source, lid, lra, ldec)
        assert nearest is not None
        results.append({
            "group_id": group,
            "control_source_id": sid,
            "ra_deg": ra,
            "dec_deg": dec,
            "nearest_known_lens_arcsec": nearest[0],
            "nearest_catalog": nearest[1],
            "nearest_source_id": nearest[2],
            "nearest_ra_deg": nearest[3],
            "nearest_dec_deg": nearest[4],
            "passes_5arcsec": nearest[0] >= 5.0,
        })

    status = "PASS_18_OF_18_KNOWN_LENS_CLEARANCE" if all(x["passes_5arcsec"] for x in results) else "FAIL_KNOWN_LENS_CLEARANCE"
    report = {
        "schema_version": 1,
        "status": status,
        "claim_boundary": "Public-source geometric exclusion audit only. No image access, model score, future-validation target, or blind search is performed.",
        "known_lens_rows": len(all_lenses),
        "source_receipts": receipts,
        "candidate_count": len(results),
        "minimum_clearance_arcsec": min(x["nearest_known_lens_arcsec"] for x in results),
        "results": results,
    }
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    (OUT / "P494_H24_KNOWN_LENS_CLEARANCE.json").write_bytes(payload)
    (OUT / "SHA256SUMS.txt").write_text(f"{sha256(payload)}  P494_H24_KNOWN_LENS_CLEARANCE.json\n")
    print(json.dumps({"status": status, "known_lens_rows": len(all_lenses), "minimum_clearance_arcsec": report["minimum_clearance_arcsec"]}, indent=2))
    if status != "PASS_18_OF_18_KNOWN_LENS_CLEARANCE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
