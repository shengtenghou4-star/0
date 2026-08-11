#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import requests

TAP = "https://archive.eso.org/tap_cat/sync"
TABLES = ("safcat.ESO_P_KIDS_DR5_1_1", "safcat.ESO_P_KIDS_DR5_2_1")
PRIVATE_IN = Path("private_input")
PRIVATE_OUT = Path("private_output")


def query(session: requests.Session, adql: str) -> dict:
    r = session.get(TAP, params={"REQUEST":"doQuery","LANG":"ADQL","FORMAT":"json","QUERY":adql}, timeout=180)
    r.raise_for_status()
    p=r.json()
    names=[str(x.get("name","")) for x in p.get("metadata",[])]
    return {"names":names,"rows":[dict(zip(names,row)) for row in p.get("data",[])]}


def main() -> None:
    manifest=json.loads((PRIVATE_IN/"GRADEA_PROSPECTIVE16_EXACT_REPLAY_MANIFEST.json").read_text())
    controls=[o for o in manifest["objects"] if o["role"]=="control"]
    ids=sorted({o["source_id"] for o in controls})
    if len(ids)!=48:
        raise RuntimeError("expected 48 unique control source IDs")
    session=requests.Session()
    session.headers["User-Agent"]="HOU-LENS-P4.9.8-GradeA-private-bulk-catalog-probe/1.0"
    quoted=",".join("'"+x.replace("'","''")+"'" for x in ids)
    hits=[]; errors=[]
    for table in TABLES:
        try:
            res=query(session,f"SELECT ID,RAJ2000,DECJ2000 FROM {table} WHERE ID IN ({quoted})")
            for row in res["rows"]:
                hits.append({"table_name":table,"ID":str(row["ID"]),"RAJ2000":float(row["RAJ2000"]),"DECJ2000":float(row["DECJ2000"])})
        except Exception as exc:
            errors.append({"table_name":table,"error":type(exc).__name__+": "+str(exc)})
    by={}
    for h in hits:
        by.setdefault(h["ID"],[]).append(h)
    records=[]
    for obj in controls:
        hh=by.get(obj["source_id"],[])
        records.append({
            "target_id":obj["target_id"],"source_id":obj["source_id"],
            "manifest_ra_deg":obj["ra_deg"],"manifest_dec_deg":obj["dec_deg"],
            "catalog_hits":hh,
        })
    unique_hit_count=sum(len(r["catalog_hits"])==1 for r in records)
    PRIVATE_OUT.mkdir(exist_ok=True)
    payload={
        "schema_version":1,
        "protocol":"HOU-LENS-P4.9.8-GRADEA-PRIVATE-BULK-KIDS-DR5-CENTROID-PROBE",
        "control_count":48,
        "unique_exact_id_hit_count":unique_hit_count,
        "records":records,
        "errors":errors,
        "scores_computed":False,
        "model_code_executed":False,
        "future_validation_touched":False,
    }
    (PRIVATE_OUT/"private_bulk_catalog_probe.json").write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"PRIVATE_BULK_CATALOG_PROBE_WRITTEN","controls":48,"unique_exact_id_hits":unique_hit_count,"errors":len(errors)}))

if __name__=="__main__":
    main()
