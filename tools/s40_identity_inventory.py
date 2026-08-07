from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scipy.io import whosmat

ROOT = Path("s40_data/training")
OUT = Path("s40_phaseA")
OUT.mkdir(parents=True, exist_ok=True)

IDENTITY_RE = re.compile(r"(neuron|cell|name|label|identity|ident|track|reference|registration|correspond|atlas|ava|bfp)", re.I)
POSITION_RE = re.compile(r"(pos|position|coord|centroid|xyz|location|point)", re.I)
QUALITY_RE = re.compile(r"(quality|confidence|score|valid|good|bad|error)", re.I)
FLUOR_RE = re.compile(r"(ratio|gcamp|rfp|bfp|fluor|signal|activity|calcium)", re.I)
TEXT_EXT = {".txt", ".csv", ".tsv", ".json", ".yaml", ".yml", ".md"}


def safe_text(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    # Structural phase: only small human-readable metadata. Large tables are not opened.
    if len(raw) > 256_000:
        return {"bytes": len(raw), "opened": False, "reason": "too_large_for_structural_text_inventory"}
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return {
        "bytes": len(raw),
        "opened": True,
        "line_count": len(lines),
        "preview": lines[:80],
    }


def mat_inventory(path: Path) -> dict[str, Any]:
    try:
        items = whosmat(path)
        variables = [{"name": n, "shape": list(shape), "class": cls} for n, shape, cls in items]
        return {"parser": "scipy.whosmat", "variables": variables, "error": None}
    except Exception as e:
        return {"parser": "scipy.whosmat", "variables": [], "error": f"{type(e).__name__}: {e}"}


def classify(records: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    explicit_cross = []
    correspondence = []
    stable_ratio = []
    for rid, rec in records.items():
        candidate_names = []
        candidate_corr = []
        ratio_shapes = []
        for f in rec["files"]:
            for v in f.get("variables", []):
                name = v["name"]
                if IDENTITY_RE.search(name):
                    candidate_names.append({"file": f["path"], **v})
                if re.search(r"(reference|registration|correspond|atlas|match|mapping|trackid)", name, re.I):
                    candidate_corr.append({"file": f["path"], **v})
                if name.lower() == "ratio2" and len(v["shape"]) == 2 and min(v["shape"]) >= 5:
                    ratio_shapes.append({"file": f["path"], **v})
            if f.get("text", {}).get("opened"):
                joined = "\n".join(f["text"].get("preview", []))
                if re.search(r"\b(AVA[LR]?|AVAL|AVAR|neuron name|cell name|identity|atlas)\b", joined, re.I):
                    candidate_names.append({"file": f["path"], "text_identity_marker": True})
                if re.search(r"(cross[- ]?record|correspondence|registration mapping|reference neuron)", joined, re.I):
                    candidate_corr.append({"file": f["path"], "text_correspondence_marker": True})
        if candidate_names:
            explicit_cross.append(rid)
        if candidate_corr:
            correspondence.append(rid)
        if ratio_shapes:
            stable_ratio.append(rid)
        rec["identity_candidate_fields"] = candidate_names
        rec["correspondence_candidate_fields"] = candidate_corr
        rec["ratio2_structural_traces"] = ratio_shapes

    # Conservative: mere identity-looking field names do not prove cross-record identity.
    # I3/I2 require explicit machine-verifiable cross-record mappings, which Phase A can flag but not infer.
    # The script only upgrades automatically to I1 from a stable 2D Ratio2 trace in >=3/4 recordings.
    if len(stable_ratio) >= 3:
        cls = "I1_WITHIN_RECORD_STABLE_IDENTITY_ONLY"
    else:
        cls = "I0_IDENTITY_UNRESOLVED"
    evidence = {
        "records_with_identity_candidate_fields": explicit_cross,
        "records_with_correspondence_candidate_fields": correspondence,
        "records_with_structural_ratio2_traces": stable_ratio,
        "automatic_cross_record_upgrade_prohibited": True,
        "reason": "Cross-record identity is not inferred from field names, channel number, activity or similarity."
    }
    return cls, evidence


def main() -> None:
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file())
    record_dirs = sorted(p for p in ROOT.rglob("*_MS") if p.is_dir())
    records: dict[str, Any] = {}
    for folder in record_dirs:
        rid = folder.name[:-3]
        rec_files = []
        for p in sorted(x for x in folder.rglob("*") if x.is_file()):
            row: dict[str, Any] = {"path": str(p.relative_to(ROOT)), "suffix": p.suffix.lower(), "bytes": p.stat().st_size}
            if p.suffix.lower() == ".mat":
                row.update(mat_inventory(p))
            elif p.suffix.lower() in TEXT_EXT:
                row["text"] = safe_text(p)
            rec_files.append(row)
        records[rid] = {"folder": str(folder.relative_to(ROOT)), "files": rec_files}

    top_text = []
    for p in paths:
        if any(str(p).startswith(str(d) + "/") for d in record_dirs):
            continue
        if p.suffix.lower() in TEXT_EXT:
            top_text.append({"path": str(p.relative_to(ROOT)), "text": safe_text(p)})

    identity_class, classification_evidence = classify(records)
    result = {
        "schema": "bio-001-s40-phaseA-identity-inventory-v1",
        "structural_only": True,
        "numeric_neural_behavior_values_inspected": False,
        "record_count": len(records),
        "records": records,
        "top_level_text_metadata": top_text,
        "identity_class": identity_class,
        "classification_evidence": classification_evidence,
        "phaseB_route": (
            "WITHIN_RECORD_INNOVATION_ONLY" if identity_class == "I1_WITHIN_RECORD_STABLE_IDENTITY_ONLY"
            else "STOP_IDENTITY_NOT_IDENTIFIABLE"
        ),
        "cross_record_biological_identity_claim_allowed": False,
    }
    (OUT / "s40_phaseA_identity_inventory.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "identity_class": identity_class,
        "record_count": len(records),
        "phaseB_route": result["phaseB_route"],
        "cross_record_claim_allowed": False,
    }))


if __name__ == "__main__":
    main()
