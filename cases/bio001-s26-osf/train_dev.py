from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import scipy.io
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
RAW = ROOT / "raw"
EXTRACTED = ROOT / "extracted"
for directory in (OUT, RAW, EXTRACTED):
    directory.mkdir(parents=True, exist_ok=True)

PREREG_COMMIT = "f8652a7883fd63b5e9574ce10daef4365b41344e"
SOURCE_CODE_COMMIT = "226a27533d8b99ebf4448f0214fa19caf6a71a2b"
ARCHIVES = {
    "training": {
        "name": "AML310_moving.tar.gz",
        "url": "https://osf.io/download/evhrg/",
        "bytes": 348444164,
        "sha256": "144126ee9a49d311c3393deea434e1a0963d55de35318e25d98d48f9c175250a",
    },
    "development": {
        "name": "AML310_transition.tar.gz",
        "url": "https://osf.io/download/4u8h2/",
        "bytes": 486820852,
        "sha256": "c524561ae35c09e65b9c0cd4b8ac84df01bf40cbf067c036a3e2a4a5b59fe63e",
    },
}
SEALED_HOLDOUT = {
    "name": "AML32_chip.tar.gz",
    "url": "https://osf.io/download/r2jne/",
    "bytes": 314689799,
    "sha256": "e72be6957c37f70b64cabb51c44ebd0c1501eb11b588878b13823d09067bac5e",
}
LAGS = [0, 1, 2, 3, 5, 8, 13]
HORIZONS = [1, 5, 10]
PRIMARY_HORIZON = 10
STRIDE = 5
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
NEURAL_NAMES = [
    "population_mean",
    "population_standard_deviation",
    "population_q10",
    "population_q25",
    "population_median",
    "population_q75",
    "population_q90",
    "fraction_above_plus_1_recording_z",
    "fraction_below_minus_1_recording_z",
    "mean_positive_temporal_difference",
    "mean_negative_temporal_difference",
]
USER_AGENT = "BIO-001-S26-OSF-neural-velocity/1.0"


def dump(name: str, payload: Any) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(role: str, spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if spec["name"] == SEALED_HOLDOUT["name"] or spec["url"] == SEALED_HOLDOUT["url"]:
        raise RuntimeError("sealed holdout acquisition rejected")
    destination = RAW / spec["name"]
    digest = hashlib.sha256()
    total = 0
    with requests.get(
        spec["url"],
        stream=True,
        timeout=(30, 600),
        allow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        final_url = response.url
        content_type = response.headers.get("content-type")
    actual_digest = digest.hexdigest()
    if total != spec["bytes"] or actual_digest != spec["sha256"]:
        raise RuntimeError(
            f"{role} archive identity mismatch: bytes={total}/{spec['bytes']} "
            f"sha={actual_digest}/{spec['sha256']}"
        )
    return destination, {
        "role": role,
        "filename": spec["name"],
        "source_url": spec["url"],
        "final_url": final_url,
        "content_type": content_type,
        "bytes": total,
        "sha256": actual_digest,
        "official_identity_match": True,
    }


def safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    resolved = (destination / member.name).resolve()
    if destination.resolve() not in resolved.parents and resolved != destination.resolve():
        raise RuntimeError(f"unsafe tar path: {member.name}")
    tar.extract(member, destination)


def parse_dataset_list(raw: bytes) -> list[dict[str, Any]]:
    records = []
    for line_number, original in enumerate(raw.decode("utf-8", errors="replace").splitlines(), start=1):
        cleaned = original.split("#", 1)[0].strip()
        if not cleaned:
            continue
        tokens = cleaned.split()
        identifier = tokens[0]
        cutoff = None
        if len(tokens) >= 2:
            try:
                cutoff = int(float(tokens[1]))
            except ValueError:
                cutoff = None
        records.append({
            "identifier": identifier,
            "cut_volume": cutoff,
            "line_number": line_number,
            "original": original,
        })
    if not records:
        raise RuntimeError("empty official dataset list")
    return records


def extract_selected(role: str, archive: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    destination = EXTRACTED / role
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        list_members = [m for m in members if m.isfile() and m.name.endswith("_datasets.txt")]
        if len(list_members) != 1:
            raise RuntimeError(f"{role}: expected exactly one datasets list, got {len(list_members)}")
        dataset_records = parse_dataset_list(tar.extractfile(list_members[0]).read())
        identifiers = {record["identifier"] for record in dataset_records}
        selected_members = [
            member for member in members
            if member.isfile()
            and (
                member == list_members[0]
                or any(identifier in member.name for identifier in identifiers)
            )
        ]
        for member in selected_members:
            safe_extract_member(tar, member, destination)
    recording_receipts = []
    for record in dataset_records:
        candidates = sorted(
            path.parent for path in destination.rglob("heatDataMS.mat")
            if record["identifier"] in str(path)
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"{role}/{record['identifier']}: expected one heatDataMS recording, got {candidates}"
            )
        folder = candidates[0]
        required = ["heatDataMS.mat", "centerline.mat"]
        missing = [name for name in required if not (folder / name).exists()]
        if missing:
            raise RuntimeError(f"{role}/{record['identifier']}: missing {missing}")
        recording_receipts.append({
            **record,
            "folder": str(folder.relative_to(destination)),
            "heatDataMS_sha256": sha256(folder / "heatDataMS.mat"),
            "centerline_sha256": sha256(folder / "centerline.mat"),
        })
    return recording_receipts, {
        "role": role,
        "archive": archive.name,
        "dataset_list_member": list_members[0].name,
        "recording_count": len(recording_receipts),
        "recordings": recording_receipts,
    }


def as_vector(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).squeeze()
    if vector.ndim != 1:
        raise ValueError(f"{label} is not a vector: {vector.shape}")
    return vector


def struct_field(structure: Any, field: str) -> Any:
    if hasattr(structure, field):
        return getattr(structure, field)
    if isinstance(structure, np.void) and structure.dtype.names and field in structure.dtype.names:
        return structure[field]
    if isinstance(structure, np.ndarray) and structure.dtype.names and field in structure.dtype.names:
        return structure[field].squeeze()
    raise KeyError(f"behavior field {field} not found in {type(structure)}")


def orient_neural(matrix: np.ndarray, frame_hint: int) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"neural matrix must be 2D: {array.shape}")
    if abs(array.shape[1] - frame_hint) <= abs(array.shape[0] - frame_hint):
        return array
    return array.T


def interpolate_rows(matrix: np.ndarray) -> np.ndarray:
    result = np.empty_like(matrix, dtype=float)
    time = np.arange(matrix.shape[1])
    for index, row in enumerate(matrix):
        finite = np.isfinite(row)
        if not finite.any():
            result[index] = np.nan
            continue
        result[index] = np.interp(time, time[finite], row[finite])
    return result


def neural_summaries(neural: np.ndarray) -> np.ndarray:
    means = np.mean(neural, axis=1, keepdims=True)
    scales = np.std(neural, axis=1, keepdims=True)
    scales[scales < 1e-8] = 1.0
    z = (neural - means) / scales
    dz = np.diff(z, axis=1, prepend=z[:, :1])
    summaries = np.column_stack([
        np.mean(z, axis=0),
        np.std(z, axis=0),
        np.quantile(z, 0.10, axis=0),
        np.quantile(z, 0.25, axis=0),
        np.quantile(z, 0.50, axis=0),
        np.quantile(z, 0.75, axis=0),
        np.quantile(z, 0.90, axis=0),
        np.mean(z > 1.0, axis=0),
        np.mean(z < -1.0, axis=0),
        np.mean(np.maximum(dz, 0.0), axis=0),
        np.mean(np.minimum(dz, 0.0), axis=0),
    ])
    if summaries.shape[1] != len(NEURAL_NAMES):
        raise AssertionError(summaries.shape)
    return summaries


@dataclass
class Recording:
    role: str
    identifier: str
    velocity: np.ndarray
    angular_velocity: np.ndarray
    neural_summary: np.ndarray
    frame_count: int
    neuron_count: int
    cut_volume: int | None
    source_folder: str


def load_recording(role: str, base: Path, receipt: dict[str, Any]) -> Recording:
    folder = base / receipt["folder"]
    data = scipy.io.loadmat(folder / "heatDataMS.mat", squeeze_me=True, struct_as_record=False)
    behavior = data["behavior"]
    velocity = as_vector(struct_field(behavior, "v"), "behavior.v")
    pc12 = np.asarray(struct_field(behavior, "pc1_2"), dtype=float).squeeze()
    if pc12.ndim != 2:
        raise ValueError(f"pc1_2 must be 2D, got {pc12.shape}")
    if pc12.shape[0] == 2:
        pc1, pc2 = pc12[0], pc12[1]
    elif pc12.shape[1] == 2:
        pc1, pc2 = pc12[:, 0], pc12[:, 1]
    else:
        raise ValueError(f"pc1_2 has no axis of length 2: {pc12.shape}")
    phase = np.unwrap(np.arctan2(pc2, pc1))
    angular_velocity = np.gradient(phase) * 6.0

    neural_key = "Ratio2" if "Ratio2" in data else "gPhotoCorr"
    neural = orient_neural(np.asarray(data[neural_key], dtype=float), len(velocity))
    n = min(len(velocity), len(angular_velocity), neural.shape[1])
    if receipt["cut_volume"] is not None:
        n = min(n, receipt["cut_volume"] + 1)
    velocity = velocity[:n]
    angular_velocity = angular_velocity[:n]
    neural = neural[:, :n]

    majority_finite = np.mean(np.isfinite(neural), axis=0) >= 0.5
    behavior_finite = np.isfinite(velocity) & np.isfinite(angular_velocity)
    valid = majority_finite & behavior_finite
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 100:
        raise ValueError(f"{receipt['identifier']}: fewer than 100 valid frames")
    start, stop = valid_indices[0], valid_indices[-1] + 1
    velocity = velocity[start:stop]
    angular_velocity = angular_velocity[start:stop]
    neural = neural[:, start:stop]
    keep_neurons = np.mean(np.isfinite(neural), axis=1) >= 0.5
    neural = neural[keep_neurons]
    if neural.shape[0] == 0:
        raise ValueError(f"{receipt['identifier']}: no usable neurons")
    neural = interpolate_rows(neural)
    neural = neural[np.all(np.isfinite(neural), axis=1)]
    if neural.shape[0] == 0:
        raise ValueError(f"{receipt['identifier']}: interpolation left no finite neurons")

    for vector in (velocity, angular_velocity):
        finite = np.isfinite(vector)
        if not finite.all():
            index = np.arange(vector.size)
            vector[~finite] = np.interp(index[~finite], index[finite], vector[finite])
    summaries = neural_summaries(neural)
    frames = min(len(velocity), len(angular_velocity), summaries.shape[0])
    return Recording(
        role=role,
        identifier=receipt["identifier"],
        velocity=velocity[:frames],
        angular_velocity=angular_velocity[:frames],
        neural_summary=summaries[:frames],
        frame_count=frames,
        neuron_count=neural.shape[0],
        cut_volume=receipt["cut_volume"],
        source_folder=receipt["folder"],
    )


def make_rows(recording: Recording, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_lag = max(LAGS)
    anchors = np.arange(max_lag, recording.frame_count - horizon, STRIDE)
    behavior_columns = []
    neural_columns = []
    for lag in LAGS:
        indices = anchors - lag
        behavior_columns.extend([
            recording.velocity[indices],
            recording.angular_velocity[indices],
        ])
        neural_columns.extend([
            recording.neural_summary[indices, feature_index]
            for feature_index in range(recording.neural_summary.shape[1])
        ])
    behavior = np.column_stack(behavior_columns)
    neural_plus_behavior = np.column_stack(behavior_columns + neural_columns)
    target = recording.velocity[anchors + horizon]
    return behavior, neural_plus_behavior, target


def fit_scaler(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def apply_scaler(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (matrix - mean) / scale


def fit_model(matrix: np.ndarray, target: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(matrix, target)
    return model


def evaluate_recording(
    recording: Recording,
    horizon: int,
    model_m1: Ridge,
    model_m2: Ridge,
    scaler_m1: tuple[np.ndarray, np.ndarray],
    scaler_m2: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    X1, X2, y = make_rows(recording, horizon)
    p0 = X1[:, 0]
    p1 = model_m1.predict(apply_scaler(X1, *scaler_m1))
    p2 = model_m2.predict(apply_scaler(X2, *scaler_m2))
    errors = {}
    for name, prediction in (("M0_persistence", p0), ("M1_behavior_only", p1), ("M2_neural_plus_behavior", p2)):
        rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
        mae = float(np.mean(np.abs(prediction - y)))
        errors[name] = {"rmse": rmse, "mae": mae}
    target_scale = float(np.std(y))
    if target_scale < 1e-8:
        target_scale = 1.0
    return {
        "recording": recording.identifier,
        "anchors": int(len(y)),
        "target_standard_deviation": target_scale,
        "metrics": errors,
        "nrmse": {name: metrics["rmse"] / target_scale for name, metrics in errors.items()},
        "ratios": {
            "M2_to_M0": errors["M2_neural_plus_behavior"]["rmse"] / errors["M0_persistence"]["rmse"],
            "M2_to_M1": errors["M2_neural_plus_behavior"]["rmse"] / errors["M1_behavior_only"]["rmse"],
            "M1_to_M0": errors["M1_behavior_only"]["rmse"] / errors["M0_persistence"]["rmse"],
        },
        "all_predictions_finite": bool(np.isfinite(p0).all() and np.isfinite(p1).all() and np.isfinite(p2).all()),
    }


def select_alpha(
    train_recordings: list[Recording],
    dev_recordings: list[Recording],
    model_kind: str,
) -> tuple[float, dict[str, Any]]:
    training_by_horizon = {}
    for horizon in HORIZONS:
        X_blocks, y_blocks = [], []
        for recording in train_recordings:
            X1, X2, y = make_rows(recording, horizon)
            X_blocks.append(X1 if model_kind == "M1" else X2)
            y_blocks.append(y)
        matrix = np.vstack(X_blocks)
        target = np.concatenate(y_blocks)
        mean, scale = fit_scaler(matrix)
        training_by_horizon[horizon] = (apply_scaler(matrix, mean, scale), target, mean, scale)

    candidates = []
    for alpha in ALPHAS:
        horizon_results = {}
        selection_values = []
        for horizon in HORIZONS:
            matrix, target, mean, scale = training_by_horizon[horizon]
            model = fit_model(matrix, target, alpha)
            per_recording = []
            for recording in dev_recordings:
                X1, X2, y = make_rows(recording, horizon)
                X = X1 if model_kind == "M1" else X2
                prediction = model.predict(apply_scaler(X, mean, scale))
                target_std = float(np.std(y)) or 1.0
                nrmse = float(np.sqrt(np.mean((prediction - y) ** 2)) / target_std)
                per_recording.append({"recording": recording.identifier, "nrmse": nrmse})
            balanced = float(np.mean([row["nrmse"] for row in per_recording]))
            selection_values.append(balanced)
            horizon_results[str(horizon)] = {"recording_balanced_nrmse": balanced, "per_recording": per_recording}
        selection_score = float(np.mean(selection_values))
        candidates.append({"alpha": alpha, "selection_score": selection_score, "horizons": horizon_results})
    candidates.sort(key=lambda row: (row["selection_score"], -row["alpha"]))
    return float(candidates[0]["alpha"]), {"candidates": candidates, "selected": candidates[0]}


def train_final_by_horizon(
    train_recordings: list[Recording],
    alpha_m1: float,
    alpha_m2: float,
) -> dict[int, dict[str, Any]]:
    models = {}
    for horizon in HORIZONS:
        X1_blocks, X2_blocks, y_blocks = [], [], []
        for recording in train_recordings:
            X1, X2, y = make_rows(recording, horizon)
            X1_blocks.append(X1)
            X2_blocks.append(X2)
            y_blocks.append(y)
        X1 = np.vstack(X1_blocks)
        X2 = np.vstack(X2_blocks)
        y = np.concatenate(y_blocks)
        scaler1 = fit_scaler(X1)
        scaler2 = fit_scaler(X2)
        model1 = fit_model(apply_scaler(X1, *scaler1), y, alpha_m1)
        model2 = fit_model(apply_scaler(X2, *scaler2), y, alpha_m2)
        models[horizon] = {
            "M1": model1,
            "M2": model2,
            "scaler_M1": scaler1,
            "scaler_M2": scaler2,
            "training_samples": int(len(y)),
        }
    return models


def serialize_models(models: dict[int, dict[str, Any]], alpha_m1: float, alpha_m2: float) -> dict[str, Any]:
    payload = {
        "schema": "bio-001-s26-osf-neural-velocity-development-model-v1",
        "preregistration_commit": PREREG_COMMIT,
        "source_code_commit": SOURCE_CODE_COMMIT,
        "lags_frames": LAGS,
        "horizons_frames": HORIZONS,
        "anchor_stride_frames": STRIDE,
        "neural_summary_names": NEURAL_NAMES,
        "alpha_M1": alpha_m1,
        "alpha_M2": alpha_m2,
        "models": {},
        "holdout_requested": False,
        "claim_boundary": "Development-selected model only. The AML32_chip archive was not requested or opened.",
    }
    for horizon, bundle in models.items():
        payload["models"][str(horizon)] = {
            "training_samples": bundle["training_samples"],
            "M1": {
                "coef": np.asarray(bundle["M1"].coef_).tolist(),
                "intercept": float(bundle["M1"].intercept_),
                "feature_mean": bundle["scaler_M1"][0].tolist(),
                "feature_scale": bundle["scaler_M1"][1].tolist(),
            },
            "M2": {
                "coef": np.asarray(bundle["M2"].coef_).tolist(),
                "intercept": float(bundle["M2"].intercept_),
                "feature_mean": bundle["scaler_M2"][0].tolist(),
                "feature_scale": bundle["scaler_M2"][1].tolist(),
            },
        }
    return payload


def main() -> None:
    source_receipts = []
    extracted_receipts = {}
    all_recordings: dict[str, list[Recording]] = {}
    for role, spec in ARCHIVES.items():
        archive, receipt = download(role, spec)
        source_receipts.append(receipt)
        recording_receipts, extraction_receipt = extract_selected(role, archive)
        extracted_receipts[role] = extraction_receipt
        base = EXTRACTED / role
        recordings = [load_recording(role, base, record) for record in recording_receipts]
        all_recordings[role] = recordings

    if any(SEALED_HOLDOUT["name"] in str(path) for path in ROOT.rglob("*")):
        # The constant name in this source file is allowed; filesystem content is not.
        leaked_files = [str(path) for path in ROOT.rglob("*") if path.is_file() and path.name == SEALED_HOLDOUT["name"]]
        if leaked_files:
            raise RuntimeError(f"sealed holdout file leak: {leaked_files}")

    train = all_recordings["training"]
    dev = all_recordings["development"]
    if not train or not dev:
        raise RuntimeError("empty train or development recording set")

    alpha_m1, selection_m1 = select_alpha(train, dev, "M1")
    alpha_m2, selection_m2 = select_alpha(train, dev, "M2")
    models = train_final_by_horizon(train, alpha_m1, alpha_m2)

    per_horizon = {}
    for horizon in HORIZONS:
        bundle = models[horizon]
        records = [
            evaluate_recording(
                recording,
                horizon,
                bundle["M1"],
                bundle["M2"],
                bundle["scaler_M1"],
                bundle["scaler_M2"],
            )
            for recording in dev
        ]
        per_horizon[str(horizon)] = {
            "per_recording": records,
            "recording_balanced_M2_to_M0": float(np.mean([row["ratios"]["M2_to_M0"] for row in records])),
            "recording_balanced_M2_to_M1": float(np.mean([row["ratios"]["M2_to_M1"] for row in records])),
            "recording_balanced_M1_to_M0": float(np.mean([row["ratios"]["M1_to_M0"] for row in records])),
            "fraction_recordings_M2_beats_M1": float(np.mean([row["ratios"]["M2_to_M1"] < 1.0 for row in records])),
            "all_predictions_finite": bool(all(row["all_predictions_finite"] for row in records)),
        }

    primary = per_horizon[str(PRIMARY_HORIZON)]
    horizons_beating_both = sum(
        metrics["recording_balanced_M2_to_M0"] < 1.0
        and metrics["recording_balanced_M2_to_M1"] < 1.0
        for metrics in per_horizon.values()
    )
    gate = {
        "all_predictions_finite": bool(all(row["all_predictions_finite"] for row in per_horizon.values())),
        "primary_M2_to_M0": primary["recording_balanced_M2_to_M0"],
        "primary_M2_to_M1": primary["recording_balanced_M2_to_M1"],
        "primary_fraction_recordings_M2_beats_M1": primary["fraction_recordings_M2_beats_M1"],
        "horizons_where_M2_beats_both": int(horizons_beating_both),
    }
    gate["pass"] = bool(
        gate["all_predictions_finite"]
        and gate["primary_M2_to_M0"] <= 0.95
        and gate["primary_M2_to_M1"] <= 0.98
        and gate["horizons_where_M2_beats_both"] >= 2
        and gate["primary_fraction_recordings_M2_beats_M1"] >= 0.60
    )

    recording_manifest = []
    arrays = {}
    for role, recordings in all_recordings.items():
        for recording in recordings:
            key = f"{role}__{recording.identifier}"
            arrays[f"{key}__velocity"] = recording.velocity
            arrays[f"{key}__angular_velocity"] = recording.angular_velocity
            arrays[f"{key}__neural_summary"] = recording.neural_summary
            recording_manifest.append({
                "role": role,
                "identifier": recording.identifier,
                "frames": recording.frame_count,
                "neurons": recording.neuron_count,
                "cut_volume": recording.cut_volume,
                "source_folder": recording.source_folder,
            })
    arrays_path = OUT / "extracted_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)

    dump("source_archive_receipts.json", {
        "schema": "bio-001-s26-osf-source-archive-receipts-v1",
        "archives": source_receipts,
        "sealed_holdout": {
            "filename": SEALED_HOLDOUT["name"],
            "requested": False,
            "downloaded": False,
            "opened": False,
        },
    })
    dump("recording_manifest.json", {
        "schema": "bio-001-s26-osf-recording-manifest-v1",
        "extraction": extracted_receipts,
        "recordings": recording_manifest,
        "extracted_arrays_file": arrays_path.name,
        "extracted_arrays_sha256": sha256(arrays_path),
    })
    model_payload = serialize_models(models, alpha_m1, alpha_m2)
    dump("development_model.json", model_payload)
    dump("development_results.json", {
        "schema": "bio-001-s26-osf-development-results-v1",
        "preregistration_commit": PREREG_COMMIT,
        "source_code_commit": SOURCE_CODE_COMMIT,
        "training_recording_count": len(train),
        "development_recording_count": len(dev),
        "selection_M1": selection_m1,
        "selection_M2": selection_m2,
        "selected_alpha_M1": alpha_m1,
        "selected_alpha_M2": alpha_m2,
        "metrics": per_horizon,
        "development_gate": gate,
        "holdout_requested": False,
        "CeRSI_v2_delta": 0.0,
        "status": "PASS_DEVELOPMENT_GATE" if gate["pass"] else "FAIL_DEVELOPMENT_GATE",
    })
    dump("run_receipt.json", {
        "schema": "bio-001-s26-osf-train-dev-run-receipt-v1",
        "status": "PASS_DEVELOPMENT_GATE" if gate["pass"] else "FAIL_DEVELOPMENT_GATE",
        "source_archive_receipts_sha256": sha256(OUT / "source_archive_receipts.json"),
        "recording_manifest_sha256": sha256(OUT / "recording_manifest.json"),
        "extracted_arrays_sha256": sha256(arrays_path),
        "development_model_sha256": sha256(OUT / "development_model.json"),
        "development_results_sha256": sha256(OUT / "development_results.json"),
        "selected_alpha_M1": alpha_m1,
        "selected_alpha_M2": alpha_m2,
        "development_gate": gate,
        "holdout_requested": False,
        "CeRSI_v2_delta": 0.0,
    })

    # Raw archives are verified and represented by exact OSF hashes, but are not
    # uploaded to the GitHub artifact because they total ~835 MB. The immutable
    # extracted arrays and full model/results are retained.
    shutil.rmtree(RAW)
    shutil.rmtree(EXTRACTED)
    print(json.dumps({
        "status": "PASS_DEVELOPMENT_GATE" if gate["pass"] else "FAIL_DEVELOPMENT_GATE",
        "training_recordings": len(train),
        "development_recordings": len(dev),
        "selected_alpha_M1": alpha_m1,
        "selected_alpha_M2": alpha_m2,
        "development_gate": gate,
        "holdout_requested": False,
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        dump("failure_receipt.json", {
            "schema": "bio-001-s26-osf-train-dev-failure-v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "preregistration_commit": PREREG_COMMIT,
            "holdout_requested": False,
            "CeRSI_v2_delta": 0.0,
        })
        raise
