from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

HORIZONS = (1, 6, 12, 30)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
STRIDE = 6
PREFIX_FRACTION = 0.20
MIN_PREFIX = 200


@dataclass
class Recording:
    identifier: str
    path: Path
    behavior_x: np.ndarray
    coupled_x: np.ndarray
    targets: dict[int, np.ndarray]
    persistence: dict[int, np.ndarray]
    anchors: np.ndarray
    target_scale: dict[int, float]
    feature_names_behavior: list[str]
    feature_names_coupled: list[str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interpolate_short_gaps(values: np.ndarray, max_gap: int = 3) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    if out.ndim == 1:
        out = out[None, :]
        squeeze = True
    else:
        squeeze = False
    for row in range(out.shape[0]):
        x = out[row]
        finite = np.isfinite(x)
        index = 0
        while index < x.size:
            if finite[index]:
                index += 1
                continue
            start = index
            while index < x.size and not finite[index]:
                index += 1
            end = index
            length = end - start
            if length <= max_gap and start > 0 and end < x.size and finite[start - 1] and finite[end]:
                x[start:end] = np.linspace(x[start - 1], x[end], length + 2)[1:-1]
                finite[start:end] = True
    return out[0] if squeeze else out


def orient_time_vector(value: object, expected: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float).squeeze()
    if array.ndim != 1:
        raise ValueError(f"expected one-dimensional behavior field, got {array.shape}")
    if expected is not None and array.size != expected:
        raise ValueError(f"behavior length {array.size} does not equal expected {expected}")
    return array


def orient_components(value: object, expected_time: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).squeeze()
    if array.ndim == 1:
        if array.size != expected_time:
            raise ValueError(f"component vector length {array.size} != {expected_time}")
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"unsupported component array shape {array.shape}")
    if array.shape[0] == expected_time and array.shape[1] <= 8:
        return array
    if array.shape[1] == expected_time and array.shape[0] <= 8:
        return array.T
    raise ValueError(f"cannot orient component array {array.shape} to time {expected_time}")


def population_features(neural_column: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.mean(neural_column),
            np.std(neural_column),
            np.quantile(neural_column, 0.10),
            np.quantile(neural_column, 0.25),
            np.quantile(neural_column, 0.50),
            np.quantile(neural_column, 0.75),
            np.quantile(neural_column, 0.90),
            np.mean(neural_column > 0.0),
            np.mean(neural_column < 0.0),
        ],
        dtype=float,
    )


def load_recording(path: Path, identifier: str) -> Recording:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "Ratio2" not in payload or "behavior" not in payload:
        raise ValueError(f"{path}: required Ratio2/behavior fields missing")
    behavior = payload["behavior"]
    velocity = orient_time_vector(getattr(behavior, "v"))
    time_length = velocity.size

    ratio = np.asarray(payload["Ratio2"], dtype=float)
    if ratio.ndim != 2:
        raise ValueError(f"{path}: Ratio2 is not 2D: {ratio.shape}")
    if ratio.shape[1] == time_length:
        neural = ratio
    elif ratio.shape[0] == time_length:
        neural = ratio.T
    else:
        raise ValueError(f"{path}: Ratio2 {ratio.shape} does not align with behavior {time_length}")

    pc12 = orient_components(getattr(behavior, "pc1_2"), time_length)
    pc3 = orient_components(getattr(behavior, "pc_3"), time_length)
    common = min(time_length, neural.shape[1], pc12.shape[0], pc3.shape[0])
    velocity = velocity[:common]
    neural = neural[:, :common]
    pc12 = pc12[:common]
    pc3 = pc3[:common]

    velocity = interpolate_short_gaps(velocity)
    neural = interpolate_short_gaps(neural)
    for column in range(pc12.shape[1]):
        pc12[:, column] = interpolate_short_gaps(pc12[:, column])
    for column in range(pc3.shape[1]):
        pc3[:, column] = interpolate_short_gaps(pc3[:, column])

    prefix_end = max(MIN_PREFIX, int(math.floor(PREFIX_FRACTION * common)))
    if prefix_end + max(HORIZONS) + 1 >= common:
        raise ValueError(f"{path}: recording too short after prefix")
    prefix = neural[:, :prefix_end]
    if not np.all(np.isfinite(prefix)):
        raise ValueError(f"{path}: nonfinite neural value in PCA prefix")

    median = np.median(prefix, axis=1)
    mad = np.median(np.abs(prefix - median[:, None]), axis=1) * 1.4826
    prefix_std = np.std(prefix, axis=1)
    bad = (~np.isfinite(mad)) | (mad <= 1e-12)
    mad[bad] = prefix_std[bad]
    bad = (~np.isfinite(mad)) | (mad <= 1e-12)
    mad[bad] = 1.0
    normalized = np.clip((neural - median[:, None]) / mad[:, None], -10.0, 10.0)
    if not np.all(np.isfinite(normalized[:, :prefix_end])):
        raise ValueError(f"{path}: normalized prefix is nonfinite")

    prefix_frames = normalized[:, :prefix_end].T
    pca_mean = np.mean(prefix_frames, axis=0)
    centered = prefix_frames - pca_mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    if vt.shape[0] < 5 or np.count_nonzero(singular_values > 1e-12) < 5:
        raise ValueError(f"{path}: fewer than five nondegenerate PCA components")
    components = vt[:5].copy()
    prefix_scores = centered @ components.T
    prefix_population_mean = np.mean(prefix_frames, axis=1)
    for component in range(5):
        correlation = np.corrcoef(prefix_scores[:, component], prefix_population_mean)[0, 1]
        if np.isfinite(correlation) and correlation < 0:
            components[component] *= -1.0
    pca_scores = (normalized.T - pca_mean) @ components.T

    pop = np.full((common, 9), np.nan, dtype=float)
    valid_neural_frame = np.all(np.isfinite(normalized), axis=0)
    for frame in np.flatnonzero(valid_neural_frame):
        pop[frame] = population_features(normalized[:, frame])

    primary_curvature = pc12[:, 0]
    secondary_curvature = pc12[:, 1] if pc12.shape[1] >= 2 else np.zeros(common)
    pc3_primary = pc3[:, 0]

    behavior_feature_names = [
        "velocity",
        "delta_velocity",
        "absolute_velocity",
        "primary_curvature",
        "delta_primary_curvature",
        "secondary_pc12",
        "pc3",
        "velocity_x_primary_curvature",
    ]
    pop_names = ["neural_mean", "neural_std", "neural_q10", "neural_q25", "neural_q50", "neural_q75", "neural_q90", "neural_positive_fraction", "neural_negative_fraction"]
    pca_names = [f"neural_pc{i}" for i in range(1, 6)]
    coupled_names = (
        behavior_feature_names
        + pop_names
        + [f"delta_{name}" for name in pop_names]
        + pca_names
        + [f"delta_{name}" for name in pca_names]
        + [f"velocity_x_{name}" for name in pop_names + pca_names]
    )

    behavior_rows: list[np.ndarray] = []
    coupled_rows: list[np.ndarray] = []
    anchor_rows: list[int] = []
    targets = {horizon: [] for horizon in HORIZONS}
    persistence = {horizon: [] for horizon in HORIZONS}

    start = prefix_end + 1
    stop = common - max(HORIZONS)
    for anchor in range(start, stop, STRIDE):
        behavior_row = np.array(
            [
                velocity[anchor],
                velocity[anchor] - velocity[anchor - 1],
                abs(velocity[anchor]),
                primary_curvature[anchor],
                primary_curvature[anchor] - primary_curvature[anchor - 1],
                secondary_curvature[anchor],
                pc3_primary[anchor],
                velocity[anchor] * primary_curvature[anchor],
            ],
            dtype=float,
        )
        current_pop = pop[anchor]
        previous_pop = pop[anchor - 1]
        current_pca = pca_scores[anchor]
        previous_pca = pca_scores[anchor - 1]
        coupled_row = np.concatenate(
            [
                behavior_row,
                current_pop,
                current_pop - previous_pop,
                current_pca,
                current_pca - previous_pca,
                velocity[anchor] * np.concatenate([current_pop, current_pca]),
            ]
        )
        future = np.array([velocity[anchor + horizon] for horizon in HORIZONS])
        if not (np.all(np.isfinite(behavior_row)) and np.all(np.isfinite(coupled_row)) and np.all(np.isfinite(future))):
            continue
        behavior_rows.append(behavior_row)
        coupled_rows.append(coupled_row)
        anchor_rows.append(anchor)
        for horizon, value in zip(HORIZONS, future):
            targets[horizon].append(float(value))
            persistence[horizon].append(float(velocity[anchor]))

    if len(anchor_rows) < 200:
        raise ValueError(f"{path}: only {len(anchor_rows)} valid anchors")
    behavior_x = np.vstack(behavior_rows)
    coupled_x = np.vstack(coupled_rows)
    target_arrays = {horizon: np.asarray(values, dtype=float) for horizon, values in targets.items()}
    persistence_arrays = {horizon: np.asarray(values, dtype=float) for horizon, values in persistence.items()}
    target_scale = {}
    for horizon, values in target_arrays.items():
        scale = float(np.std(values))
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError(f"{path}: degenerate target scale at horizon {horizon}")
        target_scale[horizon] = scale
    return Recording(
        identifier=identifier,
        path=path,
        behavior_x=behavior_x,
        coupled_x=coupled_x,
        targets=target_arrays,
        persistence=persistence_arrays,
        anchors=np.asarray(anchor_rows, dtype=int),
        target_scale=target_scale,
        feature_names_behavior=behavior_feature_names,
        feature_names_coupled=coupled_names,
    )


def weighted_standardizer(matrices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    weights = []
    for matrix in matrices:
        weights.append(np.full(matrix.shape[0], 1.0 / matrix.shape[0]))
    total = sum(float(np.sum(weight)) for weight in weights)
    weights = [weight * (len(matrices) / total) for weight in weights]
    x = np.vstack(matrices)
    w = np.concatenate(weights)
    mean = np.sum(x * w[:, None], axis=0) / np.sum(w)
    variance = np.sum(((x - mean) ** 2) * w[:, None], axis=0) / np.sum(w)
    scale = np.sqrt(np.maximum(variance, 1e-24))
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    return mean, scale, weights


def fit_ridge(recordings: list[Recording], feature_set: str, alpha: float) -> dict:
    matrices = [record.behavior_x if feature_set == "behavior" else record.coupled_x for record in recordings]
    mean, scale, weight_vectors = weighted_standardizer(matrices)
    x = np.vstack([(matrix - mean) / scale for matrix in matrices])
    weights = np.concatenate(weight_vectors)
    design = np.column_stack([np.ones(x.shape[0]), x])
    sqrt_w = np.sqrt(weights)
    weighted_design = design * sqrt_w[:, None]
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = {}
    for horizon in HORIZONS:
        y = np.concatenate([record.targets[horizon] for record in recordings])
        weighted_y = y * sqrt_w
        lhs = weighted_design.T @ weighted_design + penalty
        rhs = weighted_design.T @ weighted_y
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(lhs) @ rhs
        coefficients[str(horizon)] = beta
    return {"feature_set": feature_set, "alpha": alpha, "mean": mean, "scale": scale, "coefficients": coefficients}


def predict(model: dict, record: Recording, horizon: int) -> np.ndarray:
    x = record.behavior_x if model["feature_set"] == "behavior" else record.coupled_x
    standardized = (x - model["mean"]) / model["scale"]
    design = np.column_stack([np.ones(standardized.shape[0]), standardized])
    return design @ model["coefficients"][str(horizon)]


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def evaluate(model: dict, recordings: list[Recording]) -> dict:
    per_recording = {}
    aggregate_losses = []
    for record in recordings:
        horizon_metrics = {}
        for horizon in HORIZONS:
            target = record.targets[horizon]
            model_prediction = predict(model, record, horizon)
            persistence_prediction = record.persistence[horizon]
            scale = record.target_scale[horizon]
            model_nrmse = rmse(model_prediction, target) / scale
            persistence_nrmse = rmse(persistence_prediction, target) / scale
            horizon_metrics[str(horizon)] = {
                "n": int(target.size),
                "target_std": scale,
                "model_rmse": rmse(model_prediction, target),
                "model_nrmse": model_nrmse,
                "persistence_rmse": rmse(persistence_prediction, target),
                "persistence_nrmse": persistence_nrmse,
                "model_over_persistence": model_nrmse / persistence_nrmse,
            }
            aggregate_losses.append(model_nrmse)
        per_recording[record.identifier] = horizon_metrics
    return {"mean_nrmse": float(np.mean(aggregate_losses)), "per_recording": per_recording}


def serializable_model(model: dict, feature_names: list[str], source_recordings: list[str]) -> dict:
    return {
        "feature_set": model["feature_set"],
        "alpha": model["alpha"],
        "feature_names": feature_names,
        "mean": model["mean"].tolist(),
        "scale": model["scale"].tolist(),
        "coefficients": {key: value.tolist() for key, value in model["coefficients"].items()},
        "horizons": list(HORIZONS),
        "stride": STRIDE,
        "prefix_fraction": PREFIX_FRACTION,
        "minimum_prefix": MIN_PREFIX,
        "source_recordings": source_recordings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", required=True)
    parser.add_argument("--split-receipt", required=True)
    parser.add_argument("--archive-receipts", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    extracted_root = Path(args.extracted_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split_receipt).read_text(encoding="utf-8"))
    archive_receipts = json.loads(Path(args.archive_receipts).read_text(encoding="utf-8"))

    role_by_identifier: dict[str, str] = {}
    condition_by_identifier: dict[str, str] = {}
    for condition, receipt in split.items():
        for role in ("train", "dev", "holdout"):
            for identifier in receipt["split"][role]:
                role_by_identifier[identifier] = role
                condition_by_identifier[identifier] = condition

    mat_paths = sorted(extracted_root.rglob("heatDataMS.mat"))
    recordings: list[Recording] = []
    recording_receipts = []
    for path in mat_paths:
        matches = [identifier for identifier in role_by_identifier if identifier in str(path)]
        if len(matches) != 1:
            raise RuntimeError(f"cannot assign recording path {path}; matches={matches}")
        identifier = matches[0]
        if role_by_identifier[identifier] == "holdout":
            raise RuntimeError(f"holdout recording was extracted before model freeze: {identifier}")
        record = load_recording(path, identifier)
        recordings.append(record)
        recording_receipts.append(
            {
                "identifier": identifier,
                "condition": condition_by_identifier[identifier],
                "role": role_by_identifier[identifier],
                "path": str(path.relative_to(extracted_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "anchors": int(record.anchors.size),
                "behavior_features": int(record.behavior_x.shape[1]),
                "coupled_features": int(record.coupled_x.shape[1]),
            }
        )

    train = [record for record in recordings if role_by_identifier[record.identifier] == "train"]
    dev = [record for record in recordings if role_by_identifier[record.identifier] == "dev"]
    if len(train) != 6 or len(dev) != 2:
        raise RuntimeError(f"expected 6 train and 2 development recordings; got {len(train)}, {len(dev)}")

    selection = {}
    selected_models = {}
    for feature_set in ("behavior", "coupled"):
        candidates = []
        for alpha in ALPHAS:
            model = fit_ridge(train, feature_set, alpha)
            evaluation = evaluate(model, dev)
            candidates.append({"alpha": alpha, "mean_nrmse": evaluation["mean_nrmse"], "evaluation": evaluation})
        candidates.sort(key=lambda item: (round(item["mean_nrmse"] / 1e-12) * 1e-12, item["alpha"]))
        best = min(candidates, key=lambda item: (item["mean_nrmse"], item["alpha"]))
        selected_models[feature_set] = fit_ridge(train + dev, feature_set, best["alpha"])
        selection[feature_set] = {"selected_alpha": best["alpha"], "selected_mean_nrmse": best["mean_nrmse"], "candidates": candidates}

    behavior_names = train[0].feature_names_behavior
    coupled_names = train[0].feature_names_coupled
    if any(record.feature_names_behavior != behavior_names or record.feature_names_coupled != coupled_names for record in recordings):
        raise RuntimeError("feature name mismatch across recordings")

    model_artifact = {
        "schema": "bio-001-s26-hallinen-frozen-model-v1",
        "preregistration_commit": "2200444ea0e43069597da277ed7a6378b7bdea67",
        "feature_amendment_commit": "596318171b14b3998b8fceb46384399e799952c4",
        "archive_receipts": archive_receipts,
        "recording_receipts": recording_receipts,
        "holdout_identifiers": [identifier for identifier, role in role_by_identifier.items() if role == "holdout"],
        "behavior_model": serializable_model(selected_models["behavior"], behavior_names, [record.identifier for record in train + dev]),
        "coupled_model": serializable_model(selected_models["coupled"], coupled_names, [record.identifier for record in train + dev]),
        "holdout_status": "UNEXTRACTED_AND_UNOPENED",
        "CeRSI_v2_delta": 0.0,
    }
    model_path = output_dir / "HALLINEN_S26_FROZEN_MODEL.json"
    model_path.write_text(json.dumps(model_artifact, indent=2), encoding="utf-8")
    development_path = output_dir / "HALLINEN_S26_DEVELOPMENT_SELECTION.json"
    development_path.write_text(json.dumps({"selection": selection, "recording_receipts": recording_receipts}, indent=2), encoding="utf-8")
    receipt = {
        "model_filename": model_path.name,
        "model_bytes": model_path.stat().st_size,
        "model_sha256": sha256(model_path),
        "development_filename": development_path.name,
        "development_bytes": development_path.stat().st_size,
        "development_sha256": sha256(development_path),
        "selected_behavior_alpha": selection["behavior"]["selected_alpha"],
        "selected_coupled_alpha": selection["coupled"]["selected_alpha"],
        "development_behavior_nrmse": selection["behavior"]["selected_mean_nrmse"],
        "development_coupled_nrmse": selection["coupled"]["selected_mean_nrmse"],
        "development_coupled_over_behavior": selection["coupled"]["selected_mean_nrmse"] / selection["behavior"]["selected_mean_nrmse"],
        "train_recordings": [record.identifier for record in train],
        "development_recordings": [record.identifier for record in dev],
        "holdout_identifiers": model_artifact["holdout_identifiers"],
        "holdout_opened": False,
        "CeRSI_v2_delta": 0.0,
    }
    (output_dir / "HALLINEN_S26_MODEL_FREEZE_RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
