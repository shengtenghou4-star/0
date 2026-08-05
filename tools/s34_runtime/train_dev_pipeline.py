from __future__ import annotations

"""Frozen train/development pipeline for BIO-001 S27/S28.

The module contains no holdout URL and refuses paths containing the sealed
archive name. It consumes only extracted AML310 training/development roots.
All evidence JSON is strict: NaN and Infinity are rejected.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import Ridge

from official_loader_port import load_recording_folder, parse_dataset_list_text

LAGS = (0, 1, 2, 3, 5, 8, 13)
HORIZONS = (1, 5, 10)
PRIMARY_HORIZON = 10
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SEALED_TOKEN = "AML32_chip"
RATIO_DENOMINATOR_FLOOR = 1e-12
NEURAL_SUMMARY_NAMES = (
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
)
BEHAVIOR_NAMES = ("CMSVelocity", "Curvature")


@dataclass(frozen=True)
class PreparedRecording:
    recording_id: str
    role: str
    absolute_frames: np.ndarray
    velocity: np.ndarray
    curvature: np.ndarray
    neural_summary: np.ndarray
    neurons: int


@dataclass(frozen=True)
class SampleBlock:
    recording_id: str
    horizon: int
    X_behavior: np.ndarray
    X_neural_behavior: np.ndarray
    y: np.ndarray
    persistence: np.ndarray
    current_absolute_frames: np.ndarray
    target_absolute_frames: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def assert_not_holdout_path(path: Path) -> None:
    resolved = str(path.resolve())
    if SEALED_TOKEN.lower() in resolved.lower():
        raise RuntimeError(f"sealed holdout path rejected: {resolved}")


def _zscore_neurons_within_recording(neural: np.ndarray) -> np.ndarray:
    values = np.asarray(neural, dtype=float)
    if values.ndim != 2:
        raise ValueError("neural array must be neurons x valid frames")
    mean = np.mean(values, axis=1, keepdims=True)
    scale = np.std(values, axis=1, keepdims=True)
    scale[scale < RATIO_DENOMINATOR_FLOOR] = 1.0
    z = (values - mean) / scale
    if not np.all(np.isfinite(z)):
        raise ValueError("within-recording neural z-score is nonfinite")
    return z


def neural_summaries(neural: np.ndarray, absolute_frames: np.ndarray) -> np.ndarray:
    """Build frozen neural summaries without bridging nonconsecutive frames."""
    z = _zscore_neurons_within_recording(neural)
    frames = np.asarray(absolute_frames, dtype=int)
    if frames.ndim != 1 or frames.size != z.shape[1]:
        raise ValueError("absolute frame vector does not align to neural array")
    if np.any(np.diff(frames) <= 0):
        raise ValueError("absolute frames must be strictly increasing")
    summaries = np.column_stack([
        np.mean(z, axis=0),
        np.std(z, axis=0),
        np.quantile(z, 0.10, axis=0),
        np.quantile(z, 0.25, axis=0),
        np.median(z, axis=0),
        np.quantile(z, 0.75, axis=0),
        np.quantile(z, 0.90, axis=0),
        np.mean(z > 1.0, axis=0),
        np.mean(z < -1.0, axis=0),
        np.full(frames.size, np.nan),
        np.full(frames.size, np.nan),
    ])
    for index in range(1, frames.size):
        if frames[index] - frames[index - 1] != 1:
            continue
        delta = z[:, index] - z[:, index - 1]
        summaries[index, 9] = np.mean(np.maximum(delta, 0.0))
        summaries[index, 10] = np.mean(np.minimum(delta, 0.0))
    return summaries


def prepare_recording(
    recording_id: str, role: str, loaded: dict[str, Any]
) -> PreparedRecording:
    frames = np.asarray(loaded["Neurons"]["I_valid_map"], dtype=int)
    neural = np.asarray(
        loaded["Neurons"]["I_smooth_interp_crop_noncontig"], dtype=float
    )
    velocity = np.asarray(
        loaded["Behavior_crop_noncontig"]["CMSVelocity"], dtype=float
    )
    curvature = np.asarray(
        loaded["Behavior_crop_noncontig"]["Curvature"], dtype=float
    )
    if not (frames.size == neural.shape[1] == velocity.size == curvature.size):
        raise ValueError(f"recording arrays do not align for {recording_id}")
    return PreparedRecording(
        recording_id=recording_id,
        role=role,
        absolute_frames=frames,
        velocity=velocity,
        curvature=curvature,
        neural_summary=neural_summaries(neural, frames),
        neurons=int(neural.shape[0]),
    )


def build_samples(recording: PreparedRecording, horizon: int) -> SampleBlock:
    frame_to_row = {
        int(frame): row for row, frame in enumerate(recording.absolute_frames)
    }
    x_behavior: list[np.ndarray] = []
    x_full: list[np.ndarray] = []
    targets: list[float] = []
    persistence: list[float] = []
    current_frames: list[int] = []
    target_frames: list[int] = []
    for current_frame in recording.absolute_frames:
        current = int(current_frame)
        target = current + int(horizon)
        required = [current - lag for lag in LAGS]
        if target not in frame_to_row or any(
            frame not in frame_to_row for frame in required
        ):
            continue
        rows = [frame_to_row[frame] for frame in required]
        target_row = frame_to_row[target]
        behavior_parts = [
            np.array([
                recording.velocity[row], recording.curvature[row]
            ], dtype=float)
            for row in rows
        ]
        neural_parts = [recording.neural_summary[row] for row in rows]
        behavior_vector = np.concatenate(behavior_parts)
        neural_vector = np.concatenate(neural_parts)
        full_vector = np.concatenate([behavior_vector, neural_vector])
        if not (
            np.all(np.isfinite(behavior_vector))
            and np.all(np.isfinite(full_vector))
            and np.isfinite(recording.velocity[target_row])
        ):
            continue
        x_behavior.append(behavior_vector)
        x_full.append(full_vector)
        targets.append(float(recording.velocity[target_row]))
        persistence.append(float(recording.velocity[frame_to_row[current]]))
        current_frames.append(current)
        target_frames.append(target)
    if not targets:
        raise ValueError(
            f"no valid samples for {recording.recording_id} horizon {horizon}"
        )
    return SampleBlock(
        recording_id=recording.recording_id,
        horizon=horizon,
        X_behavior=np.vstack(x_behavior),
        X_neural_behavior=np.vstack(x_full),
        y=np.asarray(targets, dtype=float),
        persistence=np.asarray(persistence, dtype=float),
        current_absolute_frames=np.asarray(current_frames, dtype=int),
        target_absolute_frames=np.asarray(target_frames, dtype=int),
    )


def build_all_samples(
    recordings: Iterable[PreparedRecording],
) -> dict[int, list[SampleBlock]]:
    result: dict[int, list[SampleBlock]] = {
        horizon: [] for horizon in HORIZONS
    }
    for recording in recordings:
        for horizon in HORIZONS:
            result[horizon].append(build_samples(recording, horizon))
    return result


def _fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale[scale < RATIO_DENOMINATOR_FLOOR] = 1.0
    return mean, scale


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    mean, scale = _fit_scaler(X)
    standardized = (X - mean) / scale
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(standardized, y)
    return {
        "alpha": float(alpha),
        "mean": mean,
        "scale": scale,
        "coef": np.asarray(model.coef_, dtype=float),
        "intercept": float(model.intercept_),
    }


def _predict(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    standardized = (X - model["mean"]) / model["scale"]
    return standardized @ model["coef"] + model["intercept"]


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(
        (np.asarray(observed) - np.asarray(predicted)) ** 2
    )))


def _recording_metric(
    block: SampleBlock, prediction: np.ndarray
) -> dict[str, Any]:
    model_rmse = _rmse(block.y, prediction)
    persistence_rmse = _rmse(block.y, block.persistence)
    target_scale = max(float(np.std(block.y)), RATIO_DENOMINATOR_FLOOR)
    return {
        "recording_id": block.recording_id,
        "samples": int(block.y.size),
        "rmse": model_rmse,
        "nrmse_by_recording_target_sd": model_rmse / target_scale,
        "persistence_rmse": persistence_rmse,
        "model_to_persistence_rmse_ratio": (
            model_rmse / max(persistence_rmse, RATIO_DENOMINATOR_FLOOR)
        ),
        "prediction_finite": bool(np.all(np.isfinite(prediction))),
    }


def _train_models_for_alpha(
    training: dict[int, list[SampleBlock]],
    development: dict[int, list[SampleBlock]],
    alpha: float,
    feature_key: str,
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    float,
]:
    models: dict[int, dict[str, Any]] = {}
    metrics: dict[int, list[dict[str, Any]]] = {}
    selection_values: list[float] = []
    for horizon in HORIZONS:
        train_blocks = training[horizon]
        X_train = np.vstack([
            getattr(block, feature_key) for block in train_blocks
        ])
        y_train = np.concatenate([block.y for block in train_blocks])
        model = _fit_ridge(X_train, y_train, alpha)
        models[horizon] = model
        horizon_metrics = []
        for block in development[horizon]:
            prediction = _predict(model, getattr(block, feature_key))
            metric = _recording_metric(block, prediction)
            horizon_metrics.append(metric)
            selection_values.append(
                metric["nrmse_by_recording_target_sd"]
            )
        metrics[horizon] = horizon_metrics
    score = float(np.mean(selection_values))
    if not np.isfinite(score):
        raise ValueError("nonfinite development selection score")
    return models, metrics, score


def select_model_family(
    training: dict[int, list[SampleBlock]],
    development: dict[int, list[SampleBlock]],
    feature_key: str,
) -> dict[str, Any]:
    candidates = []
    by_alpha: dict[
        float,
        tuple[
            dict[int, dict[str, Any]],
            dict[int, list[dict[str, Any]]],
        ],
    ] = {}
    for alpha in ALPHAS:
        models, metrics, score = _train_models_for_alpha(
            training, development, alpha, feature_key
        )
        candidates.append({
            "alpha": float(alpha),
            "selection_recording_balanced_mean_nrmse": score,
            "development_metrics": metrics,
        })
        by_alpha[float(alpha)] = (models, metrics)
    candidates.sort(key=lambda item: (
        item["selection_recording_balanced_mean_nrmse"],
        -item["alpha"],
    ))
    selected_alpha = float(candidates[0]["alpha"])
    models, metrics = by_alpha[selected_alpha]
    return {
        "selected_alpha": selected_alpha,
        "selection_rule": (
            "minimum unweighted mean per-recording NRMSE across horizons "
            "1,5,10; ties choose larger alpha"
        ),
        "candidates": candidates,
        "models": models,
        "development_metrics": metrics,
    }


def _metric_map(
    metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {item["recording_id"]: item for item in metrics}


def evaluate_gate(
    m1: dict[str, Any],
    m2: dict[str, Any],
    development: dict[int, list[SampleBlock]],
) -> dict[str, Any]:
    horizon_summary: dict[str, Any] = {}
    all_finite = True
    horizons_beating_both = 0
    for horizon in HORIZONS:
        m1_map = _metric_map(m1["development_metrics"][horizon])
        m2_map = _metric_map(m2["development_metrics"][horizon])
        ratios_to_persistence = []
        ratios_to_m1 = []
        for block in development[horizon]:
            identifier = block.recording_id
            m1_metric = m1_map[identifier]
            m2_metric = m2_map[identifier]
            all_finite = (
                all_finite
                and m1_metric["prediction_finite"]
                and m2_metric["prediction_finite"]
            )
            ratios_to_persistence.append(
                m2_metric["model_to_persistence_rmse_ratio"]
            )
            ratios_to_m1.append(
                m2_metric["rmse"]
                / max(m1_metric["rmse"], RATIO_DENOMINATOR_FLOOR)
            )
        mean_to_persistence = float(np.mean(ratios_to_persistence))
        mean_to_m1 = float(np.mean(ratios_to_m1))
        if not np.isfinite(mean_to_persistence) or not np.isfinite(mean_to_m1):
            raise ValueError("nonfinite development gate ratio")
        if mean_to_persistence < 1.0 and mean_to_m1 < 1.0:
            horizons_beating_both += 1
        horizon_summary[str(horizon)] = {
            "recording_balanced_mean_M2_to_persistence_RMSE_ratio": (
                mean_to_persistence
            ),
            "recording_balanced_mean_M2_to_M1_RMSE_ratio": mean_to_m1,
            "recording_ratios_to_persistence": ratios_to_persistence,
            "recording_ratios_to_M1": ratios_to_m1,
        }
    primary_m1 = _metric_map(
        m1["development_metrics"][PRIMARY_HORIZON]
    )
    primary_m2 = _metric_map(
        m2["development_metrics"][PRIMARY_HORIZON]
    )
    identifiers = sorted(primary_m2)
    fraction_beating_m1 = float(np.mean([
        primary_m2[identifier]["rmse"] < primary_m1[identifier]["rmse"]
        for identifier in identifiers
    ]))
    primary = horizon_summary[str(PRIMARY_HORIZON)]
    checks = {
        "all_predictions_finite": bool(all_finite),
        "primary_M2_to_persistence_ratio_at_most_0_95": (
            primary[
                "recording_balanced_mean_M2_to_persistence_RMSE_ratio"
            ] <= 0.95
        ),
        "primary_M2_to_M1_ratio_at_most_0_98": (
            primary["recording_balanced_mean_M2_to_M1_RMSE_ratio"]
            <= 0.98
        ),
        "at_least_two_horizons_M2_beats_both": (
            horizons_beating_both >= 2
        ),
        "at_least_60pct_recordings_M2_beats_M1_primary": (
            fraction_beating_m1 >= 0.60
        ),
    }
    return {
        "horizons": horizon_summary,
        "horizons_where_M2_beats_both": horizons_beating_both,
        "primary_fraction_recordings_M2_beats_M1": fraction_beating_m1,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "action_if_fail": (
            "Preserve failure; do not request or open AML32_chip; "
            "CeRSI-v2 delta remains 0."
        ),
    }


def serialize_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "alpha": model["alpha"],
        "feature_mean": model["mean"].tolist(),
        "feature_scale": model["scale"].tolist(),
        "coef": model["coef"].tolist(),
        "intercept": model["intercept"],
    }


def serialize_family(family: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_alpha": family["selected_alpha"],
        "selection_rule": family["selection_rule"],
        "models_by_horizon": {
            str(horizon): serialize_model(family["models"][horizon])
            for horizon in HORIZONS
        },
    }


def _find_dataset_list(root: Path) -> Path:
    matches = sorted(root.rglob("*_datasets.txt"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one dataset list under {root}, found {matches}"
        )
    return matches[0]


def _find_recording_folder(root: Path, recording_id: str) -> Path:
    expected_name = f"{recording_id}_MS"
    matches = sorted(
        path for path in root.rglob(expected_name) if path.is_dir()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one folder {expected_name} under {root}, found {matches}"
        )
    return matches[0]


def load_archive_recordings(
    root: Path, role: str
) -> tuple[list[PreparedRecording], dict[str, Any]]:
    assert_not_holdout_path(root)
    dataset_list = _find_dataset_list(root)
    assert_not_holdout_path(dataset_list)
    text = dataset_list.read_text(encoding="utf-8")
    entries = parse_dataset_list_text(text)
    recordings = []
    receipt_records = []
    for entry in entries:
        folder = _find_recording_folder(root, entry.recording_id)
        assert_not_holdout_path(folder)
        loaded = load_recording_folder(
            folder, cut_volume_inclusive=entry.cut_volume_inclusive
        )
        prepared = prepare_recording(entry.recording_id, role, loaded)
        recordings.append(prepared)
        receipt_records.append({
            "recording_id": entry.recording_id,
            "cut_volume_inclusive": entry.cut_volume_inclusive,
            "neurons": prepared.neurons,
            "valid_frames": int(prepared.absolute_frames.size),
            "first_absolute_frame": int(prepared.absolute_frames[0]),
            "last_absolute_frame": int(prepared.absolute_frames[-1]),
        })
    return recordings, {
        "role": role,
        "root": str(root.resolve()),
        "dataset_list_path": str(dataset_list.resolve()),
        "dataset_list_sha256": sha256_file(dataset_list),
        "recordings": receipt_records,
    }


def run(
    training_root: Path, development_root: Path, output: Path
) -> dict[str, Any]:
    assert_not_holdout_path(training_root)
    assert_not_holdout_path(development_root)
    training_recordings, training_receipt = load_archive_recordings(
        training_root, "training"
    )
    development_recordings, development_receipt = load_archive_recordings(
        development_root, "development"
    )
    training_samples = build_all_samples(training_recordings)
    development_samples = build_all_samples(development_recordings)
    m1 = select_model_family(
        training_samples, development_samples, "X_behavior"
    )
    m2 = select_model_family(
        training_samples, development_samples, "X_neural_behavior"
    )
    gate = evaluate_gate(m1, m2, development_samples)
    model_payload = {
        "schema": "bio-001-s28-frozen-pre-holdout-model-v1",
        "target": "future CMSVelocity",
        "behavior_features": list(BEHAVIOR_NAMES),
        "neural_summary_features": list(NEURAL_SUMMARY_NAMES),
        "lags_frames": list(LAGS),
        "horizons_frames": list(HORIZONS),
        "ratio_denominator_floor": RATIO_DENOMINATOR_FLOOR,
        "strict_json": True,
        "absolute_frame_rule": (
            "Every lag and target must exist at its exact source absolute "
            "frame; samples never bridge removed invalid frames."
        ),
        "M1_behavior_only": serialize_family(m1),
        "M2_neural_plus_behavior": serialize_family(m2),
        "holdout_url_present": False,
        "sealed_token": SEALED_TOKEN,
    }
    results_payload = {
        "schema": "bio-001-s28-development-results-v1",
        "M1_selected_alpha": m1["selected_alpha"],
        "M2_selected_alpha": m2["selected_alpha"],
        "M1_candidates": m1["candidates"],
        "M2_candidates": m2["candidates"],
        "development_gate": gate,
        "holdout_requested": False,
        "CeRSI_v2_delta": 0.0,
    }
    extraction_payload = {
        "schema": "bio-001-s28-extraction-receipt-v1",
        "training": training_receipt,
        "development": development_receipt,
        "sample_counts": {
            "training": {
                str(horizon): {
                    block.recording_id: int(block.y.size)
                    for block in training_samples[horizon]
                }
                for horizon in HORIZONS
            },
            "development": {
                str(horizon): {
                    block.recording_id: int(block.y.size)
                    for block in development_samples[horizon]
                }
                for horizon in HORIZONS
            },
        },
        "holdout_accessed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "extraction_receipt.json", extraction_payload)
    write_json(output / "frozen_model_pre_holdout.json", model_payload)
    write_json(output / "development_results.json", results_payload)
    receipt = {
        "schema": "bio-001-s28-train-dev-run-receipt-v1",
        "status": (
            "DEVELOPMENT_PASS" if gate["pass"] else "DEVELOPMENT_FAIL"
        ),
        "extraction_receipt_sha256": sha256_file(
            output / "extraction_receipt.json"
        ),
        "frozen_model_sha256": sha256_file(
            output / "frozen_model_pre_holdout.json"
        ),
        "development_results_sha256": sha256_file(
            output / "development_results.json"
        ),
        "strict_json": True,
        "holdout_requested": False,
        "holdout_opened": False,
        "CeRSI_v2_delta": 0.0,
    }
    write_json(output / "run_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.training_root, args.development_root, args.output
    ), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
