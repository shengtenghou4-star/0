from __future__ import annotations

"""Python 3 port of the frozen PredictionCode recording loader semantics.

This module preserves the scientific semantics of
leiferlab/PredictionCode@226a27533d8b99ebf4448f0214fa19caf6a71a2b,
particularly utility/data_handler.py. It intentionally does not implement any
model fitting, archive selection, or holdout access.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import io as scipy_io
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.signal import medfilt

OFFICIAL_REPOSITORY = "leiferlab/PredictionCode"
OFFICIAL_COMMIT = "226a27533d8b99ebf4448f0214fa19caf6a71a2b"
OFFICIAL_SOURCE_PATH = "utility/data_handler.py"
VOLUME_ACQUISITION_RATE_HZ = 6.0
GCAMP_GAUSSIAN_SIGMA_FRAMES = 5.0
VALID_FRAME_MAX_NAN_FRACTION = 0.5


@dataclass(frozen=True)
class DatasetListEntry:
    recording_id: str
    cut_volume_inclusive: int | None


def parse_dataset_list_text(text: str) -> list[DatasetListEntry]:
    """Parse source-defined ``*_datasets.txt`` semantics."""
    entries: list[DatasetListEntry] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) not in (1, 2):
            raise ValueError(f"dataset-list line {line_number} has {len(tokens)} tokens")
        recording_id = tokens[0]
        if recording_id in seen:
            raise ValueError(f"duplicate recording id: {recording_id}")
        seen.add(recording_id)
        cutoff = int(tokens[1]) if len(tokens) == 2 else None
        if cutoff is not None and cutoff < 0:
            raise ValueError(f"negative cutoff for {recording_id}: {cutoff}")
        entries.append(DatasetListEntry(recording_id, cutoff))
    if not entries:
        raise ValueError("dataset list contains no recordings")
    return entries


def expfunc(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(-b * x) + c


def fit_photobleaching(activity_trace: np.ndarray, vps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Faithful Python 3 port of PredictionCode ``fitPhotobleaching``."""
    trace = np.asarray(activity_trace, dtype=float).copy()
    if trace.ndim != 1:
        raise ValueError("fit_photobleaching only accepts one-dimensional traces")
    if trace.size < 3:
        raise ValueError("photobleaching trace is too short")
    x_vals = np.arange(trace.shape[-1], dtype=float) / float(vps)
    non_nans = ~np.isnan(trace)
    if np.count_nonzero(non_nans) < 3:
        raise ValueError("photobleaching trace has fewer than three finite samples")
    scale_factor = float(np.nanmean(trace))
    if not np.isfinite(scale_factor) or scale_factor == 0.0:
        raise ValueError("photobleaching trace has invalid mean scale")
    scaled = trace / scale_factor
    max_x = float(np.nanmax(x_vals))
    finite_max = float(np.nanmax(scaled[non_nans]))
    finite_mean = float(np.nanmean(scaled))
    bounds = (
        [0.0, 1.0 / (8.0 * max_x), 0.0],
        [finite_max * 1.5, 0.5, 2.0 * finite_mean],
    )
    initial = [finite_max / 2.0, 2.0 / max_x, finite_mean]
    popt, pcov = curve_fit(
        expfunc, x_vals[non_nans], scaled[non_nans], p0=initial,
        bounds=bounds, maxfev=20000,
    )
    residual = scaled - expfunc(x_vals, *popt)
    exclude_outliers = non_nans.copy()
    with np.errstate(invalid="ignore"):
        exclude_outliers[np.abs(residual) > (3.0 * np.nanstd(residual))] = False
    if np.count_nonzero(exclude_outliers) < 3:
        exclude_outliers = non_nans
    try:
        popt, pcov = curve_fit(
            expfunc, x_vals[exclude_outliers], scaled[exclude_outliers],
            p0=popt, bounds=bounds, maxfev=20000,
        )
    except Exception:
        popt, pcov = curve_fit(
            expfunc, x_vals[exclude_outliers], scaled[exclude_outliers],
            p0=popt, maxfev=20000,
        )
    popt = np.asarray(popt, dtype=float)
    popt[0] *= scale_factor
    popt[2] *= scale_factor
    return popt, np.asarray(pcov, dtype=float), x_vals


def correct_photobleaching(raw: np.ndarray, vps: float = VOLUME_ACQUISITION_RATE_HZ) -> np.ndarray:
    """Apply the source exponential/flat-line photobleaching rule."""
    raw_array = np.asarray(raw, dtype=float)
    if raw_array.ndim not in (1, 2):
        raise ValueError("raw fluorescent traces must be one- or two-dimensional")
    if raw_array.ndim == 2 and raw_array.shape[1] <= raw_array.shape[0]:
        raise ValueError("expected neurons x time with more time points than neurons")
    medfilt_window = int(np.ceil(12.6 * vps / 2.0) * 2 + 1)
    if raw_array.ndim == 1:
        smoothed = medfilt(raw_array, medfilt_window)
        popt, _pcov, x_vals = fit_photobleaching(smoothed, vps)
        return popt[0] * raw_array / expfunc(x_vals, *popt)
    smoothed = medfilt(raw_array, [1, medfilt_window])
    corrected = np.zeros_like(raw_array, dtype=float)
    for row in range(raw_array.shape[0]):
        popt, _pcov, x_vals = fit_photobleaching(smoothed[row], vps)
        residual_fit = raw_array[row] - expfunc(x_vals, *popt)
        sum_sq_fit = float(np.nansum(np.square(residual_fit)))
        flat_line = float(np.nanmean(raw_array[row]))
        sum_sq_flat = float(np.nansum(np.square(raw_array[row] - flat_line)))
        if sum_sq_fit > sum_sq_flat:
            corrected[row] = raw_array[row]
        else:
            corrected[row] = popt[0] * raw_array[row] / expfunc(x_vals, *popt)
    return corrected


def close_nan_holes(values: np.ndarray) -> np.ndarray:
    """Preserve the source binary closing rule for isolated temporal NaNs."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("close_nan_holes expects neurons x time")
    structure = np.zeros((3, 3), dtype=int)
    structure[1, :] = 1
    nan_mask = np.isnan(array)
    closed = ndimage.binary_erosion(
        ndimage.binary_dilation(nan_mask, structure=structure),
        structure=structure,
    )
    output = array.copy()
    output[closed] = np.nan
    return output


def decorrelate_neurons_linear(red: np.ndarray, green: np.ndarray) -> np.ndarray:
    """Remove per-neuron green signal linearly explained by red signal."""
    red_array = np.asarray(red, dtype=float)
    green_array = np.asarray(green, dtype=float)
    if red_array.shape != green_array.shape or red_array.ndim != 2:
        raise ValueError("red and green must have equal neurons x time shape")
    output = np.full_like(green_array, np.nan, dtype=float)
    nan_mask = np.isnan(red_array) | np.isnan(green_array)
    for neuron in range(red_array.shape[0]):
        usable = ~nan_mask[neuron]
        if np.count_nonzero(usable) < 2:
            continue
        design = np.column_stack([
            red_array[neuron, usable], np.ones(np.count_nonzero(usable))
        ])
        best_fit, *_ = np.linalg.lstsq(
            design, green_array[neuron, usable], rcond=None
        )
        output[neuron] = green_array[neuron] - (
            best_fit[0] * red_array[neuron] + best_fit[1]
        )
    return output


def gauss_filter_nan(values: np.ndarray, sigma: float) -> np.ndarray:
    """Source NaN-weighted Gaussian smoothing with final interpolation."""
    vector = np.asarray(values, dtype=float).copy()
    if vector.ndim != 1:
        raise ValueError("gauss_filter_nan expects a one-dimensional trace")
    numerator_input = vector.copy()
    numerator_input[np.isnan(vector)] = 0.0
    numerator = ndimage.gaussian_filter1d(numerator_input, sigma=sigma)
    weights = np.ones_like(vector)
    weights[np.isnan(vector)] = 0.0
    denominator = ndimage.gaussian_filter1d(weights, sigma=sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        interpolated = numerator / denominator
    valid = np.isfinite(interpolated)
    invalid = ~valid
    if np.any(invalid) and np.any(valid):
        interpolated[invalid] = np.interp(
            np.flatnonzero(invalid), np.flatnonzero(valid), interpolated[valid]
        )
    elif np.any(invalid):
        interpolated[invalid] = 0.0
    return interpolated


def get_curvature(centerlines: np.ndarray) -> np.ndarray:
    """Source curvature calculation in inverse body-length units."""
    centerline_array = np.asarray(centerlines, dtype=float)
    if centerline_array.ndim != 3 or centerline_array.shape[2] != 2:
        raise ValueError("centerlines must have shape frames x points x 2")
    num_curvature_points = centerline_array.shape[1]
    diff_vec = np.diff(centerline_array, axis=1)
    tangent_angle = np.unwrap(
        np.arctan2(-diff_vec[:, :, 1], diff_vec[:, :, 0]), axis=-1
    )
    curvature = np.unwrap(np.diff(tangent_angle, axis=1), axis=-1)
    return curvature * num_curvature_points


def get_curvature_metric(
    curvature: np.ndarray,
    roi_start: int = 15,
    roi_end: int = 80,
    num_stds: float = 6.0,
) -> np.ndarray:
    """Mean source curvature in the fixed body ROI with interpolation."""
    array = np.asarray(curvature, dtype=float)
    if array.ndim != 2:
        raise ValueError("curvature must be frames x body-points")
    if roi_start < 0 or roi_end > array.shape[1] or roi_start >= roi_end:
        raise ValueError(f"invalid curvature ROI {roi_start}:{roi_end} for {array.shape}")
    metric = np.mean(array[:, roi_start:roi_end], axis=1)
    outliers = np.abs(metric - np.mean(metric)) > num_stds * np.std(metric)
    metric = metric.copy()
    metric[outliers] = np.nan
    missing = np.isnan(metric)
    finite = ~missing
    if np.any(missing):
        if not np.any(finite):
            raise ValueError("curvature metric has no finite values")
        metric[missing] = np.interp(
            np.flatnonzero(missing), np.flatnonzero(finite), metric[finite]
        )
    return metric


def nearest_centerline_indices(
    centerline_time: np.ndarray, volume_time: np.ndarray
) -> np.ndarray:
    cl_time = np.asarray(centerline_time, dtype=float).squeeze()
    vol_time = np.asarray(volume_time, dtype=float).squeeze()
    if cl_time.ndim != 1 or vol_time.ndim != 1:
        raise ValueError("time arrays must be one-dimensional")
    if np.any(np.diff(cl_time) < 0):
        raise ValueError("centerline time is not monotonic")
    indices = np.rint(
        np.interp(vol_time, cl_time, np.arange(cl_time.size))
    ).astype(int)
    return np.clip(indices, 0, cl_time.size - 1)


def _as_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).squeeze()
    if vector.ndim != 1:
        raise ValueError(f"{name} is not one-dimensional: {vector.shape}")
    return vector


def preprocess_recording_arrays(
    *,
    r_raw: np.ndarray,
    g_raw: np.ndarray,
    r_photo_corr_mask: np.ndarray,
    g_photo_corr_mask: np.ndarray,
    volume_time: np.ndarray,
    velocity: np.ndarray,
    curvature_metric_at_volume_time: np.ndarray,
    x_position: np.ndarray,
    y_position: np.ndarray,
    cut_volume_inclusive: int | None,
    flagged_volumes: Iterable[int] = (),
    window_gcamp: float = GCAMP_GAUSSIAN_SIGMA_FRAMES,
    volume_acquisition_rate: float = VOLUME_ACQUISITION_RATE_HZ,
) -> dict[str, Any]:
    """Apply official neural/behavior preprocessing to extracted arrays."""
    r_raw_array = np.asarray(r_raw, dtype=float)
    g_raw_array = np.asarray(g_raw, dtype=float)
    if r_raw_array.shape != g_raw_array.shape or r_raw_array.ndim != 2:
        raise ValueError("r_raw and g_raw must share neurons x time shape")
    time = _as_vector(volume_time, "volume_time")
    n_time = min(r_raw_array.shape[1], time.size)
    r_raw_array = r_raw_array[:, :n_time]
    g_raw_array = g_raw_array[:, :n_time]
    time = time[:n_time]
    r_mask = np.asarray(r_photo_corr_mask, dtype=float)[:, :n_time]
    g_mask = np.asarray(g_photo_corr_mask, dtype=float)[:, :n_time]
    if r_mask.shape != r_raw_array.shape or g_mask.shape != g_raw_array.shape:
        raise ValueError("photo-correction masks must match raw arrays")
    all_indices = np.arange(n_time)
    cutoff = n_time if cut_volume_inclusive is None else int(cut_volume_inclusive)
    data_indices = all_indices[all_indices <= cutoff]
    identity_indices = all_indices[all_indices > cutoff]
    if data_indices.size == 0:
        raise ValueError("cutoff leaves no behavior data")
    red = correct_photobleaching(
        r_raw_array[:, data_indices], volume_acquisition_rate
    )
    green = correct_photobleaching(
        g_raw_array[:, data_indices], volume_acquisition_rate
    )
    red[np.isnan(r_mask[:, data_indices])] = np.nan
    green[np.isnan(g_mask[:, data_indices])] = np.nan
    red = close_nan_holes(red)
    green = close_nan_holes(green)
    flagged = np.asarray(list(flagged_volumes), dtype=int).ravel()
    flagged = flagged[(flagged >= 0) & (flagged < red.shape[1])]
    if flagged.size:
        red[:, flagged] = np.nan
        green[:, flagged] = np.nan
    corrected = decorrelate_neurons_linear(red, green)
    smoothed_interp = np.vstack([
        gauss_filter_nan(row, window_gcamp) for row in corrected
    ])
    if not np.all(np.isfinite(smoothed_interp)):
        raise ValueError("source smoothing produced nonfinite values")
    valid_map_local = np.flatnonzero(
        np.mean(np.isnan(corrected), axis=0) < VALID_FRAME_MAX_NAN_FRACTION
    )
    if valid_map_local.size == 0:
        raise ValueError("no frames pass the source majority-neuron validity gate")
    valid_map_absolute = data_indices[valid_map_local]
    velocity_vector = _as_vector(velocity, "velocity")[:n_time]
    curvature_vector = _as_vector(
        curvature_metric_at_volume_time, "curvature"
    )[:n_time]
    x_vector = _as_vector(x_position, "x_position")[:n_time]
    y_vector = _as_vector(y_position, "y_position")[:n_time]
    if min(
        velocity_vector.size, curvature_vector.size,
        x_vector.size, y_vector.size
    ) < n_time:
        raise ValueError("behavior arrays are shorter than aligned neural timebase")
    shifted_time = time.copy()
    shifted_time -= shifted_time[valid_map_absolute[0]]
    return {
        "BehaviorFull": {
            "CMSVelocity": velocity_vector[data_indices],
            "Curvature": curvature_vector[data_indices],
            "X": x_vector[data_indices],
            "Y": y_vector[data_indices],
        },
        "Behavior_crop_noncontig": {
            "CMSVelocity": velocity_vector[valid_map_absolute],
            "Curvature": curvature_vector[valid_map_absolute],
            "X": x_vector[valid_map_absolute],
            "Y": y_vector[valid_map_absolute],
        },
        "Neurons": {
            "I": corrected,
            "I_Time": shifted_time[data_indices],
            "I_smooth_interp": smoothed_interp,
            "I_smooth_interp_crop_noncontig": smoothed_interp[:, valid_map_local],
            "I_Time_crop_noncontig": shifted_time[valid_map_absolute],
            "I_valid_map": valid_map_absolute,
        },
        "Identities": {
            "rRaw": r_raw_array[:, identity_indices],
            "gRaw": g_raw_array[:, identity_indices],
        },
        "audit": {
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
            "source_path": OFFICIAL_SOURCE_PATH,
            "cut_volume_inclusive": cut_volume_inclusive,
            "input_frames": int(n_time),
            "behavior_frames_before_validity_gate": int(data_indices.size),
            "valid_frames": int(valid_map_local.size),
            "neurons": int(r_raw_array.shape[0]),
            "flagged_volumes_applied": flagged.tolist(),
            "validity_rule": "mean_nan_fraction_across_neurons < 0.5",
            "source_behavior_names": ["CMSVelocity", "Curvature", "X", "Y"],
        },
    }


def _extract_behavior_cell(behavior: np.ndarray) -> tuple[np.ndarray, ...]:
    """Mirror ``data['behavior'][0][0].T`` from frozen source."""
    cell = np.asarray(behavior, dtype=object)[0][0]
    fields = tuple(np.asarray(cell, dtype=object).T)
    if len(fields) != 6:
        raise ValueError(f"expected six behavior fields, found {len(fields)}")
    return tuple(np.asarray(field) for field in fields)


def load_recording_folder(
    folder: str | Path,
    *,
    cut_volume_inclusive: int | None = None,
) -> dict[str, Any]:
    """Load one extracted recording folder with source-native semantics."""
    folder_path = Path(folder)
    heat_path = folder_path / "heatDataMS.mat"
    if not heat_path.exists():
        heat_path = folder_path / "heatData.mat"
    heat = scipy_io.loadmat(heat_path)
    _etho, x_pos, y_pos, velocity_matrix, _pc12, _pc3 = \
        _extract_behavior_cell(heat["behavior"])
    velocity = np.asarray(velocity_matrix, dtype=float)[:, 0]
    centerline_data = scipy_io.loadmat(folder_path / "centerline.mat")
    centerline = np.rollaxis(
        np.asarray(centerline_data["centerline"], dtype=float), 2, 0
    )
    centerline_time = _as_vector(heat["clTime"], "clTime")
    volume_time = _as_vector(heat["hasPointsTime"], "hasPointsTime")
    centerline_indices = nearest_centerline_indices(
        centerline_time, volume_time
    )
    curvature = get_curvature(centerline)
    curvature_metric = get_curvature_metric(curvature)
    curvature_at_volume = curvature_metric[centerline_indices]
    flagged: list[int] = []
    if "flagged_volumes" in heat and np.asarray(heat["flagged_volumes"]).size:
        flagged = np.asarray(heat["flagged_volumes"])[0].astype(int).ravel().tolist()
    return preprocess_recording_arrays(
        r_raw=np.asarray(heat["rRaw"], dtype=float),
        g_raw=np.asarray(heat["gRaw"], dtype=float),
        r_photo_corr_mask=np.asarray(heat["rPhotoCorr"], dtype=float),
        g_photo_corr_mask=np.asarray(heat["gPhotoCorr"], dtype=float),
        volume_time=volume_time,
        velocity=velocity,
        curvature_metric_at_volume_time=curvature_at_volume,
        x_position=np.asarray(x_pos, dtype=float),
        y_position=np.asarray(y_pos, dtype=float),
        cut_volume_inclusive=cut_volume_inclusive,
        flagged_volumes=flagged,
    )
