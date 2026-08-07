"""Signed utility calibration for LAZARUS residual routing.

The calibrator maps training-only, group-cross-fitted regime probabilities to the
fraction of a signed residual correction that should be applied under squared
error. It is deliberately small: one monotone sigmoid for positive corrections
and one for negative corrections. Target-batch statistics are never accepted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-8


@dataclass(frozen=True)
class MonotoneSigmoid:
    slope: float
    intercept: float

    def transform(self, probability: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), _EPS, 1.0 - _EPS)
        logit = np.log(p) - np.log1p(-p)
        z = np.clip(self.slope * logit + self.intercept, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class SignedUtilityCalibration:
    positive: MonotoneSigmoid
    negative: MonotoneSigmoid
    objective: str = "regime-balanced delta-squared weighted OOF prediction MSE"
    supervised_calibration_fits: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    def predict(
        self,
        raw_prediction: np.ndarray,
        residual_correction: np.ndarray,
        p_resistant: np.ndarray,
        p_potent: np.ndarray,
    ) -> np.ndarray:
        raw = np.asarray(raw_prediction, dtype=float)
        delta = np.asarray(residual_correction, dtype=float)
        p_r = np.asarray(p_resistant, dtype=float)
        p_p = np.asarray(p_potent, dtype=float)
        if not (raw.shape == delta.shape == p_r.shape == p_p.shape):
            raise ValueError("all prediction arrays must have identical shape")
        if not np.all(np.isfinite(np.column_stack([raw, delta, p_r, p_p]))):
            raise ValueError("non-finite inference input")
        if np.any((p_r < 0.0) | (p_r > 1.0) | (p_p < 0.0) | (p_p > 1.0)):
            raise ValueError("probabilities must be in [0, 1]")
        weight = np.where(
            delta >= 0.0,
            self.positive.transform(p_p),
            self.negative.transform(p_r),
        )
        return raw + delta * weight


def _regime_balanced_delta_weights(delta: np.ndarray, regime: np.ndarray) -> np.ndarray:
    base = np.square(delta)
    out = np.zeros_like(base)
    present = []
    for label in ("resistant", "middle", "potent"):
        mask = regime == label
        total = float(np.sum(base[mask]))
        if total > 0.0:
            present.append((mask, total))
    if len(present) < 2:
        raise ValueError("at least two regimes with nonzero correction energy are required")
    for mask, total in present:
        out[mask] = base[mask] / total / len(present)
    return out


def _fit_one(
    probability: np.ndarray,
    raw: np.ndarray,
    delta: np.ndarray,
    truth: np.ndarray,
    regime: np.ndarray,
) -> MonotoneSigmoid:
    if len(probability) < 50:
        raise ValueError("at least 50 OOF rows are required per correction sign")
    weights = _regime_balanced_delta_weights(delta, regime)
    p = np.clip(probability, _EPS, 1.0 - _EPS)
    logit = np.log(p) - np.log1p(-p)

    def objective(theta: np.ndarray) -> float:
        slope, intercept = theta
        z = np.clip(slope * logit + intercept, -40.0, 40.0)
        utility_weight = 1.0 / (1.0 + np.exp(-z))
        prediction = raw + delta * utility_weight
        mse = float(np.sum(weights * np.square(truth - prediction)))
        regularization = 1e-4 * ((slope - 1.0) ** 2 + intercept ** 2)
        return mse + regularization

    result = minimize(
        objective,
        x0=np.array([1.0, 0.0], dtype=float),
        method="L-BFGS-B",
        bounds=((0.0, 8.0), (-8.0, 8.0)),
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 2000, "maxls": 50},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"utility calibration optimization failed: {result.message}")
    return MonotoneSigmoid(float(result.x[0]), float(result.x[1]))


def fit_signed_utility_calibration(
    *,
    raw_prediction_oof: Iterable[float],
    residual_correction_oof: Iterable[float],
    p_resistant_oof: Iterable[float],
    p_potent_oof: Iterable[float],
    truth_training: Iterable[float],
    actual_regime_training: Iterable[str],
    oof_fold_id: Iterable[int],
) -> SignedUtilityCalibration:
    """Fit exactly two deterministic training-only utility calibrators.

    Every supplied row must be genuinely out-of-fold for the raw, residual and
    multinomial-router predictions. The caller must generate the fold IDs before
    reading labels and must preserve them in the execution receipt.
    """
    raw = np.asarray(list(raw_prediction_oof), dtype=float)
    delta = np.asarray(list(residual_correction_oof), dtype=float)
    p_r = np.asarray(list(p_resistant_oof), dtype=float)
    p_p = np.asarray(list(p_potent_oof), dtype=float)
    truth = np.asarray(list(truth_training), dtype=float)
    regime = np.asarray(list(actual_regime_training), dtype=str)
    fold_id = np.asarray(list(oof_fold_id), dtype=int)
    n = len(raw)
    if n == 0 or not all(len(x) == n for x in (delta, p_r, p_p, truth, regime, fold_id)):
        raise ValueError("all OOF training arrays must be nonempty and equal length")
    if len(np.unique(fold_id)) != 5 or np.any(fold_id < 0) or np.any(fold_id > 4):
        raise ValueError("exactly five OOF folds numbered 0..4 are required")
    if not set(np.unique(regime)).issubset({"resistant", "middle", "potent"}):
        raise ValueError("invalid training regime")
    matrix = np.column_stack([raw, delta, p_r, p_p, truth])
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite OOF training input")
    if np.any((p_r < 0.0) | (p_r > 1.0) | (p_p < 0.0) | (p_p > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")

    positive_mask = delta >= 0.0
    negative_mask = ~positive_mask
    positive = _fit_one(
        p_p[positive_mask], raw[positive_mask], delta[positive_mask],
        truth[positive_mask], regime[positive_mask],
    )
    negative = _fit_one(
        p_r[negative_mask], raw[negative_mask], delta[negative_mask],
        truth[negative_mask], regime[negative_mask],
    )
    return SignedUtilityCalibration(positive=positive, negative=negative)
