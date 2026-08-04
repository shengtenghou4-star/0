#!/usr/bin/env python3
"""Public-safe frozen LAZARUS calibration-orthogonal residual runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge

CANDIDATES = (
    "orth_residual_uniform_alpha10_v1",
    "orth_residual_tail2_alpha10_v1",
    "orth_residual_uniform_novelty_alpha10_v1",
    "orth_residual_tail2_novelty_alpha10_v1",
)
CONFIGS = (
    {"name": CANDIDATES[0], "pointwise_tail_multiplier": 1.0, "novelty_gate": False},
    {"name": CANDIDATES[1], "pointwise_tail_multiplier": 2.0, "novelty_gate": False},
    {"name": CANDIDATES[2], "pointwise_tail_multiplier": 1.0, "novelty_gate": True},
    {"name": CANDIDATES[3], "pointwise_tail_multiplier": 2.0, "novelty_gate": True},
)
REFERENCE = "raw_bilinear_ridge_alpha_10"
RIDGE_ALPHA = 10.0
EXPECTED_WIDTH = 2400
EXPECTED_TOTAL_FITS = 25
GATES = {
    "maximum_per_seed_overall_rmse_relative_degradation": 0.005,
    "maximum_per_seed_overall_pearson_degradation": 0.02,
    "minimum_mean_overall_prediction_truth_std_ratio_gain": 0.03,
    "minimum_potent_rmse_win_seeds": 4,
    "minimum_potent_absolute_bias_win_seeds": 4,
    "minimum_potent_spearman_win_seeds": 4,
    "minimum_mean_potent_prediction_truth_std_ratio_gain": 0.05,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_legacy(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("lazarus_legacy_public", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reviewed legacy payload: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "SEEDS", "REFERENCE", "RESISTANT", "POTENT", "read_pairs", "feature_tables",
        "selected_embed", "joint", "eval_rows", "write_csv", "load_family_map",
        "resolve_antibodies", "build_edges", "freeze_splits", "build_matrix", "map_splits",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ValueError(f"reviewed payload interface missing: {missing}")
    if tuple(module.SEEDS) != (626, 627, 628, 629, 630):
        raise ValueError("development seed set drifted")
    if str(module.REFERENCE) != REFERENCE:
        raise ValueRror("raw reference identity drifted")
    return module


def point_weights(target: np.ndarray, multiplier: float, resistant: float, potent: float) -> np.ndarray:
    if multiplier not in (1.0, 2.0):
        raise ValueError("tail multiplier drifted")
    weights = np.ones(target.shape, dtype=np.float64)
    weights[(target <= resistant) | (target >= potent)] = multiplier
    weights /= float(np.mean(weights))
    return weights


def orthogonalize_features(
    train_features: np.ndarray,
    test_features: np.ndarray,
    raw_train: np.ndarray,
    raw_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("feature matrices must be two-dimensional")
    if train_features.shape[1] != EXPECTED_WIDTH or test_features.shape[1] != EXPECTED_WIDTH:
        raise ValueError("correction feature width drifted")
    if len(train_features) != len(raw_train) or len(test_features) != len(raw_test):
        raise ValueError("raw-score and feature row counts differ")
    nuisance_train = np.column_stack((np.ones(len(raw_train), dtype=np.float64), raw_train))
    nuisance_test = np.column_stack((np.ones(len(raw_test), dtype=np.float64), raw_test))
    gram = nuisance_train.T @ nuisance_train
    if not np.isfinite(gram).all() or np.linalg.cond(gram) > 1e12:
        raise ValueError("ill-conditioned nuisance design")
    cross = nuisance_train.T @ np.asarray(train_features, dtype=np.float64)
    coefficients = np.linalg.solve(gram, cross)
    train_orth = np.asarray(train_features, dtype=np.float64) - nuisance_train @ coefficients
    test_orth = np.asarray(test_features, dtype=np.float64) - nuisance_test @ coefficients
    means = np.abs(np.mean(train_orth, axis=0))
    centered_raw = raw_train - float(np.mean(raw_train))
    raw_norm = float(np.linalg.norm(centered_raw))
    col_norm = np.linalg.norm(train_orth, axis=0)
    denom = raw_norm * col_norm
    inner = np.zeros(train_orth.shape[1], dtype=np.float64)
    valid = (denom > 0) & (col_norm > 1e-12)
    inner[valid] = np.abs(centered_raw @ train_orth[:, valid]) / denom[valid]
    diagnostics = {
        "maximum_absolute_feature_mean": float(np.max(means)),
        "maximum_absolute_feature_raw_score_inner_product_after_normalization": float(np.max(inner)),
        "nuisance_gram_condition_number": float(np.linalg.cond(gram)),
        "zero_norm_feature_columns": int(np.sum(~valid)),
    }
    if diagnostics["maximum_absolute_feature_mean"] > 1e-10:
        raise AssertionError(f"constant orthogonality failed: {diagnostics}")
    if diagnostics["maximum_absolute_feature_raw_score_inner_product_after_normalization"] > 1e-10:
        raise AssertionError(f"raw-score orthogonality failed: {diagnostics}")
    if not np.isfinite(train_orth).all() or not np.isfinite(test_orth).all():
        raise ValueError("non-finite orthogonalized features")
    return train_orth, test_orth, diagnostics



def entity_novelty(
    train_embeddings: dict[str, np.ndarray],
    test_embeddings: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, Any]]:
    train_ids = sorted(train_embeddings)
    test_ids = sorted(test_embeddings)
    if len(train_ids) < 2 or not test_ids:
        raise ValueError("insufficient entities for novelty gate")
    train = np.asarray([train_embeddings[key] for key in train_ids], dtype=np.float64)
    test = np.asarray([test_embeddings[key] for key in test_ids], dtype=np.float64)
    train_dist = cdist(train, train, metric="euclidean")
    np.fill_diagonal(train_dist, np.inf)
    leave_one_nearest = np.min(train_dist, axis=1)
    scale = float(np.median(leave_one_nearest))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid novelty normalization scale")
    test_nearest = np.min(cdist(test, train, metric="euclidean"), axis=1)
    normalized = test_nearest / scale
    excess = np.maximum(0.0, normalized - 1.0)
    result = {key: float(value) for key, value in zip(test_ids, excess, strict=True)}
    return result, {
        "training_entities": len(train_ids),
        "test_entities": len(test_ids),
        "median_training_leave_one_nearest_distance": scale,
        "minimum_test_normalized_distance": float(np.min(normalized)),
        "median_test_normalized_distance": float(np.median(normalized)),
        "maximum_test_normalized_distance": float(np.max(normalized)),
        "minimum_test_excess": float(np.min(excess)),
        "median_test_excess": float(np.median(excess)),
        "maximum_test_excess": float(np.max(excess)),
    }



def row_novelty_gate(
    rows: list[Any],
    antibody_excess: dict[str, float],
    virus_excess: dict[str, float],
) -> np.ndarray:
    gate = np.asarray(
        [1.0 / (1.0 + 0.5 * (antibody_excess[row.antibody] + virus_excess[row.virus])) for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(gate).all() or float(np.min(gate)) <= 0 or float(np.max(gate)) > 1.0:
        raise ValueError("invalid novelty gate")
    return gate



def fit_residual_candidate(
    train_features: np.ndarray,
    residual_target: np.ndarray,
    test_features: np.ndarray,
    sample_weight: np.ndarray,
    novelty_gate: np.ndarray,
    *,
    apply_novelty_gate: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False, solver="lsqr", tol=1e-8)
    model.fit(train_features, residual_target, sample_weight=sample_weight)
    correction = np.asarray(model.predict(test_features), dtype=np.float64)
    if apply_novelty_gate:
        correction = correction * novelty_gate
    if not np.isfinite(correction).all():
        raise ValueError("non-finite residual correction")
    return correction, {
        "coefficient_norm": float(np.linalg.norm(model.coef_)),
        "ungated_correction_mean": float(np.mean(model.predict(test_features))),
        "final_correction_mean": float(np.mean(correction)),
        "final_correction_std": float(np.std(correction)),
        "novelty_gate_applied": bool(apply_novelty_gate),
    }


def summarize(metric_rows: list[dict[str, Any]], seeds: tuple[int, ...]) -> tuple[list[dict[str, Any]], str | None]:
    indexed = {(int(row["seed"]), str(row["model"]), str(row["regime"])): row for row in metric_rows}
    output: list[dict[str, Any]] = []
    for rankK[Ù[[ˆ[[Y\˜]JÐS‘QUTÊN‚ˆ™[ˆ\ÝÙ›Ø]HH×BˆX\œÛÛ—ÙØZ[Žˆ\ÝÙ›Ø]HH×BˆÜX\›X[—ÙØZ[Žˆ\ÝÙ›Ø]HH×Bˆ›\ÙWÙØZ[Žˆ\ÝÙ›Ø]HH×Bˆ\Ü\œÚ[Û—ÙØZ[Žˆ\ÝÙ›Ø]HH×BˆÝ[Ü›\ÙNˆ\ÝÙ›Ø]HH×BˆÝ[ØšX\Îˆ\ÝÙ›Ø]HH×BˆÝ[ÜÜX\›X[Žˆ\ÝÙ›Ø]HH×BˆÝ[Ù\Ü\œÚ[ÛŽˆ\ÝÙ›Ø]HH×Bˆ›ÜˆÙYY[ˆÙYYÎ‚ˆ˜]ÈH[™^YÊÙYY‘Q‘T‘SÑK›Ý™\˜[ŠWBˆÝ\œ™[H[™^YÊÙYY[Ù[›Ý™\˜[ŠWBˆ˜]×ÜH[™^YÊÙYY‘Q‘T‘SÑKœÝ[ŠWBˆÝ\œ™[ÜH[™^YÊÙYY[Ù[œÝ[ŠWBˆ™[˜\[™
›Ø]
Ý\œ™[Èœ›\ÙH—JHÈ›Ø]
˜]ÖÈœ›\ÙH—JHHKŒ
BˆX\œÛÛ—ÙØZ[‹˜\[™
›Ø]
Ý\œ™[ÈœX\œÛÛˆ—JHH›Ø]
˜]ÖÈœX\œÛÛˆ—JJBˆÜX\›X[—ÙØZ[‹˜\[™
›Ø]
Ý\œ™[ÈœÜX\›X[ˆ—JHH›Ø]
˜]ÖÈœÜX\›X[ˆ—JJBˆ›\ÙWÙØZ[‹˜\[™
›Ø]
˜]ÖÈœ›\ÙH—JHH›Ø]
Ý\œ™[Èœ›\ÙH—JJBˆ\Ü\œÚ[Û—ÙØZ[‹˜\[™
›Ø]
Ý\œ™[Èœ™YXÝ[Û—Ý]ÜÝÜ˜][È—JHH›Ø]
˜]ÖÈœ™YXÝ[Û—Ý]ÜÝÜ˜][È—JJBˆÝ[Ü›\ÙK˜\[™
›Ø]
˜]×ÜÈœ›\ÙH—JHH›Ø]
Ý\œ™[ÜÈœ›\ÙH—JJBˆÝ[ØšX\Ë˜\[™
XœÊ›Ø]
˜]×ÜÈ˜šX\È—JJHHXœÊ›Ø]
Ý\œ™[ÜÈ˜šX\È—JJJBˆÝ[ÜÜX\›X[‹˜\[™
›Ø]
Ý\œ™[ÜÈœÜX\›X[ˆ—JHH›Ø]
˜]×ÜÈœÜX\›X[ˆ—JJBˆÝ[Ù\Ü\œÚ[Û‹˜\[™
›Ø]
Ý\œ™[ÜÈœ™YXÝ[Û—Ý]ÜÝÜ˜][È—JHH›Ø]
˜]×ÜÈœ™YXÝ[Û—Ý]ÜÝÜ˜][È—JJBˆØ]WÝ˜[Y\ÈHÂˆ›Ý™\˜[Ü›\ÙWÙÝX\™Ü\ÜÈŽˆX^
™[
HHÐUTÖÈ›X^[][WÜ\—ÜÙYYÛÝ™\˜[Ü›\ÙWÜ™[]]™WÙYÜ˜Y][Ûˆ—Kˆ›Ý™\˜[ÜX\œÛÛ—ÙÝX\™Ü\ÜÈŽˆZ[ŠX\œÛÛ—ÙØZ[ŠHHQÐUTÖÈ›X^[][WÜ\—ÜÙYYÛÝ™\˜[ÜX\œÛÛ—ÙYÜ˜Y][Ûˆ—Kˆ›YX[—ÛÝ™\˜[Ü›\ÙWÚ[\›Ý™\ÈŽˆ›Ø]
œ›YX[Š›\ÙWÙØZ[ŠJHˆˆ›YX[—ÛÝ™\˜[ÜÜX\›X[—Ú[\›Ý™\ÈŽˆ›Ø]
œ›YX[ŠÜX\›X[—ÙØZ[ŠJHˆˆ›Ý™\˜[Ù\Ü\œÚ[Û—ÙØ]WÜ\ÜÈŽˆ›Ø]
œ›YX[Š\Ü\œÚ[Û—ÙØZ[ŠJHHÐUTÖÈ›Z[š[][WÛYX[—ÛÝ™\˜[Ü™YXÝ[Û—Ý]ÜÝÜ˜][×ÙØZ[ˆ—KˆœÝ[Ü›\ÙWÝÚ[—ÙØ]WÜ\ÜÈŽˆÝ[J˜[YHˆ›Üˆ˜[YH[ˆÝ[Ü›\ÙJHHÐUTÖÈ›Z[š[][WÜÝ[Ü›\ÙWÝÚ[—ÜÙYYÈ—KˆœÝ[ØšX\×ÝÚ[—ÙØ]WÜ\ÜÈŽˆÝ[J˜[YHˆ›Üˆ˜[YH[ˆÝ[ØšX\ÊHHÐUTÖÈ›Z[š[][WÜÝ[ØXœÛÛ]WØšX\×ÝÚ[—ÜÙYYÈ—KˆœÝ[ÜÜX\›X[—ÝÚ[—ÙØ]WÜ\ÜÈŽˆÝ[J˜[YHˆ›Üˆ˜[YH[ˆÝ[ÜÜX\›X[ŠHHÐUTÖÈ›Z[š[][WÜÝ[ÜÜX\›X[—ÝÚ[—ÜÙYYÈ—Kˆ›YX[—ÜÝ[Ü›\ÙWÚ[\›Ý™\ÈŽˆ›Ø]
œ›YX[ŠÝ[Ü›\ÙJJHˆˆ›YX[—ÜÝ[ØšX\×Ú[\›Ý™\ÈŽˆ›Ø]
œ›YX[ŠÝ[ØšX\ÊJHˆˆ›YX[—ÜÝ[ÜÜX\›X[—Ú[\›Ý™\ÈŽˆ›Ø]
œ›YX[ŠÝ[ÜÜX\›X[ŠJHˆˆœÝ[Ù\Ü\œÚ[Û—ÙØ]WÜ\ÜÈŽˆ›Ø]
œ›YX[ŠÝ[Ù\Ü\œÚ[ÛŠJHHÐUTÖÈ›Z[š[][WÛYX[—ÜÝ[Ü™YXÝ[Û—Ý]ÜÝÜ˜][×ÙØZ[ˆ—KˆBˆÝ]]˜\[™
ßBˆ›[Ù[Žˆ[Ù[ˆ˜ÛÛ\^]WÜ˜[šÈŽˆ˜[šËˆ™[YÚX›HŽˆ[
Ø]WÝ˜[Y\Ë˜[Y\Ê
JKˆ›X^[][WÛÝ™\˜[Ü›\ÙWÜ™[]]™WØÚ[™ÙHŽˆX^
™[
Kˆ›Z[š[][WÛÝ™\˜[ÜX\œÛÛ—ÙØZ[ˆŽˆZ[ŠX\œÛÛ—ÙØZ[ŠKˆ›YX[—ÛÝ™\˜[Ü›\ÙWÙØZ[ˆŽˆ›Ø]
œ›YX[Š›\ÙWÙØZ[ŠJKˆ›YX[—ÛÝ™\˜[ÜÜX\›X[—ÙØZ[ˆŽˆ›Ø]
œ›YX[ŠÜX\›X[—ÙØZ[ŠJKˆ›YX[—ÛÝ™\˜[Ù\Ü\œÚ[Û—ÙØZ[ˆŽˆ›Ø]
œ›YX[Š\Ü\œÚ[Û—ÙØZ[ŠJKˆœÝ[Ü›\ÙWÝÚ[—ÜÙYYÈŽˆÝ[J˜[YHˆ›Üˆ˜[YH[ˆÝ[Ü›\ÙJKˆ›Z[š[][WÜÝ[Ü›\ÙWÙØZ[ˆŽˆZ[ŠÝ[Ü›\ÙJKˆ›YX[—ÜÝ[Ü›\ÙWÙØZ[ˆŽˆ›Ø]
œ›YX[ŠÝ[Ü›\ÙJJKˆœÝ[ØXœÛÛ]WØšX\×ÝÚ[—ÜÙYYÈŽˆÝ[J˜[YHˆ›Üˆ˜[YH[ˆÝ[ØšX\ÊKˆ›Z[š[][WÜÝ[ØXœÛÛ]WØšX\×ÙØZ[ˆŽˆZ[ŠÝ[ØšX\ÊKˆ›YX[—ÜÝ[ØXœÛÛ]WØšX\×ÙØZ[ˆŽˆ›Ø]
œ›YX[ŠÝ[ØšX\ÊJKˆœÝ[ÜÜX\›X[—ÝÚ[—ÜÙYYÈŽˆÝ[J˜[YHˆ›Üˆ˜[YH[ˆÝ[ÜÜX\›X[ŠKˆ›Z[š[][WÜÝ[ÜÜX\›X[—ÙØZ[ˆŽˆZ[ŠÝ[ÜÜX\›X[ŠKˆ›YX[—ÜÝ[ÜÜX\›X[—ÙØZ[ˆŽˆ›Ø]
œ›YX[ŠÝ[ÜÜX\›X[ŠJKˆ›YX[—ÜÝ[Ù\Ü\œÚ[Û—ÙØZ[ˆŽˆ›Ø]
œ›YX[ŠÝ[Ù\Ü\œÚ[ÛŠJKˆ
Š™Ø]WÝ˜[Y\ËˆJBˆ[YÚX›HHÜ›ÝÈ›Üˆ›ÝÈ[ˆÝ]]Yˆ›ÝÖÈ™[YÚX›H—WBˆ[YÚX›KœÛÜ
Ù^O[[X™H›ÝÎˆ
ˆY›Ø]
›ÝÖÈ›Z[š[][WÜÝ[Ü›\ÙWÙØZ[ˆ—JKˆY›Ø]
›ÝÖÈ›Z[š[][WÜÝ[ØXœÛÛ]WØšX\×ÙØZ[ˆ—JKˆY›Ø]
›ÝÖÈ›Z[š[][WÜÝ[ÜÜX\›X[—ÙØZ[ˆ—JKˆY›Ø]
›ÝÖÈ›YX[—ÛÝ™\˜[Ü›\ÙWÙØZ[ˆ—JKˆ[
›ÝÖÈ˜ÛÛ\^]WÜ˜[šÈ—JKˆ
JBˆ™]\›ˆÝ]]ÝŠ[YÚX›VÌVÈ›[Ù[—JHYˆ[YÚX›H[ÙH›Û™B‚‚™Yˆ[—Û[Ù[ÊYØXÞNˆ[žKX\YˆXÝÚ[\VÔ]]WK[X›ÙWÜÙ\]Y[˜Ù\ÎˆXÝÜÝ‹Ý—Kš\\×ÜÙ\]Y[˜Ù\ÎˆXÝÜÝ‹Ý—KÝ]]ˆ]
HOˆXÝÜÝ‹[žWN‚ˆY]šX×Ü›ÝÜÎˆ\ÝÙXÝÜÝ‹[žWWHH×Bˆ™YXÝ[Û—Ü›ÝÜÎˆ\ÝÙXÝÜÝ‹[žWWHH×BˆÙYYÙXYÛ›ÜÝXÜÎˆ\ÝÙXÝÜÝ‹[žWWHH×Bˆš]ÈHˆ›ÜˆÙYY[ˆYØXÞK”ÑQQÎ‚ˆ˜Z[—Ü]\ÝÜ]HX\YÜÙYYBˆ˜Z[ˆHYØXÞKœ™XYÜZ\œÊ˜Z[—Ü]
Bˆ\ÝHYØXÞKœ™XYÜZ\œÊ\ÝÜ]
BˆWÝ˜Z[ˆHœ˜\Ø\œ˜^JÜ›ÝË\™Ù]›Üˆ›ÝÈ[ˆ˜Z[—K\O[œ™›Ø]
BˆWÝ\ÝHœ˜\Ø\œ˜^JÜ›ÝË\™Ù]›Üˆ›ÝÈ[ˆ\ÝK\O[œ™›Ø]
Bˆ˜]×Ý˜Z[—Ù™X]\™\Ë˜]×Ý\ÝÙ™X]\™\ÈHYØXÞK™™X]\™WÝX›\Ê˜Z[‹\Ý[X›ÙWÜÙ\]Y[˜Ù\Ëš\\×ÜÙ\]Y[˜Ù\ÊBˆ˜Z[—Ø[X›ÙY\ÈHÛÜY
Ü›ÝË˜[X›ÙH›Üˆ›ÝÈ[ˆ˜Z[ŸJBˆ\ÝØ[X›ÙY\ÈHÛÜY
Ü›ÝË˜[X›ÙH›Üˆ›ÝÈ[ˆ\ÝJBˆ˜Z[—Ýš\\Ù\ÈHÛÜY
Ü›ÝËš\\È›Üˆ›ÝÈ[ˆ˜Z[ŸJBˆ\ÝÝš\\Ù\ÈHÛÜY
Ü›ÝËš\\È›Üˆ›ÝÈ[ˆ\ÝJBˆ[X›ÙWÝ˜Z[‹[X›ÙWÝ\ÝHYØXÞKœÙ[XÝYÙ[X™Y
[X›ÙWÜÙ\]Y[˜Ù\Ë˜Z[—Ø[X›ÙY\Ë\ÝØ[X›ÙY\Ë˜[X›ÙHŠBˆš\\×Ý˜Z[‹š\\×Ý\ÝHYØXÞKœÙ[XÝYÙ[X™Y
š\\×ÜÙ\]Y[˜Ù\Ë˜Z[—Ýš\\Ù\Ë\ÝÝš\\Ù\Ëš\\ÈŠBˆ›Ú[Ý˜Z[ˆHYØXÞKš›Ú[
˜Z[‹[X›ÙWÝ˜Z[‹š\\×Ý˜Z[ŠBˆ›Ú[Ý\ÝHYØXÞKš›Ú[
\Ý[X›ÙWÝ\Ýš\\×Ý\Ý
Bˆ˜]×Û[Ù[HšYÙJ[OLLŒš]Ú[\˜Ù\UYKÛÛ™\H›Ü\ˆ‹ÛLYKN
Bˆ˜]×Û[Ù[™š]
˜]×Ý˜Z[—Ù™X]\™\ËWÝ˜Z[ŠBˆ˜]×Ý˜Z[ˆHœ˜\Ø\œ˜^J˜]×Û[Ù[œ™YXÝ
˜]×Ý˜Z[—Ù™X]\™\ÊK\O[œ™›Ø]
Bˆ˜]×Ý\ÝHœ˜\Ø\œ˜^J˜]×Û[Ù[œ™YXÝ
˜]×Ý\ÝÙ™X]\™\ÊK\O[œ™›Ø]
Bˆš]È
ÏHBˆ˜Z[—ÛÜ\ÝÛÜÜÙXYÈHÜÙÛÛ˜[^™WÙ™X]\™\Ê›Ú[Ý˜Z[‹›Ú[Ý\Ý˜]×Ý˜Z[‹˜]×Ý\Ý
Bˆ™\ÚYX[Ý\™Ù]HWÝ˜Z[ˆH˜]×Ý˜Z[‚ˆ[X›ÙWÙ^Ù\ÜË[X›ÙWÛ›Ý™[HH[]WÛ›Ý™[J[X›ÙWÝ˜Z[‹[X›ÙWÝ\Ý
Bˆš\\×Ù^Ù\ÜËš\\×Û›Ý™[HH[]WÛ›Ý™[Jš\\×Ý˜Z[‹š\\×Ý\Ý
Bˆ›Ý™[WÙØ]HH›Ý×Û›Ý™[WÙØ]J\Ý[X›ÙWÙ^Ù\ÜËš\\×Ù^Ù\ÜÊBˆ™YXÝ[ÛœÎˆXÝÜÝ‹œ›™\œ˜^WHHÔ‘Q‘T‘SÑNˆ˜]×Ý\ÝBˆØ[™Y]WÙXYÛ›ÜÝXÜÎˆ\ÝÙXÝÜÝ‹[žWWHH×Bˆ›ÜˆÛÛ™šYÈ[ˆÓÓ‘’QÔÎ‚ˆÙZYÚÈHÚ[ÝÙZYÚÊWÝ˜Z[‹›Ø]
ÛÛ™šYÖÈœÚ[Ú\ÙWÝZ[Û][\Y\ˆ—JK›Ø]
YØXÞK”‘TÒTÕS•
K›Ø]
YØXÞK”ÕS•
JBˆÛÜœ™XÝ[Û‹XYÛ›ÜÝXÜÈHš]Ü™\ÚYX[ØØ[™Y]Jˆ˜Z[—ÛÜˆ™\ÚYX[Ý\™Ù]ˆ\ÝÛÜˆÙZYÚËˆ›Ý™[WÙØ]Kˆ\WÛ›Ý™[WÙØ]OX›ÛÛ
ÛÛ™šYÖÈ››Ý™[WÙØ]H—JKˆ
Bˆ™YXÝ[ÛœÖÜÝŠÛÛ™šYÖÈ›˜[YH—JWHH˜]×Ý\Ý
ÈÛÜœ™XÝ[Û‚ˆXYÛ›ÜÝXÜË\]JÂˆ›[Ù[ŽˆÝŠÛÛ™šYÖÈ›˜[YH—JKˆœÚ[Ú\ÙWÝZ[Û][\Y\ˆŽˆ›Ø]
ÛÛ™šYÖÈœÚ[Ú\ÙWÝZ[Û][\Y\ˆ—JKˆ››Ý™[WÙØ]WÛZ[š[][HŽˆ›Ø]
œ›Z[Š›Ý™[WÙØ]JJKˆ››Ý™[WÙØ]WÛYYX[ˆŽˆ›Ø]
œ›YYX[Š›Ý™[WÙØ]JJKˆ››Ý™[WÙØ]WÛX^[][HŽˆ›Ø]
œ›X^
›Ý™[WÙØ]JJKˆJBˆØ[™Y]WÙXYÛ›ÜÝXÜË˜\[™
XYÛ›ÜÝXÜÊBˆš]È
ÏHBˆ›Üˆ[Ù[™YXÝ[Ûˆ[ˆ™YXÝ[ÛœËš][\Ê
N‚ˆY]šX×Ü›ÝÜË™^[™
YØXÞK™]˜[Ü›ÝÜÊÙYY[Ù[WÝ\Ý™YXÝ[ÛŠJBˆ™YXÝ[Û—Ü›ÝÜË™^[™
ÂˆœÙYYŽˆÙYYˆ›[Ù[Žˆ[Ù[ˆ˜[X›ÙHŽˆ›ÝË˜[X›ÙKˆ™˜[Z[WÚYŽˆ›ÝË™˜[Z[WÚYˆš\\ÈŽˆ›ÝËš\\Ëˆ]Û™YØ]]™WÛ—ÚXÍLŽˆ›Ü›X]
›ÝË\™Ù]‹ŒMÙÈŠKˆœ™YXÝ[Û—Û™YØ]]™WÛ—ÚXÍLŽˆ›Ü›X]
›Ø]
˜[YJK‹ŒMÙÈŠKˆH›Üˆ›ÝË˜[YH[ˆš\
\Ý™YXÝ[Û‹ÝšXÝUYJJBˆÙYYÙXYÛ›ÜÝXÜË˜\[™
ÂˆœÙYYŽˆÙYYˆ˜Z[—Ù[šY\ÈŽˆ[Š˜Z[ŠKˆ\ÝÙ[šY\ÈŽˆ[Š\Ý
Kˆ›ÜÙÛÛ˜[^˜][ÛˆŽˆÜÙXYËˆ˜[X›ÙWÛ›Ý™[HŽˆ[X›ÙWÛ›Ý™[Kˆš\\×Û›Ý™[HŽˆš\\×Û›Ý™[Kˆ˜Ø[™Y]WÛ[Ù[ÈŽˆØ[™Y]WÙXYÛ›ÜÝXÜËˆJBˆ[›Ú[Ý˜Z[‹›Ú[Ý\Ý˜Z[—ÛÜ\ÝÛÜˆYˆš]ÈOHVPÕQÕÕSÑ’UÎ‚ˆ˜Z\ÙH\ÜÙ\[Û‘\œ›ÜŠˆ™š]YÙ]Z\ÛX]ÚˆÙš]ßHŠBˆÝ[[X\šY\ËÙ[XÝYHÝ[[X\š^™JY]šX×Ü›ÝÜË\JYØXÞK”ÑQQÊJBˆYØXÞKÜš]WØÜÝŠÝ]]È˜Ø[Xœ˜][Û—ÛÜÙÛÛ˜[ÛY]šXÜ×ÝŒK˜ÜÝˆ‹Y]šX×Ü›ÝÜÊBˆYØXÞKÜš]WØÜÝŠÝ]]È˜Ø[Xœ˜][Û—ÛÜÙÛÛ˜[ØØ[™Y]WÜÝ[[X\žWÝŒK˜ÜÝˆ‹Ý[[X\šY\ÊBˆYØXÞKÜš]WØÜÝŠÝ]]È˜Ø[Xœ˜][Û—ÛÜÙÛÛ˜[Ü™YXÝ[Ûœ×Üš]˜]WÝŒK˜ÜÝˆ‹™YXÝ[Û—Ü›ÝÜÊBˆ™\Ý[HÂˆœØÚ[XWÝ™\œÚ[ÛˆŽˆŒKŒ‹ˆœÝYÙHŽˆ™]™[ÜY[[Û›H˜]Ë[Ù™œÙ]Ø[Xœ˜][Û‹[ÜÙÛÛ˜[ÛÛ™][Û˜[™\ÚYX[[Ù[‹ˆ™]™[ÜY[ÜÙYYÈŽˆ\Ý
YØXÞK”ÑQQÊKˆœ™Y™\™[˜ÙWÛ[Ù[Žˆ‘Q‘T‘SÑKˆ˜Ø[™Y]WÛ[Ù[ÈŽˆ\Ý
ÐS‘QUTÊKˆ™[YÚX›WÛ[Ù[ÈŽˆÜ›ÝÖÈ›[Ù[—H›Üˆ›ÝÈ[ˆÝ[[X\šY\ÈYˆ›ÝÖÈ™[YÚX›H—WKˆœÙ[XÝYÛ[Ù[ŽˆÙ[XÝYˆœÙ[XÝ[Û—ÜÝ]\ÈŽˆ™[YÚX›WÛ[Ù[ÜÙ[XÝYˆYˆÙ[XÝY[ÙH››×Û[Ù[Ü\ÜÙY‹ˆ›[Ù[Ùš]ÈŽˆš]Ëˆ™š]ØYÙ]ŽˆÈœ™Y™\™[˜ÙWÙš]×Ü\—ÜÙYYŽˆK˜Ø[™Y]WÙš]×Ü\—ÜÙYYŽˆÝ[Ùš]ÈŽˆVPÕQÕÕSÑ’UË˜Y\]™WÙ^[œÚ[Û—Ø[ÝÙYŽˆKˆœÙYYÙXYÛ›ÜÝXÜÈŽˆÙYYÙXYÛ›ÜÝXÜËˆœÙYYŒÌWÝ\™Ù]×Ü™XYŽˆˆ›˜]\™WÝ\™Ù]×Ü™XYŽˆˆœÙYYÌ—Ý\™Ù]×Ü™XYŽˆˆœš]˜]WØž]\×Ü™XYŽˆˆ˜ÛÛœÝ[YYØÛÛ™š\›X][Û—Ü™]\ÙHŽˆ˜[ÙKˆ˜]]ÛX]X×ØÛÛ™š\›X][Û—Ý\ÙHŽˆ˜[ÙKˆBˆ
Ý]]È˜Ø[Xœ˜][Û—ÛÜÙÛÛ˜[ÜÝ[[X\žWÝŒKšœÛÛˆŠKÜš]WÝ^
œÛÛ‹™[\Ê™\Ý[[™[L‹ÛÜÚÙ^\ÏUYJH
È—ˆ‹[˜ÛÙ[™ÏH]‹NŠBˆ™]\›ˆ™\Ý[‚‚™YˆÙ[—Ý\Ý

HOˆXÝÜÝ‹[žWN‚ˆ›™ÈHœœ˜[™ÛK™Y˜][Ü›™ÊŒŠBˆ˜]×Ý˜Z[ˆH›™Ë››Ü›X[
Ú^™OM
Bˆ˜]×Ý\ÝH›™Ë››Ü›X[
Ú^™OLLJBˆÝ˜Z[ˆH›™Ë››Ü›X[
Ú^™OJVPÕQÕÒQ
JBˆÝ\ÝH›™Ë››Ü›X[
Ú^™OJLKVPÕQÕÒQ
JBˆÝ˜Z[–Î‹HH‹Œ
ÈËŒ
ˆ˜]×Ý˜Z[‚ˆÝ\ÝÎ‹HH‹Œ
ÈËŒ
ˆ˜]×Ý\Ýˆ˜Z[—ÛÜ\ÝÛÜXYÛ›ÜÝXÜÈHÜÙÛÛ˜[^™WÙ™X]\™\ÊÝ˜Z[‹Ý\Ý˜]×Ý˜Z[‹˜]×Ý\Ý
BˆYˆ˜Z[—ÛÜœÚ\HOHÝ˜Z[‹œÚ\HÜˆ\ÝÛÜœÚ\HOHÝ\ÝœÚ\N‚ˆ˜Z\ÙH\ÜÙ\[Û‘\œ›ÜŠ›ÜÙÛÛ˜[^˜][ÛˆÚ\HšYŠBˆ˜Z[—ÛX\HÙˆÚ_HŽˆ›™Ë››Ü›X[
Ú^™OM
H›ÜˆH[ˆ˜[™ÙJJ_Bˆ\ÝÛX\HÙˆœÞÚ_HŽˆ›™Ë››Ü›X[
Ú^™OM
H›ÜˆH[ˆ˜[™ÙJÊ_Bˆ^Ù\ÜË›Ý™[HH[]WÛ›Ý™[J˜Z[—ÛX\\ÝÛX\
BˆÛ\ÜÈ›ÝÎ‚ˆYˆ×Ú[š]×ÊÙ[‹[X›ÙNˆÝ‹š\\ÎˆÝŠNˆÙ[‹˜[X›ÙKÙ[‹š\\ÈH[X›ÙKš\\Âˆ›ÝÜÈHÔ›ÝÊœÌ‹œÌHŠK›ÝÊœÌˆ‹œÌŠWBˆØ]HH›Ý×Û›Ý™[WÙØ]J›ÝÜË^Ù\ÜË^Ù\ÜÊBˆYˆ›Ýœ˜[

Ø]Hˆ
H	ˆ
Ø]HHJJN‚ˆ˜Z\ÙH\ÜÙ\[Û‘\œ›ÜŠ››Ý™[HØ]H˜[™ÙHšYŠBˆ™]\›ˆÈ›ÜÙÛÛ˜[^˜][ÛˆŽˆXYÛ›ÜÝXÜË››Ý™[HŽˆ›Ý™[K™Ø]HŽˆØ]KÛ\Ý

K˜Ø[™Y]WØÛÝ[Žˆ[ŠÐS‘QUTÊK™š]ØYÙ]ŽˆVPÕQÕÕSÑ’UßB‚‚™YˆXZ[Š
HOˆ[‚ˆ\œÙ\ˆH\™Ü\œÙK\™Ý[Y[\œÙ\Š\ØÜš\[ÛW×ÙØ××ÊBˆ\œÙ\‹˜YØ\™Ý[Y[
‹K[YØXÞH‹\OT]
Bˆ\œÙ\‹˜YØ\™Ý[Y[
‹KY]H‹\OT]
Bˆ\œÙ\‹˜YØ\™Ý[Y[
‹KY˜[Z[Y\È‹\OT]
Bˆ\œÙ\‹˜YØ\™Ý[Y[
‹K[Ý]]‹\OT]
Bˆ\œÙ\‹˜YØ\™Ý[Y[
‹K\Ù[‹]\Ý‹XÝ[ÛHœÝÜ™WÝYHŠBˆ\™ÜÈH\œÙ\‹œ\œÙWØ\™ÜÊ
BˆYˆ\™ÜËœÙ[—Ý\Ý‚ˆš[
œÛÛ‹™[\ÊÙ[—Ý\Ý

KÛÜÚÙ^\ÏUYJJBˆ™]\›ˆˆYˆ›Ý[

\™ÜË›YØXÞK\™ÜË™]K\™ÜË™˜[Z[Y\Ë\™ÜË›Ý]]
JN‚ˆ\œÙ\‹™\œ›ÜŠ‹K[YØXÞKKY]KKY˜[Z[Y\È[™K[Ý]]\™H™\]Z\™Y›Üˆ^XÝ][ÛˆŠBˆYØXÞHHØYÛYØXÞJ\™ÜË›YØXÞJBˆ\™ÜË›Ý]]›ZÙ\Š\™[ÏUYK^\ÝÛÚÏUYJBˆ˜[Z[WØžKÈHYØXÞK›ØYÙ˜[Z[WÛX\
\™ÜË™˜[Z[Y\ÊBˆ[X›ÙWÜÙ\]Y[˜Ù\ÈHYØXÞKœ™\ÛÛ™WØ[X›ÙY\Ê\™ÜË™]K˜[Z[WØžJBˆYÙ\ÈHYØXÞK˜Z[ÙYÙ\Ê\™ÜË™]HÈ˜\ÜØ^WÌŒ‹LËLK‹˜[Z[WØžK\™ÜË›Ý]]
BˆÜ]ÈHYØXÞK™œ™Y^™WÜÜ]ÊYÙ\Ë˜[Z[WØžK\™ÜË›Ý]]
BˆX]š^X]š^Ø[X›ÙY\ËX]š^Ýš\\Ù\Ëš\\×ÜÙ\]Y[˜Ù\ÈHYØXÞK˜Z[ÛX]š^
ˆ\™ÜË™]HÈ˜\ÜØ^WÌŒ‹LËLK‹ˆ\™ÜË™]HÈš\œÙ\\×ØXWÓ×ÌŒ‹LËLK™˜\ÝH‹ˆ\™ÜË›Ý]]ˆ
BˆX\YHYØXÞK›X\ÜÜ]ÊÜ]ËX]š^X]š^Ø[X›ÙY\ËX]š^Ýš\\Ù\Ë\™ÜË›Ý]]
Bˆ™\Ý[H[—Û[Ù[ÊYØXÞKX\Y[X›ÙWÜÙ\]Y[˜Ù\Ëš\\×ÜÙ\]Y[˜Ù\Ë\™ÜË›Ý]]
Bˆ™XÙZ\HÂˆœØÚ[XWÝ™\œÚ[ÛˆŽˆŒKŒ‹ˆœX›X×ØÛÛ\]HŽˆYKˆœš]˜]WØž]\×Ü™XYŽˆˆ˜ÛÛ™š\›X][Û—Ý\™Ù]×Ü™XYŽˆˆ›[Ù[Ùš]ÈŽˆVPÕQÕÕSÑ’UËˆœ™\Ý[Žˆ™\Ý[ˆœ[›™\—ÜÚLMˆŽˆÚLMŠ]
×Ùš[W×ÊJKˆ›YØXÞWÜ^[ØYÜÚLMˆŽˆÚLMŠ\™ÜË›YØXÞJKˆ™˜[Z[WÙÜ˜\ÜÚLMˆŽˆÚLMŠ\™ÜË™˜[Z[Y\ÊKˆ™š[\ÈŽˆÜ]›˜[YNˆÈ˜ž]\ÈŽˆ]œÝ]

KœÝÜÚ^™KœÚLMˆŽˆÚLMŠ]
_H›Üˆ][ˆÛÜY
\™ÜË›Ý]]š]\™\Š
JHYˆ]š\×Ùš[J
_KˆBˆ
\™ÜË›Ý]]È”P“P×ÓÔ•ÑÓÓSÔ‘SVWÔ‘PÑRTšœÛÛˆŠKÜš]WÝ^
œÛÛ‹™[\Ê™XÙZ\[™[L‹ÛÜÚÙ^\ÏUYJH
È—ˆ‹[˜ÛÙ[™ÏH]‹NŠBˆš[
œÛÛ‹™[\ÊÈ›[Ù[Ùš]ÈŽˆVPÕQÕÕSÑ’UËœÙ[XÝ[Û—ÜÝ]\ÈŽˆ™\Ý[ÈœÙ[XÝ[Û—ÜÝ]\È—KœÙ[XÝYÛ[Ù[Žˆ™\Ý[ÈœÙ[XÝYÛ[Ù[—_KÛÜÚÙ^\ÏUYJJBˆ™]\›ˆ‚‚šYˆ×Û˜[YW×ÈOH—×ÛXZ[—×ÈŽ‚ˆ˜Z\ÙHÞ\Ý[Q^]
XZ[Š
JB