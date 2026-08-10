from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path

import h5py
import numpy as np
import requests
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
RAW = OUT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

FILES = [
    ("train", "2023-03-30-01-cudata.h5", 4707115),
    ("train", "2023-06-21-01-cudata.h5", 4707119),
    ("train", "2023-06-23-08-cudata.h5", 4707120),
    ("train", "2023-06-29-01-cudata.h5", 4707121),
    ("train", "2023-06-29-13-cudata.h5", 4707095),
    ("train", "2023-07-14-08-cudata.h5", 4707097),
    ("development", "2024-05-28-10-cudata.h5", 4707114),
]
FORBIDDEN = {"2024-06-18-12-cudata.h5"}
ALLOWED_NAMES = {x[1] for x in FILES}
ALLOWED_IDS = {x[2] for x in FILES}
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
HORIZONS = [1, 5, 10, 20]
STRIDE = 25
NAMES = [
    "velocity", "angular_velocity", "sin_head_angle", "cos_head_angle",
    "pumping", "distance_from_copper_center", "worm_curvature",
    "neural_population_mean", "neural_population_std",
    "neural_fraction_above_plus_1z", "neural_fraction_below_minus_1z",
]
PRIMARY = [0, 1, 4, 5, 6, 7, 8]
SIG = b"\x89HDF\r\n\x1a\n"


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def acquire(role, name, file_id):
    if name in FORBIDDEN or name not in ALLOWED_NAMES or file_id not in ALLOWED_IDS:
        raise RuntimeError("allowlist rejection")
    path = RAW / name
    url = f"https://datadryad.org/api/v2/files/{file_id}/download"
    err = None
    for _ in range(5):
        try:
            with requests.get(url, stream=True, timeout=(30, 300), allow_redirects=True,
                              headers={"User-Agent": "BIO-001-S26-fixed-case/1.0"}) as r:
                r.raise_for_status()
                with path.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
            if path.read_bytes()[:8] != SIG:
                raise RuntimeError("not HDF5")
            return path
        except Exception as e:
            err = e
            path.unlink(missing_ok=True)
    raise RuntimeError(f"download failed for {name}: {err}")


def vec(x, label):
    a = np.asarray(x, dtype=float).squeeze()
    if a.ndim != 1:
        raise ValueError(f"{label} shape {a.shape}")
    ok = np.isfinite(a)
    if not ok.any():
        raise ValueError(f"{label} all nonfinite")
    if not ok.all():
        i = np.arange(a.size)
        a = np.interp(i, i[ok], a[ok])
    return a


def read(handle, key):
    if key not in handle:
        keys = []
        handle.visit(keys.append)
        raise KeyError(f"missing {key}; keys={keys}")
    return np.asarray(handle[key])


def load(path, role):
    with h5py.File(path, "r") as h:
        v = vec(read(h, "behavior/velocity"), "velocity")
        av = vec(read(h, "behavior/angular_velocity"), "angular_velocity")
        ha = vec(read(h, "behavior/head_angle"), "head_angle")
        pump = vec(read(h, "behavior/pumping"), "pumping")
        dist = vec(read(h, "behavior/dist_from_cuCenter"), "distance")
        curv = vec(read(h, "behavior/worm_curvature"), "curvature")
        tr = np.asarray(read(h, "gcamp/trace_array"), dtype=float)
    frame_hint = min(map(len, [v, av, ha, pump, dist, curv]))
    if tr.ndim != 2:
        raise ValueError(f"trace shape {tr.shape}")
    if abs(tr.shape[0] - frame_hint) < abs(tr.shape[1] - frame_hint):
        tr = tr.T
    n = min(frame_hint, tr.shape[1])
    kept = [vec(row[:n], f"neuron_{i}") for i, row in enumerate(tr) if np.isfinite(row[:n]).any()]
    if not kept or n < 100:
        raise ValueError("insufficient aligned neural data")
    tr = np.vstack(kept)
    state = np.column_stack([
        v[:n], av[:n], np.sin(ha[:n]), np.cos(ha[:n]), pump[:n], dist[:n], curv[:n],
        tr.mean(0), tr.std(0), (tr > 1).mean(0), (tr < -1).mean(0),
    ])
    if state.shape != (n, 11) or not np.isfinite(state).all():
        raise ValueError(f"bad state {state.shape}")
    return {"role": role, "name": path.name, "state": state, "frames": n,
            "neurons": tr.shape[0], "bytes": path.stat().st_size, "sha256": sha(path)}


def phi(cur, prev):
    cur = np.asarray(cur); prev = np.asarray(prev)
    one = cur.ndim == 1
    if one:
        cur, prev = cur[None, :], prev[None, :]
    neural = cur[:, 7:11]
    x = np.concatenate([cur, cur * cur, cur[:, [0]] * neural,
                        cur[:, [5]] * neural, cur - prev], axis=1)
    return x[0] if one else x


def transitions(animals, mean, scale):
    xs, ys = [], []
    for a in animals:
        z = (a["state"] - mean) / scale
        xs.append(phi(z[1:-1], z[:-2])); ys.append(z[2:])
    return np.vstack(xs), np.vstack(ys)


def stabilize(model):
    c = np.asarray(model.coef_, dtype=float)
    j = c[:, :11] + c[:, 30:41]
    before = float(np.max(np.abs(np.linalg.eigvals(j))))
    factor = 1.0
    if before > 0.995:
        factor = 0.995 / before
        model.coef_ = c * factor
    c2 = np.asarray(model.coef_, dtype=float)
    after = float(np.max(np.abs(np.linalg.eigvals(c2[:, :11] + c2[:, 30:41]))))
    return {"before": before, "factor": factor, "after": after}


def rollout(model, prev, cur, steps):
    for _ in range(steps):
        nxt = model.predict(phi(cur, prev)[None, :])[0]
        if not np.isfinite(nxt).all():
            return np.full(11, np.nan)
        prev, cur = cur, np.clip(nxt, -20, 20)
    return cur


def evaluate(model, animal, mean, scale):
    z = (animal["state"] - mean) / scale
    all_metrics = {}
    for horizon in HORIZONS:
        pred, obs, base = [], [], []
        for anchor in range(1, animal["frames"] - horizon, STRIDE):
            pred.append(rollout(model, z[anchor - 1], z[anchor], horizon) * scale + mean)
            obs.append(animal["state"][anchor + horizon]); base.append(animal["state"][anchor])
        pred, obs, base = map(np.asarray, (pred, obs, base))
        rm = np.sqrt(np.mean((pred - obs) ** 2, axis=0))
        rb = np.sqrt(np.mean((base - obs) ** 2, axis=0))
        ratio = np.divide(rm, rb, out=np.full(11, np.inf), where=rb > 1e-12)
        all_metrics[str(horizon)] = {
            "anchors": len(pred),
            "model_rmse": dict(zip(NAMES, rm.tolist())),
            "persistence_rmse": dict(zip(NAMES, rb.tolist())),
            "ratio": dict(zip(NAMES, ratio.tolist())),
            "mean_nrmse_all_11": float(np.mean(rm / scale)),
            "mean_ratio_primary_7": float(np.mean(ratio[PRIMARY])),
            "improved_primary_7": int(np.sum(ratio[PRIMARY] < 1)),
        }
    return all_metrics


def main():
    if FORBIDDEN & ALLOWED_NAMES:
        raise RuntimeError("holdout leak")
    animals, manifest = [], []
    for role, name, fid in FILES:
        path = acquire(role, name, fid)
        a = load(path, role); animals.append(a)
        manifest.append({k: a[k] for k in ["role", "name", "frames", "neurons", "bytes", "sha256"]} | {"dryad_file_id": fid})
    dump("source_manifest.json", {"doi": "10.5061/dryad.w9ghx3g4v", "files": manifest,
                                  "holdout_requested": False, "forbidden": sorted(FORBIDDEN)})
    train = [a for a in animals if a["role"] == "train"]
    dev = next(a for a in animals if a["role"] == "development")
    pooled = np.vstack([a["state"] for a in train])
    mean, scale = pooled.mean(0), pooled.std(0)
    scale[scale < 1e-8] = 1
    X, Y = transitions(train, mean, scale)
    candidates = []
    models = {}
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(X, Y)
        stability = stabilize(model)
        metrics = evaluate(model, dev, mean, scale)
        score = float(np.mean([metrics[str(h)]["mean_nrmse_all_11"] for h in HORIZONS]))
        candidates.append({"alpha": alpha, "selection_score": score,
                           "stability": stability, "metrics": metrics})
        models[alpha] = (model, stability, metrics)
    candidates.sort(key=lambda x: (x["selection_score"], x["alpha"]))
    alpha = candidates[0]["alpha"]
    model, stability, metrics = models[alpha]
    h10 = metrics["10"]
    ratio = h10["mean_ratio_primary_7"]
    improved = h10["improved_primary_7"]
    passed = bool(ratio <= 0.95 and improved >= 4)
    model_json = {
        "schema": "bio-001-s26-frozen-model-v1", "alpha": alpha, "state_names": NAMES,
        "coef": np.asarray(model.coef_).tolist(), "intercept": np.asarray(model.intercept_).tolist(),
        "mean": mean.tolist(), "scale": scale.tolist(), "stability": stability,
        "holdout_requested": False,
    }
    dump("frozen_model.json", model_json)
    dump("development_results.json", {
        "schema": "bio-001-s26-development-v1", "selected_alpha": alpha,
        "selection_rule": "minimum mean nRMSE across 11 channels and horizons 1,5,10,20",
        "candidates": candidates, "selected_metrics": metrics,
        "primary_gate": {"horizon": 10, "mean_ratio_primary_7": ratio,
                         "relative_improvement": 1 - ratio,
                         "improved_endpoints": improved, "pass": passed},
        "holdout_requested": False, "CeRSI_v2_delta": 0.0,
    })
    dump("run_receipt.json", {
        "status": "PASS_DEVELOPMENT_GATE" if passed else "FAIL_DEVELOPMENT_GATE",
        "selected_alpha": alpha, "h10_mean_ratio": ratio, "h10_improved": improved,
        "raw_files": len(list(RAW.glob("*.h5"))), "holdout_requested": False,
        "model_sha256": sha(OUT / "frozen_model.json"),
        "results_sha256": sha(OUT / "development_results.json"),
        "CeRSI_v2_delta": 0.0,
    })
    print(json.dumps({"selected_alpha": alpha, "h10_ratio": ratio,
                      "h10_improved": improved, "pass": passed,
                      "holdout_requested": False}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        dump("failure_receipt.json", {"type": type(e).__name__, "error": str(e),
                                      "traceback": traceback.format_exc(),
                                      "holdout_requested": False})
        raise
