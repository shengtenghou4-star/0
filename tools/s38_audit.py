from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('.')
PIPELINE_SHA256 = '077150c5846941ee38966f905cad8a1f4a662c9f476bc30e867626a11b754848'
DATA_SHA256 = '144126ee9a49d311c3393deea434e1a0963d55de35318e25d98d48f9c175250a'
SOURCES = ('REAL', 'TIME_REVERSED', 'PHASE_SCRAMBLED', 'CHANNEL_BLOCK_PERMUTED')
CONTROLS = SOURCES[1:]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


p1 = json.loads((ROOT / 's38_output_primary/s38_training_result.json').read_text())
p2 = json.loads((ROOT / 's38_output_rerun/s38_training_result.json').read_text())
r1 = json.loads((ROOT / 's38_output_primary/s38_run_receipt.json').read_text())
r2 = json.loads((ROOT / 's38_output_rerun/s38_run_receipt.json').read_text())
source = json.loads((ROOT / 's38_artifacts/source_manifest.json').read_text())
nonaccess = json.loads((ROOT / 's38_artifacts/holdout_nonaccess_receipt.json').read_text())

assert sha(ROOT / 'tools/s38_pipeline.py') == PIPELINE_SHA256
assert source['target']['bytes'] == 348_444_164
assert source['target']['sha256'] == DATA_SHA256
assert source['development_url_present'] is False
assert source['sealed_holdout_url_present'] is False
assert all(nonaccess[k] is False for k in (
    'sealed_holdout_requested', 'sealed_holdout_listed', 'sealed_holdout_downloaded',
    'sealed_holdout_opened', 'development_source_accessed'
))
assert sha(ROOT / 's38_output_primary/s38_training_result.json') == sha(ROOT / 's38_output_rerun/s38_training_result.json')
assert sha(ROOT / 's38_output_primary/s38_run_receipt.json') == sha(ROOT / 's38_output_rerun/s38_run_receipt.json')
assert p1 == p2 and r1 == r2
assert tuple(p1['summaries']) == SOURCES

# Independent gate arithmetic.
summary = p1['summaries']
cc = p1['control_comparisons']
real1, real5, real10 = summary['REAL']['1'], summary['REAL']['5'], summary['REAL']['10']
max_real = max(
    p1['outer'][rid]['sources']['REAL']['horizons'][h]['model_to_M1_ratio']
    for rid in p1['outer'] for h in ('1', '5', '10')
)
recalc = {
    'all_real_predictions_finite': all(
        p1['outer'][rid]['sources']['REAL']['horizons'][h]['model']['finite']
        for rid in p1['outer'] for h in ('1', '5', '10')
    ),
    'real_one_frame_to_M1_at_most_0_98': real1['recording_balanced_mean_model_to_M1_ratio'] <= 0.98,
    'real_one_frame_to_best_control_at_most_0_98': cc['1']['real_to_best_control_ratio'] <= 0.98,
    'real_five_frame_to_M1_at_most_0_98': real5['recording_balanced_mean_model_to_M1_ratio'] <= 0.98,
    'real_ten_frame_to_M1_at_most_0_98': real10['recording_balanced_mean_model_to_M1_ratio'] <= 0.98,
    'real_five_frame_to_best_control_at_most_0_99': cc['5']['real_to_best_control_ratio'] <= 0.99,
    'real_ten_frame_to_best_control_at_most_0_99': cc['10']['real_to_best_control_ratio'] <= 0.99,
    'real_five_frame_at_least_three_of_four_beats_M1': real5['recordings_model_beats_M1'] >= 3,
    'real_ten_frame_at_least_three_of_four_beats_M1': real10['recordings_model_beats_M1'] >= 3,
    'no_real_outer_any_horizon_above_1_10': max_real <= 1.10,
}
assert recalc == p1['checks']
assert sum(recalc.values()) == p1['passed_checks']
assert all(recalc.values()) == p1['pass']
assert abs(max_real - p1['maximum_real_outer_any_horizon_model_to_M1_ratio']) <= 1e-15
for h in ('1', '5', '10'):
    rows = cc[h]['control_mean_rmse']
    best = min(rows, key=lambda x: (rows[x], x))
    assert best == cc[h]['best_control_source']
    assert best in CONTROLS
    assert abs(cc[h]['real_to_best_control_ratio'] - cc[h]['real_mean_rmse'] / rows[best]) <= 1e-15

# Control seeds and equal-treatment receipts.
spec = importlib.util.spec_from_file_location('s38_audit_target', ROOT / 'tools/s38_pipeline.py')
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
for rid, receipt in p1['record_receipts'].items():
    assert set(receipt['controls']) == set(SOURCES)
    for src in SOURCES:
        got = receipt['controls'][src]
        assert got['seed'] == mod.stable_seed(rid, src)
        assert got['finite'] is True
        assert got['offline_negative_control'] is (src != 'REAL')
for rid, outer in p1['outer'].items():
    for src in SOURCES:
        assert set(outer['sources'][src]['selected_propagation']) == {'5', '10'}
        for h in ('5', '10'):
            sel = outer['sources'][src]['selected_propagation'][h]['selection_receipt']
            assert sel['inner_max_model_to_M1'] <= sel['inner_safety_ceiling'] + 1e-15

# Strict future-target sentinel for the REAL causal prediction machinery.
n = 240
frames = np.arange(n, dtype=int)
X = np.column_stack([np.sin(frames / 9), np.cos(frames / 17), frames / n])
residual = 0.08 * np.sin((frames + 4) / 15)
target = residual.copy()
persistence = np.zeros(n)
warm = frames < 90
scored = frames >= 120
ev1 = mod.EventSet(frames, frames + 1, X, residual, target, persistence, warm, scored)
pm1, s1 = mod.prediction_map_with_warm(ev1, 10.0, 0.999)
residual2 = residual.copy()
target2 = target.copy()
residual2[frames >= 190] += 500.0
target2[frames >= 190] += 500.0
ev2 = mod.EventSet(frames, frames + 1, X, residual2, target2, persistence, warm, scored)
pm2, s2 = mod.prediction_map_with_warm(ev2, 10.0, 0.999)
early = s1.current_frames < 190
assert np.array_equal(s1.current_frames, s2.current_frames)
assert np.allclose(s1.pred_residual[early], s2.pred_residual[early], rtol=0, atol=0)
for f in sorted(set(pm1) & set(pm2)):
    if f < 190:
        assert pm1[f] == pm2[f]

assert p1['development_evaluation_executed'] is False
assert p1['development_source_accessed_by_this_run'] is False
assert p1['AML32_chip_requested'] is False
assert p1['AML32_chip_listed'] is False
assert p1['AML32_chip_downloaded'] is False
assert p1['AML32_chip_opened'] is False

independent = {
    'schema': 'bio-001-s38-independent-audit-v1',
    'status': 'PASS',
    'pipeline_sha256': sha(ROOT / 'tools/s38_pipeline.py'),
    'primary_result_sha256': sha(ROOT / 's38_output_primary/s38_training_result.json'),
    'rerun_result_sha256': sha(ROOT / 's38_output_rerun/s38_training_result.json'),
    'exact_rerun': True,
    'gate_recalculation': 'PASS',
    'negative_control_seed_and_equal_treatment_audit': 'PASS',
    'real_future_target_causality_sentinel': 'PASS',
    'holdout_nonaccess': 'PASS',
    'passed_checks': p1['passed_checks'],
    'total_checks': p1['total_checks'],
    'training_pass': p1['pass'],
    'verdict': p1['verdict'],
    'CeRSI_v2_delta': 0.0,
}
(ROOT / 's38_independent_audit.json').write_text(json.dumps(independent, indent=2, sort_keys=True, allow_nan=False))
compact = {
    'status': r1['status'],
    'verdict': p1['verdict'],
    'checks': p1['checks'],
    'summaries': p1['summaries'],
    'control_comparisons': p1['control_comparisons'],
    'max_real_ratio': p1['maximum_real_outer_any_horizon_model_to_M1_ratio'],
    'real_outer': {rid: p1['outer'][rid]['sources']['REAL'] for rid in p1['outer']},
    'holdout_opened': False,
}
print('BIO001_S38_RESULT=' + json.dumps(compact, sort_keys=True, allow_nan=False))
