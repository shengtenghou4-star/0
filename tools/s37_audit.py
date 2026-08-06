from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('.')
PIPELINE_SHA256 = '0c5132213b58d0ec3ab241cb4bc020f35c69284f1d9d3fcc7f07fb1354ed0e5d'
DATA_SHA256 = '144126ee9a49d311c3393deea434e1a0963d55de35318e25d98d48f9c175250a'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


p1 = json.loads((ROOT / 's37_output_primary/s37_training_result.json').read_text())
p2 = json.loads((ROOT / 's37_output_rerun/s37_training_result.json').read_text())
r1 = json.loads((ROOT / 's37_output_primary/s37_run_receipt.json').read_text())
r2 = json.loads((ROOT / 's37_output_rerun/s37_run_receipt.json').read_text())
source = json.loads((ROOT / 's37_artifacts/source_manifest.json').read_text())
nonaccess = json.loads((ROOT / 's37_artifacts/holdout_nonaccess_receipt.json').read_text())

assert sha(ROOT / 'tools/s37_pipeline.py') == PIPELINE_SHA256
assert source['target']['bytes'] == 348_444_164
assert source['target']['sha256'] == DATA_SHA256
assert source['development_url_present'] is False
assert source['sealed_holdout_url_present'] is False
assert nonaccess['sealed_holdout_requested'] is False
assert nonaccess['sealed_holdout_listed'] is False
assert nonaccess['sealed_holdout_downloaded'] is False
assert nonaccess['sealed_holdout_opened'] is False
assert nonaccess['development_source_accessed'] is False
assert sha(ROOT / 's37_output_primary/s37_training_result.json') == sha(ROOT / 's37_output_rerun/s37_training_result.json')
assert sha(ROOT / 's37_output_primary/s37_run_receipt.json') == sha(ROOT / 's37_output_rerun/s37_run_receipt.json')
assert p1 == p2 and r1 == r2

h1 = p1['horizons']['1']
h10 = p1['horizons']['10']
recalculated = {
    'all_predictions_finite': all(
        rec['horizons'][h]['M1']['finite'] and rec['horizons'][h]['M4']['finite']
        for rec in p1['outer'].values() for h in ('1', '5', '10')
    ),
    'one_frame_M4_to_M1_ratio_at_most_0_98': h1['recording_balanced_mean_M4_to_M1_ratio'] <= 0.98,
    'one_frame_all_four_outer_records_M4_beats_M1': sum(
        rec['horizons']['1']['M4_to_M1_ratio'] < 1.0 for rec in p1['outer'].values()
    ) == 4,
    'primary_M4_to_M1_ratio_at_most_0_98': h10['recording_balanced_mean_M4_to_M1_ratio'] <= 0.98,
    'primary_M4_to_persistence_ratio_at_most_0_95': h10['recording_balanced_mean_M4_to_persistence_ratio'] <= 0.95,
    'at_least_three_of_four_outer_records_M4_beats_M1_primary': sum(
        rec['horizons']['10']['M4_to_M1_ratio'] < 1.0 for rec in p1['outer'].values()
    ) >= 3,
    'at_least_two_horizons_M4_beats_both': sum(
        p1['horizons'][h]['recording_balanced_mean_M4_to_M1_ratio'] < 1.0
        and p1['horizons'][h]['recording_balanced_mean_M4_to_persistence_ratio'] < 1.0
        for h in ('1', '5', '10')
    ) >= 2,
    'no_outer_record_any_horizon_M4_to_M1_above_1_10': max(
        rec['horizons'][h]['M4_to_M1_ratio']
        for rec in p1['outer'].values() for h in ('1', '5', '10')
    ) <= 1.10,
}
assert recalculated == p1['checks']
assert sum(recalculated.values()) == p1['passed_checks']
assert all(recalculated.values()) == p1['pass']
assert p1['development_evaluation_executed'] is False
assert p1['development_source_accessed_by_this_run'] is False
assert p1['AML32_chip_requested'] is False
assert p1['AML32_chip_listed'] is False
assert p1['AML32_chip_downloaded'] is False
assert p1['AML32_chip_opened'] is False

# Selection and retirement receipts are structurally complete.
for rid, rec in p1['outer'].items():
    assert sorted(rec['selected_M4_by_horizon']) == ['1', '10', '5']
    for h in ('1', '5', '10'):
        cfg = rec['selected_M4_by_horizon'][h]['config']
        sel = rec['selected_M4_by_horizon'][h]['selection_receipt']
        assert sel['inner_max_M4_to_M1'] <= sel['inner_safety_ceiling'] + 1e-15
        guard = rec['horizons'][h]['guard']
        assert guard['retirement_is_irreversible'] is True
        if cfg.get('disabled', False):
            assert rec['horizons'][h]['M4_to_M1_ratio'] == 1.0
            assert guard['mean_neural_weight'] == 0.0

# Dynamic import for independent causal and irreversible-retirement sentinels.
spec = importlib.util.spec_from_file_location('s37_audit_target', ROOT / 'tools/s37_pipeline.py')
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

n = 120
frames = np.arange(n)
X = np.column_stack([np.sin(frames / 7), np.cos(frames / 11)])
residual = np.sin((frames + 3) / 13) * 0.1
target = residual.copy()
persistence = np.zeros(n)
warm = frames < 35
scored = frames >= 40
ev = mod.EventSet(frames, frames + 1, X, residual, target, persistence, warm, scored)
a1 = mod.prequential_raw(ev, 10.0, 0.999)
a2 = mod.prequential_raw(ev, 100.0, 0.995)
a4, aw, _ad = mod.regret_guard_stream(a1, a2, 15, 1.0, 0.05, 2.0)
residual2 = residual.copy()
target2 = target.copy()
residual2[frames >= 85] += 1000.0
target2[frames >= 85] += 1000.0
ev2 = mod.EventSet(frames, frames + 1, X, residual2, target2, persistence, warm, scored)
b1 = mod.prequential_raw(ev2, 10.0, 0.999)
b2 = mod.prequential_raw(ev2, 100.0, 0.995)
b4, bw, _bd = mod.regret_guard_stream(b1, b2, 15, 1.0, 0.05, 2.0)
early = a1.current_frames < 85
assert np.array_equal(a1.current_frames, b1.current_frames)
assert np.allclose(a1.pred_residual[early], b1.pred_residual[early], rtol=0, atol=0)
assert np.allclose(a2.pred_residual[early], b2.pred_residual[early], rtol=0, atol=0)
assert np.allclose(a4[early], b4[early], rtol=0, atol=0)
assert np.allclose(aw[early], bw[early], rtol=0, atol=0)

# Crafted stream: neural branch first wins, then fails, and must retire forever.
cf = np.arange(100, dtype=int)
tf = cf + 1
y = np.zeros(100)
persist = np.zeros(100)
m1 = mod.PredictionStream(cf, tf, y, persist, np.ones(100))
m2_pred = np.concatenate([np.zeros(35), np.full(65, 3.0)])
m2 = mod.PredictionStream(cf, tf, y, persist, m2_pred)
_p4, weights, diag = mod.regret_guard_stream(m1, m2, 10, 1.0, 0.01, 4.0)
assert diag['retired'] is True
rf = int(diag['retirement_frame'])
assert np.all(weights[cf >= rf] == 0.0)

independent = {
    'schema': 'bio-001-s37-independent-audit-v1',
    'status': 'PASS',
    'pipeline_sha256': sha(ROOT / 'tools/s37_pipeline.py'),
    'primary_result_sha256': sha(ROOT / 's37_output_primary/s37_training_result.json'),
    'rerun_result_sha256': sha(ROOT / 's37_output_rerun/s37_training_result.json'),
    'exact_rerun': True,
    'gate_recalculation': 'PASS',
    'horizon_specific_selection_receipts': 'PASS',
    'causal_future_target_sentinel': 'PASS',
    'irreversible_retirement_sentinel': 'PASS',
    'passed_checks': p1['passed_checks'],
    'total_checks': p1['total_checks'],
    'training_pass': p1['pass'],
    'development_executed': False,
    'holdout_nonaccess': 'PASS',
    'CeRSI_v2_delta': 0.0,
}
(ROOT / 's37_independent_audit.json').write_text(json.dumps(independent, indent=2, sort_keys=True, allow_nan=False))
compact = {
    'status': r1['status'],
    'checks': p1['checks'],
    'horizons': p1['horizons'],
    'max_outer_ratio': p1['maximum_outer_any_horizon_M4_to_M1_ratio'],
    'max_outer_ratio_at_most_1_05': p1['maximum_outer_any_horizon_M4_to_M1_at_most_1_05'],
    'final_selection': p1['final_selection_using_all_training_records'],
    'outer_primary': {rid: rec['horizons']['10'] for rid, rec in p1['outer'].items()},
    'holdout_opened': False,
}
print('BIO001_S37_RESULT=' + json.dumps(compact, sort_keys=True, allow_nan=False))
