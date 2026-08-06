from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

root = Path('.')
p1 = json.loads((root/'s36_output_primary/s36_training_result.json').read_text())
p2 = json.loads((root/'s36_output_rerun/s36_training_result.json').read_text())
r1 = json.loads((root/'s36_output_primary/s36_run_receipt.json').read_text())
r2 = json.loads((root/'s36_output_rerun/s36_run_receipt.json').read_text())
source = json.loads((root/'s36_artifacts/source_manifest.json').read_text())
nonaccess = json.loads((root/'s36_artifacts/holdout_nonaccess_receipt.json').read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


assert sha(root/'tools/s36_pipeline.py') == '1ed1225d0c9efc0d6caeb93e733246f3a372f39529c050503726a7af95a69cc2'
assert source['target']['bytes'] == 348_444_164
assert source['target']['sha256'] == '144126ee9a49d311c3393deea434e1a0963d55de35318e25d98d48f9c175250a'
assert source['development_url_present'] is False
assert source['sealed_holdout_url_present'] is False
assert nonaccess['sealed_holdout_requested'] is False
assert nonaccess['sealed_holdout_listed'] is False
assert nonaccess['sealed_holdout_downloaded'] is False
assert nonaccess['sealed_holdout_opened'] is False
assert nonaccess['development_source_accessed'] is False
assert sha(root/'s36_output_primary/s36_training_result.json') == sha(root/'s36_output_rerun/s36_training_result.json')
assert sha(root/'s36_output_primary/s36_run_receipt.json') == sha(root/'s36_output_rerun/s36_run_receipt.json')
assert p1 == p2 and r1 == r2

h10 = p1['horizons']['10']
recalculated = {
    'all_predictions_finite': all(
        rec['horizons'][h]['M1']['finite'] and rec['horizons'][h]['M3']['finite']
        for rec in p1['outer'].values() for h in ('1','5','10')
    ),
    'primary_M3_to_M1_ratio_at_most_0_98': h10['recording_balanced_mean_M3_to_M1_ratio'] <= 0.98,
    'primary_M3_to_persistence_ratio_at_most_0_95': h10['recording_balanced_mean_M3_to_persistence_ratio'] <= 0.95,
    'at_least_two_horizons_M3_beats_both': sum(
        p1['horizons'][h]['recording_balanced_mean_M3_to_M1_ratio'] < 1.0
        and p1['horizons'][h]['recording_balanced_mean_M3_to_persistence_ratio'] < 1.0
        for h in ('1','5','10')
    ) >= 2,
    'at_least_three_of_four_outer_records_M3_beats_M1_primary': sum(
        rec['horizons']['10']['M3_to_M1_ratio'] < 1.0 for rec in p1['outer'].values()
    ) >= 3,
    'no_outer_record_any_horizon_M3_to_M1_above_1_10': max(
        rec['horizons'][h]['M3_to_M1_ratio']
        for rec in p1['outer'].values() for h in ('1','5','10')
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

# Independent causal sentinel. The explicit sys.modules registration is required
# by Python 3.13 dataclass introspection during dynamic import.
spec = importlib.util.spec_from_file_location('s36_audit_target', root/'tools/s36_pipeline.py')
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
n = 100
frames = np.arange(n)
X = np.column_stack([np.sin(frames/7), np.cos(frames/11)])
residual = np.sin((frames+3)/13)*0.1
target = residual.copy()
persistence = np.zeros(n)
warm = frames < 30
scored = frames >= 35
ev = mod.EventSet(frames, frames+1, X, residual, target, persistence, warm, scored)
a = mod.prequential_raw(ev, 10.0, 0.999)
residual2 = residual.copy()
target2 = target.copy()
residual2[frames >= 70] += 1000
target2[frames >= 70] += 1000
ev2 = mod.EventSet(frames, frames+1, X, residual2, target2, persistence, warm, scored)
b = mod.prequential_raw(ev2, 10.0, 0.999)
early = a.current_frames < 70
assert np.array_equal(a.current_frames, b.current_frames)
assert np.allclose(a.pred_residual[early], b.pred_residual[early], rtol=0, atol=0)

audit = {
    'schema': 'bio-001-s36-independent-audit-v1',
    'status': 'PASS',
    'pipeline_sha256': sha(root/'tools/s36_pipeline.py'),
    'primary_result_sha256': sha(root/'s36_output_primary/s36_training_result.json'),
    'rerun_result_sha256': sha(root/'s36_output_rerun/s36_training_result.json'),
    'exact_rerun': True,
    'gate_recalculation': 'PASS',
    'causal_future_target_sentinel': 'PASS',
    'passed_checks': p1['passed_checks'],
    'total_checks': p1['total_checks'],
    'training_pass': p1['pass'],
    'development_executed': False,
    'holdout_nonaccess': 'PASS',
    'CeRSI_v2_delta': 0.0,
}
(root/'s36_independent_audit.json').write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
compact = {
    'status': r1['status'],
    'checks': p1['checks'],
    'horizons': p1['horizons'],
    'max_outer_ratio': p1['maximum_outer_any_horizon_M3_to_M1_ratio'],
    'final_selection': p1['final_selection_using_all_training_records'],
    'outer_primary': {
        rid: rec['horizons']['10'] for rid, rec in p1['outer'].items()
    },
    'holdout_opened': False,
}
print('BIO001_S36_RESULT=' + json.dumps(compact, sort_keys=True, allow_nan=False))
