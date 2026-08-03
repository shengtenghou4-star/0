from __future__ import annotations

import runpy
import sys
from pathlib import Path


SOURCE = Path(__file__).with_name("train_hallinen.py")
text = SOURCE.read_text(encoding="utf-8")

replacements = [
    (
        "    feature_names_behavior: list[str]\n    feature_names_coupled: list[str]\n",
        "    feature_names_behavior: list[str]\n    feature_names_coupled: list[str]\n    original_channels: int\n    retained_channels: int\n",
    ),
    (
        "    prefix = neural[:, :prefix_end]\n    if not np.all(np.isfinite(prefix)):\n        raise ValueError(f\"{path}: nonfinite neural value in PCA prefix\")\n\n    median = np.median(prefix, axis=1)\n",
        "    prefix = neural[:, :prefix_end]\n    original_channels = int(neural.shape[0])\n    prefix_channel_mask = np.all(np.isfinite(prefix), axis=1)\n    retained_channels = int(np.count_nonzero(prefix_channel_mask))\n    retained_fraction = retained_channels / original_channels\n    if retained_channels < 50 or retained_fraction < 0.40:\n        raise ValueError(\n            f\"{path}: retained only {retained_channels}/{original_channels} prefix-finite channels\"\n        )\n    neural = neural[prefix_channel_mask]\n    prefix = neural[:, :prefix_end]\n\n    median = np.median(prefix, axis=1)\n",
    ),
    (
        "        feature_names_behavior=behavior_feature_names,\n        feature_names_coupled=coupled_names,\n    )\n",
        "        feature_names_behavior=behavior_feature_names,\n        feature_names_coupled=coupled_names,\n        original_channels=original_channels,\n        retained_channels=retained_channels,\n    )\n",
    ),
    (
        "                \"coupled_features\": int(record.coupled_x.shape[1]),\n            }\n",
        "                \"coupled_features\": int(record.coupled_x.shape[1]),\n                \"original_channels\": record.original_channels,\n                \"retained_channels\": record.retained_channels,\n                \"retained_fraction\": record.retained_channels / record.original_channels,\n                \"removed_channels\": record.original_channels - record.retained_channels,\n            }\n",
    ),
    (
        "        \"feature_amendment_commit\": \"596318171b14b3998b8fceb46384399e799952c4\",\n",
        "        \"feature_amendment_commit\": \"596318171b14b3998b8fceb46384399e799952c4\",\n        \"missing_neuron_amendment_commit\": \"520308336f63826bfdcbe1d95a2da4a8d667bfd8\",\n",
    ),
]

for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one patch target, found {count}: {old[:80]!r}")
    text = text.replace(old, new)

output_dir = None
for index, argument in enumerate(sys.argv):
    if argument == "--output-dir" and index + 1 < len(sys.argv):
        output_dir = Path(sys.argv[index + 1])
        break
if output_dir is None:
    raise RuntimeError("--output-dir is required")
output_dir.mkdir(parents=True, exist_ok=True)
patched = output_dir / "EXECUTED_TRAIN_HALLINEN.py"
patched.write_text(text, encoding="utf-8")

runpy.run_path(str(patched), run_name="__main__")
