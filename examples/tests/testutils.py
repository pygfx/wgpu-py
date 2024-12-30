import os
from pathlib import Path

import wgpu


ROOT = Path(__file__).parent.parent.parent  # repo root
examples_dir = ROOT / "examples"
screenshots_dir = examples_dir / "screenshots"
diffs_dir = screenshots_dir / "diffs"


def find_examples(query=None, negative_query=None, return_stems=False):
    result = []
    for example_path in examples_dir.glob("*.py"):
        example_code = example_path.read_text()
        query_match = query is None or query in example_code
        negative_query_match = (
            negative_query is None or negative_query not in example_code
        )
        if query_match and negative_query_match:
            result.append(example_path)
    result = list(sorted(result))
    if return_stems:
        result = [r.stem for r in result]
    return result


def get_default_adapter_summary():
    """Get description of adapter, or None when no adapter is available."""
    try:
        adapter = wgpu.gpu.request_adapter_sync()
    except RuntimeError:
        return None  # lib not available, or no adapter on this system
    return adapter.summary


adapter_summary = get_default_adapter_summary()
can_use_wgpu_lib = bool(adapter_summary)
is_ci = bool(os.getenv("CI", None))
