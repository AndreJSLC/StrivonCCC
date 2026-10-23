"""Batch-8 latency of an ONNX model with the referee's runtime settings (1 intra-op thread).

usage: latency.py model.onnx [...]
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "env"))
import numpy as np  # noqa: E402

from colony_one.onnx_runner import OnnxEvaluator  # noqa: E402

for m in sys.argv[1:]:
    ev = OnnxEvaluator(Path(m))
    x = (np.random.default_rng(0).random((8, 16, 8, 8)) < 0.05).astype(np.float32)
    for _ in range(20):
        ev(x)
    t = time.perf_counter()
    for _ in range(200):
        ev(x)
    print(f"{m}: {(time.perf_counter() - t) / 200 * 1000:.2f} ms per batch of 8", flush=True)
