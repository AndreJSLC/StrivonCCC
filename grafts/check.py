"""Light local check of a candidate ONNX. Cheap enough to run on a laptop.

usage: python -I grafts/check.py model.onnx [--base models/level4_final.onnx]
       [--n 2000] [--skip-validate]

1. The server's structural/allow-list check (submission_check.validate_model_file_portable).
2. One-thread ORT: batch 1/8/16 shapes, finite outputs, value in [-1, 1], batch-8 latency.
3. With --base: how much the candidate changes the base on real positions (argmax agreement over
   legal actions, mean |dvalue|, KL(base || cand) over legal actions) and the biggest value changes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "env"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from ref import load_positions  # noqa: E402


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run(sess, obs, bs=8):
    P, V = [], []
    for i in range(0, len(obs), bs):
        p, v = sess.run(None, {"observation": obs[i:i + bs]})
        P.append(p); V.append(v)
    return np.concatenate(P), np.concatenate(V)


def masked_softmax(logits, legal):
    x = np.where(legal, logits.astype(np.float64), -np.inf)
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--base", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--skip-validate", action="store_true")
    a = ap.parse_args()
    obs, legal, z, _ = load_positions()
    obs, legal, z = obs[:a.n], legal[:a.n], z[:a.n]

    if not a.skip_validate:
        from colony_one.submission_check import validate_model_file_portable
        t = validate_model_file_portable(a.model)
        print(f"[1] structural/allow-list check ACCEPTED (dry run {t:.2f}s)")

    s = session(a.model)
    for bs in (1, 16):
        p, v = s.run(None, {"observation": obs[:bs]})
        assert p.shape == (bs, 4096) and v.shape == (bs,), (p.shape, v.shape)
    t0 = time.perf_counter()
    reps = 0
    while time.perf_counter() - t0 < 2.0:
        s.run(None, {"observation": obs[reps % 200 * 8:(reps % 200) * 8 + 8]})
        reps += 1
    lat = (time.perf_counter() - t0) / reps * 1000
    p, v = run(s, obs)
    assert np.isfinite(p[legal]).all() and np.isfinite(v).all(), "non-finite outputs"
    assert (np.abs(v) <= 1.0 + 1e-6).all(), f"value out of range: {v.min()} {v.max()}"
    corr = float(np.corrcoef(v, z)[0, 1])
    print(f"[2] shapes ok, finite, value in [{v.min():.3f}, {v.max():.3f}], corr(value, game result) "
          f"{corr:.3f}, batch-8 latency {lat:.2f} ms (1 thread)")

    if a.base:
        bp, bv = run(session(a.base), obs)
        pb, pc = masked_softmax(bp, legal), masked_softmax(p, legal)
        agree = float((np.where(legal, bp, -np.inf).argmax(1) == np.where(legal, p, -np.inf).argmax(1)).mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            kl = np.where(legal, pb * (np.log(pb + 1e-12) - np.log(pc + 1e-12)), 0.0).sum(1)
        dv = v - bv
        bcorr = float(np.corrcoef(bv, z)[0, 1])
        print(f"[3] vs base: argmax agreement {agree:.1%}, mean KL(base||cand) {kl.mean():.4f}, "
              f"mean |dvalue| {np.abs(dv).mean():.4f}, value changed >0.1 on {(np.abs(dv) > 0.1).mean():.1%}; "
              f"corr with result: base {bcorr:.3f} -> cand {corr:.3f}")
        mse_b, mse_c = float(((bv - z) ** 2).mean()), float(((v - z) ** 2).mean())
        print(f"    value MSE vs game result: base {mse_b:.4f} -> cand {mse_c:.4f}")
        top = np.argsort(-np.abs(dv))[:8]
        print("    biggest value changes (pos, base -> cand, result):",
              ", ".join(f"#{i} {bv[i]:+.2f}->{v[i]:+.2f} z={z[i]:+.0f}" for i in top))


if __name__ == "__main__":
    main()
