"""Endgame tiebreak graft: steer low-material, even-material positions toward the
bare-Kernel rule's outcome (more central Kernel wins, Magenta wins equal rings).

When own and enemy non-Kernel counts are equal and their total is <= MAXP:
    value' = (1 - A) * value + A * tanh(K * ((ring_own - ring_enemy) + (0.5 if Magenta else -0.5)))
ring = Chebyshev ring index from the edge (0 outer .. 3 centre). The policy is untouched.

usage: endgame_graft.py base.onnx out.onnx [A=0.6] [K=1.5] [MAXP=2]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "grafts"))
from obuild import RING, graft  # noqa: E402

A, K, MAXP = 0.6, 1.5, 2.0
for kv in sys.argv[3:]:
    k, v = kv.split("=")
    globals()[k] = float(v)


def build(g, obs, net_policy, net_value):
    counts = g.sum_hw(obs)                                              # (B,16)
    w_own = np.zeros((16, 1), np.float32)
    w_own[0:5, 0] = 1.0
    w_en = np.zeros((16, 1), np.float32)
    w_en[6:11, 0] = 1.0
    n_own = g.reshape(g.op("MatMul", counts, g.const(w_own)), -1)       # (B,)
    n_en = g.reshape(g.op("MatMul", counts, g.const(w_en)), -1)
    ring = g.const(RING.reshape(1, 1, 8, 8))
    r_own = g.reshape(g.sum_hw(g.mul(g.planes(obs, 5, 6), ring)), -1)   # (B,)
    r_en = g.reshape(g.sum_hw(g.mul(g.planes(obs, 11, 12), ring)), -1)
    magenta = g.reshape(g.max_axes(g.planes(obs, 15, 16), (1, 2, 3)), -1)  # 1 if Magenta to move
    tb = g.add(g.sub(r_own, r_en), g.sub(magenta, g.f(0.5)))
    target = g.op("Tanh", g.scale(tb, K))
    even = g.op("Cast", g.op("Equal", n_own, n_en), to=1)
    low = g.op("Cast", g.op("LessOrEqual", g.add(n_own, n_en), g.f(MAXP)), to=1)
    alpha = g.scale(g.mul(even, low), A)                                # (B,)
    val = g.add(net_value, g.mul(alpha, g.sub(target, net_value)))
    return net_policy, val


if __name__ == "__main__":
    graft(sys.argv[1], build, sys.argv[2])
    print("saved", sys.argv[2])
