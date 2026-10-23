"""API example only: graft a no-op component onto a trained net.

usage: python -I grafts/example_graft.py models/level4_final.onnx out.onnx

Shows the tensor layouts: obs (B,16,8,8); net_policy (B,4096) = (B, from 64, to 64) flattened;
net_value (B,) in [-1, 1] for the side to move.
"""
import sys

import numpy as np

from obuild import RING, graft


def build(g, obs, net_policy, net_value):
    # a per-to-square score map (B,1,64) broadcast over from-squares: here centrality * 0 (no-op)
    occ_enemy = g.flat64(g.op("ReduceSum", g.planes(obs, 6, 12), g.ints(1), keepdims=1))  # (B,1,64)
    to_bonus = g.add(g.scale(occ_enemy, 0.0), g.const((0.0 * RING).reshape(1, 1, 64)))
    pol3 = g.reshape(net_policy, -1, 64, 64)                    # [b, from, to]
    pol = g.reshape(g.add(pol3, to_bonus), -1, 4096)
    actions_left_two = g.op("ReduceMax", g.planes(obs, 13, 14), axes=[1, 2, 3], keepdims=0)  # (B,) 1.0 or 0.5
    val = g.add(net_value, g.scale(actions_left_two, 0.0))      # (B,)
    return pol, val


if __name__ == "__main__":
    graft(sys.argv[1], build, sys.argv[2])
    print("saved", sys.argv[2])
