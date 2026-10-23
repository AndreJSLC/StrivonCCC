"""Material + centrality + kernel-threat evaluator as ONNX (no training needed).

usage: build2.py out.onnx [scale kc pc wt wd]
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

args = [float(a) for a in sys.argv[2:]]
SCALE, KC, PC, WT, WD = (args + [1.0, 0.3, 0.03, 3.0, 0.6][len(args):])[:5]

# planes 0-5 own Ping, Daemon, Exploit, Trace, Ice, Kernel; 6-11 enemy
VAL = np.array([0.15, 0.45, 0.25, 0.30, 0.50, 3.0], dtype=np.float32)
PING, DAEMON, EXPLOIT, TRACE, ICE, KERNEL = range(6)
R = 4  # conv radius


def offsets_to_kernel(offs):
    k = np.zeros((2 * R + 1, 2 * R + 1), dtype=np.float32)
    for dy, dx in offs:
        if abs(dy) <= R and abs(dx) <= R:
            k[R - dy, R - dx] = 1.0
    return k


def compose(a, b):
    return {(y1 + y2, x1 + x2) for y1, x1 in a for y2, x2 in b}


king = {(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if (y, x) != (0, 0)}
knight = {(y, x) for y in (-2, -1, 1, 2) for x in (-2, -1, 1, 2) if abs(y) != abs(x)}
pdiag = {(s * d, t * d) for d in (1, 2) for s in (-1, 1) for t in (-1, 1)}
porth = {(0, 0)} | {(s * d, 0) for d in (1, 2) for s in (-1, 1)} | {(0, s * d) for d in (1, 2) for s in (-1, 1)}
ice = {(y, x) for y in range(-2, 3) for x in range(-2, 3) if (y, x) != (0, 0) and (y == 0 or x == 0 or abs(y) == abs(x))}
daemon = king | {(s * d, 0) for d in (2, 3) for s in (-1, 1)} | {(0, s * d) for d in (2, 3) for s in (-1, 1)}
trace = {(s * d, t * d) for d in (2, 3) for s in (-1, 1) for t in (-1, 1)}
step1 = {(0, 0)} | king  # any one-step reposition (generous)

reach1 = {PING: pdiag, DAEMON: daemon, EXPLOIT: knight, TRACE: trace, ICE: ice, KERNEL: king}
reach2 = {
    PING: pdiag | compose(porth, pdiag),
    DAEMON: daemon | compose(step1, daemon),
    EXPLOIT: knight | compose(knight, knight),
    TRACE: trace | compose(step1, trace),
    ICE: ice | compose(ice, ice),
    KERNEL: {(y, x) for y in range(-2, 3) for x in range(-2, 3) if (y, x) != (0, 0)},
}

Wc = np.zeros((4, 16, 2 * R + 1, 2 * R + 1), dtype=np.float32)
for p in range(6):
    Wc[0, p] = offsets_to_kernel(reach1[p])       # own reach, 1 action
    Wc[1, p] = offsets_to_kernel(reach2[p])       # own reach, 2 actions
    Wc[2, 6 + p] = offsets_to_kernel(reach1[p])   # enemy reach, 1 action
    Wc[3, 6 + p] = offsets_to_kernel(reach2[p])   # enemy reach, 2 actions
Wk = np.zeros((4, 16, 1, 1), dtype=np.float32)
Wk[0, 6 + KERNEL] = Wk[1, 6 + KERNEL] = 1.0
Wk[2, KERNEL] = Wk[3, KERNEL] = 1.0

w = np.zeros((16, 1), dtype=np.float32)
w[0:6, 0] = VAL * SCALE
w[6:12, 0] = -VAL * SCALE

ring = np.array([[min(r, c, 7 - r, 7 - c) for c in range(8)] for r in range(8)], dtype=np.float32)
sp = np.zeros((16, 8, 8), dtype=np.float32)
for p in range(6):
    sp[p] = PC * ring
    sp[6 + p] = -PC * ring
sp[KERNEL] = KC * ring
sp[6 + KERNEL] = -KC * ring

# ind2 = 1 when the side to move has 2 actions: counts[13] = 64 * actions_left/2
Wi = np.zeros((16, 1), dtype=np.float32)
Wi[13, 0] = 1.0 / 32.0
Bi = np.array([-1.0], dtype=np.float32)
# features f = [t1, t2, d1, d2]; base uses t1 and danger, ind2 switches t1 -> t2
Wf_base = np.array([[WT], [0.0], [-WD * 0.5], [-WD]], dtype=np.float32)
Wf_ind = np.array([[-WT], [WT], [0.0], [0.0]], dtype=np.float32)

f32 = lambda a: numpy_helper.from_array(np.asarray(a, dtype=np.float32), None)
inits = [
    numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes"),
    numpy_helper.from_array(np.array([1], dtype=np.int64), "ax1"),
    numpy_helper.from_array(np.array([-1], dtype=np.int64), "flat"),
    numpy_helper.from_array(sp[None], "W_sp"),
    numpy_helper.from_array(w, "W_v"),
    numpy_helper.from_array(Wc, "W_c"),
    numpy_helper.from_array(Wk, "W_k"),
    numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one"),
    numpy_helper.from_array(Wi, "W_i"),
    numpy_helper.from_array(Bi, "B_i"),
    numpy_helper.from_array(Wf_base, "W_fb"),
    numpy_helper.from_array(Wf_ind, "W_fi"),
    numpy_helper.from_array(np.zeros((16, 4096), dtype=np.float32), "W_p"),
]
nodes = [
    helper.make_node("ReduceSum", ["observation", "axes"], ["counts"], keepdims=0),
    helper.make_node("Gemm", ["counts", "W_v"], ["mat"]),
    helper.make_node("Mul", ["observation", "W_sp"], ["xs"]),
    helper.make_node("ReduceSum", ["xs", "axes"], ["xs2"], keepdims=0),
    helper.make_node("ReduceSum", ["xs2", "ax1"], ["pos"], keepdims=1),
    helper.make_node("Conv", ["observation", "W_c"], ["att"], pads=[R, R, R, R]),
    helper.make_node("Min", ["att", "one"], ["attc"]),
    helper.make_node("Conv", ["observation", "W_k"], ["kp"]),
    helper.make_node("Mul", ["attc", "kp"], ["hit"]),
    helper.make_node("ReduceSum", ["hit", "axes"], ["feat"], keepdims=0),
    helper.make_node("Gemm", ["counts", "W_i", "B_i"], ["ind2"]),
    helper.make_node("Gemm", ["feat", "W_fb"], ["fb"]),
    helper.make_node("Gemm", ["feat", "W_fi"], ["fi"]),
    helper.make_node("Mul", ["fi", "ind2"], ["fi2"]),
    helper.make_node("Sum", ["mat", "pos", "fb", "fi2"], ["v1"]),
    helper.make_node("Tanh", ["v1"], ["v2"]),
    helper.make_node("Reshape", ["v2", "flat"], ["value"]),
    helper.make_node("Gemm", ["counts", "W_p"], ["policy"]),
]
graph = helper.make_graph(
    nodes, "heuristic",
    [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 16, 8, 8])],
    [helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 4096]),
     helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, sys.argv[1] if len(sys.argv) > 1 else "model2.onnx")
print("saved")
