"""Level 4 heuristic evaluator as ONNX: material, centrality, guard-aware kernel threats, firewall.

usage: build3.py out.onnx [key=value ...]
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

P = dict(scale=1.0, kc=0.3, pc=0.03, wt=3.0, wd=0.6, gres=0.15, fk=1.5, fp=0.5, fu=0.6)
for kv in sys.argv[2:]:
    k, v = kv.split("=")
    P[k] = float(v)

VAL = np.array([0.15, 0.45, 0.25, 0.30, 0.50, 3.0], dtype=np.float32)
PING, DAEMON, EXPLOIT, TRACE, ICE, KERNEL = range(6)
R = 6


def kern(offs, r=R):
    k = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)
    for dy, dx in offs:
        if abs(dy) <= r and abs(dx) <= r:
            k[r - dy, r - dx] = 1.0
    return k


def compose(a, b):
    return {(y1 + y2, x1 + x2) for y1, x1 in a for y2, x2 in b}


def orth(ds):
    return {(s * d, 0) for d in ds for s in (-1, 1)} | {(0, s * d) for d in ds for s in (-1, 1)}


def diag(ds):
    return {(s * d, t * d) for d in ds for s in (-1, 1) for t in (-1, 1)}


king = orth([1]) | diag([1])
knight = {(y, x) for y in (-2, -1, 1, 2) for x in (-2, -1, 1, 2) if abs(y) != abs(x)}
O = {(0, 0)}
attack = {PING: diag([1, 2]), DAEMON: king | orth([2, 3]), EXPLOIT: knight,
          TRACE: diag([2, 3]), ICE: orth([1, 2]) | diag([1, 2]), KERNEL: king}
move = {PING: O | orth([1, 2]), DAEMON: O | orth([1, 2, 3]), EXPLOIT: O | knight,
        TRACE: O | orth([1]), ICE: O | orth([1, 2]) | diag([1, 2]), KERNEL: O | king}
reach1 = attack
reach2 = {p: compose(move[p], attack[p]) for p in range(6)}

# conv A: own reach1, own reach2, enemy reach1, enemy reach2
Wc = np.zeros((4, 16, 2 * R + 1, 2 * R + 1), dtype=np.float32)
for p in range(6):
    Wc[0, p] = kern(reach1[p])
    Wc[1, p] = kern(reach2[p])
    Wc[2, 6 + p] = kern(reach1[p])
    Wc[3, 6 + p] = kern(reach2[p])
# conv G (3x3): adjacent guards. ch0 enemy guards (near enemy K), ch1 own guards
Wg = np.zeros((2, 16, 3, 3), dtype=np.float32)
ring8 = kern(king, 1)
for p in range(5):
    Wg[0, 6 + p] = ring8
    Wg[1, p] = ring8
# 1x1 selectors: kernel planes for threat features
Wk = np.zeros((4, 16, 1, 1), dtype=np.float32)
Wk[0, 11] = Wk[1, 11] = 1.0
Wk[2, 5] = Wk[3, 5] = 1.0
Wk2 = np.zeros((2, 16, 1, 1), dtype=np.float32)
Wk2[0, 11] = 1.0
Wk2[1, 5] = 1.0

w = np.zeros((16, 1), dtype=np.float32)
w[0:6, 0] = VAL * P["scale"]
w[6:12, 0] = -VAL * P["scale"]

ring = np.array([[min(r, c, 7 - r, 7 - c) for c in range(8)] for r in range(8)], dtype=np.float32)
sp = np.zeros((16, 8, 8), dtype=np.float32)
for p in range(6):
    sp[p] = P["pc"] * ring
    sp[6 + p] = -P["pc"] * ring
sp[5] = P["kc"] * ring
sp[11] = -P["kc"] * ring

# firewall: next ring = open squares adjacent to a locked square or on the board edge
edge = (ring == 0).astype(np.float32)
Wl = np.zeros((1, 16, 3, 3), dtype=np.float32)
Wl[0, 12] = 1.0
# weights per piece plane for standing on the next ring (own negative, enemy positive)
Wf = np.zeros((1, 16, 1, 1), dtype=np.float32)
for p in range(5):
    Wf[0, p] = -P["fp"] * VAL[p]
    Wf[0, 6 + p] = P["fp"] * VAL[p]
Wf[0, 5] = -P["fk"]
Wf[0, 11] = P["fk"]
# urgency gate = relu(u - fu) / (1 - fu), u = counts[14] / 64
Wu = np.zeros((16, 1), dtype=np.float32)
Wu[14, 0] = 1.0 / 64.0 / (1.0 - P["fu"])
Bu = np.array([-P["fu"] / (1.0 - P["fu"])], dtype=np.float32)

Wi = np.zeros((16, 1), dtype=np.float32)
Wi[13, 0] = 1.0 / 32.0
Bi = np.array([-1.0], dtype=np.float32)
WT, WD = P["wt"], P["wd"]
Wf_base = np.array([[WT], [0.0], [-WD * 0.5], [-WD]], dtype=np.float32)
Wf_ind = np.array([[-WT], [WT], [0.0], [0.0]], dtype=np.float32)
# guard factor per feature: [enemyGuarded, enemyGuarded, ownGuarded, ownGuarded]
Gsel = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)

T = numpy_helper.from_array
inits = [
    T(np.array([2, 3], dtype=np.int64), "axes"),
    T(np.array([1], dtype=np.int64), "ax1"),
    T(np.array([-1], dtype=np.int64), "flat"),
    T(sp[None], "W_sp"), T(w, "W_v"), T(Wc, "W_c"), T(Wk, "W_k"), T(Wg, "W_g"), T(Wk2, "W_k2"),
    T(np.array(1.0, dtype=np.float32), "one"),
    T(np.array(P["gres"], dtype=np.float32), "gres"),
    T(Gsel, "G_sel"),
    T(Wi, "W_i"), T(Bi, "B_i"), T(Wf_base, "W_fb"), T(Wf_ind, "W_fi"),
    T(edge[None, None], "edge"), T(Wl, "W_l"), T(Wf, "W_f"), T(Wu, "W_u"), T(Bu, "B_u"),
    T(np.zeros((1, 4096), dtype=np.float32), "W_p"),
]
N = helper.make_node
nodes = [
    N("ReduceSum", ["observation", "axes"], ["counts"], keepdims=0),
    N("Gemm", ["counts", "W_v"], ["mat"]),
    N("Mul", ["observation", "W_sp"], ["xs"]),
    N("ReduceSum", ["xs", "axes"], ["xs2"], keepdims=0),
    N("ReduceSum", ["xs2", "ax1"], ["pos"], keepdims=1),
    # threats
    N("Conv", ["observation", "W_c"], ["att"], pads=[R, R, R, R]),
    N("Min", ["att", "one"], ["attc"]),
    N("Conv", ["observation", "W_k"], ["kp"]),
    N("Mul", ["attc", "kp"], ["hit"]),
    N("ReduceSum", ["hit", "axes"], ["feat"], keepdims=0),
    # guards adjacent to each kernel -> guarded in {0,1}; factor = 1 - (1-gres)*guarded
    N("Conv", ["observation", "W_g"], ["gmap"], pads=[1, 1, 1, 1]),
    N("Conv", ["observation", "W_k2"], ["kp2"]),
    N("Mul", ["gmap", "kp2"], ["gk"]),
    N("ReduceSum", ["gk", "axes"], ["gsum"], keepdims=0),
    N("Min", ["gsum", "one"], ["guarded"]),
    N("MatMul", ["guarded", "G_sel"], ["g4"]),
    N("Sub", ["one", "gres"], ["gcoef"]),
    N("Mul", ["g4", "gcoef"], ["g4c"]),
    N("Sub", ["one", "g4c"], ["gfac"]),
    N("Mul", ["feat", "gfac"], ["featg"]),
    N("Gemm", ["counts", "W_i", "B_i"], ["ind2"]),
    N("Gemm", ["featg", "W_fb"], ["fb"]),
    N("Gemm", ["featg", "W_fi"], ["fi"]),
    N("Mul", ["fi", "ind2"], ["fi2"]),
    # firewall
    N("Conv", ["observation", "W_l"], ["lockn"], pads=[1, 1, 1, 1]),
    N("Add", ["lockn", "edge"], ["nr0"]),
    N("Min", ["nr0", "one"], ["nr1"]),
    N("Slice", ["observation", "s12", "e13", "ax1"], ["locked"]),
    N("Sub", ["one", "locked"], ["openm"]),
    N("Mul", ["nr1", "openm"], ["nextring"]),
    N("Mul", ["observation", "nextring"], ["onring"]),
    N("Conv", ["onring", "W_f"], ["fmap"]),
    N("ReduceSum", ["fmap", "axes"], ["fsum"], keepdims=0),
    N("Gemm", ["counts", "W_u", "B_u"], ["u0"]),
    N("Relu", ["u0"], ["ug"]),
    N("Mul", ["fsum", "ug"], ["fire"]),
    N("Sum", ["mat", "pos", "fb", "fi2", "fire"], ["v1"]),
    N("Tanh", ["v1"], ["v2"]),
    N("Reshape", ["v2", "flat"], ["value"]),
    # policy prior: logit[from, to] = pa * enemyValue[to] - pb * ownValue[from] * enemyReach2[to]
    N("Conv", ["observation", "W_ev"], ["evmap"]),
    N("Reshape", ["evmap", "shp_b1_64"], ["ev_to"]),
    N("Conv", ["observation", "W_ov"], ["ovmap"]),
    N("Reshape", ["ovmap", "shp_b64_1"], ["ov_from"]),
    N("Slice", ["attc", "s3", "e4", "ax1"], ["er2"]),
    N("Reshape", ["er2", "shp_b1_64"], ["er_to"]),
    N("Mul", ["ev_to", "ones64"], ["cap"]),
    N("Mul", ["ov_from", "er_to"], ["risk"]),
    N("Sub", ["cap", "risk"], ["pol3"]),
    N("Reshape", ["pol3", "shp_b4096"], ["policy"]),
]
Wev = np.zeros((1, 16, 1, 1), dtype=np.float32)
Wov = np.zeros((1, 16, 1, 1), dtype=np.float32)
for p in range(6):
    Wev[0, 6 + p] = P.get("pa", 2.0) * min(VAL[p], 1.0)
    Wov[0, p] = P.get("pb", 1.0) * min(VAL[p], 1.0)
inits += [
    T(Wev, "W_ev"), T(Wov, "W_ov"),
    T(np.array([-1, 1, 64], dtype=np.int64), "shp_b1_64"),
    T(np.array([-1, 64, 1], dtype=np.int64), "shp_b64_1"),
    T(np.array([-1, 4096], dtype=np.int64), "shp_b4096"),
    T(np.array([3], dtype=np.int64), "s3"), T(np.array([4], dtype=np.int64), "e4"),
    T(np.ones((1, 64, 1), dtype=np.float32), "ones64"),
]
inits = [i for i in inits if i.name != "W_p"]
inits += [T(np.array([12], dtype=np.int64), "s12"), T(np.array([13], dtype=np.int64), "e13")]
graph = helper.make_graph(
    nodes, "heuristic4",
    [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 16, 8, 8])],
    [helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 4096]),
     helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, sys.argv[1])
print("saved", P)
