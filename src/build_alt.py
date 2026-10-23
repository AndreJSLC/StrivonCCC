"""Hand-built Level 4 evaluator (value + non-uniform policy prior), no training.

usage: build_alt.py out.onnx [key=value ...]

Value (side-to-move view) = tanh(material + centrality + hanging-piece threats
  + kernel-attack threats (bodyguard-aware) + firewall danger terms).
Policy logit[f, t] = A[f] + sum_p own_p[f] * S_p[t]:  captures, kernel escape,
  firewall evacuation, avoiding attacked squares.
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

P = dict(
    scale=1.0, kc=0.3, pc=0.03,
    wt=3.0, wd=0.6,            # kernel reach (own -> enemy kernel, enemy -> own kernel)
    floor=0.3,                 # share of kernel-threat that remains when the kernel has a guard
    hang_en=0.5, hang_own=0.15,  # value of pieces that are attacked (enemy / own)
    fw_mat=0.8,                # material at risk on the ring the firewall locks next
    fw_k_own=2.0, fw_k_en=2.0, fw_k_en_done=3.0,  # kernel on the next-locked ring
    fw_pow=2.0,                # urgency exponent
    # policy
    pol=1.0,                   # global logit scale
    p_cap=1.0, p_kcap=5.0, p_att=0.6, p_cen=0.25,
    p_fw_to=2.0, p_fw_from=1.5, p_kesc=2.0, p_flee=0.6,
)
for a in sys.argv[2:]:
    k, v = a.split("=")
    assert k in P, k
    P[k] = float(v)

R = 4
PING, DAEMON, EXPLOIT, TRACE, ICE, KERNEL = range(6)
VAL = np.array([0.15, 0.45, 0.25, 0.30, 0.50, 3.0], dtype=np.float32)
PVAL = np.array([0.15, 0.45, 0.25, 0.30, 0.50, 0.0], dtype=np.float32)  # material at stake (no kernel)


def kern(offs):
    k = np.zeros((2 * R + 1, 2 * R + 1), dtype=np.float32)
    for dy, dx in offs:
        if abs(dy) <= R and abs(dx) <= R:
            k[R - dy, R - dx] = 1.0
    return k


def compose(a, b):
    return {(y1 + y2, x1 + x2) for y1, x1 in a for y2, x2 in b}


king = {(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if (y, x) != (0, 0)}
knight = {(y, x) for y in (-2, -1, 1, 2) for x in (-2, -1, 1, 2) if abs(y) != abs(x)}
orth = lambda n: {(s * d, 0) for d in range(1, n + 1) for s in (-1, 1)} | {(0, s * d) for d in range(1, n + 1) for s in (-1, 1)}
diag = lambda lo, hi: {(s * d, t * d) for d in range(lo, hi + 1) for s in (-1, 1) for t in (-1, 1)}
ice = {(y, x) for y in range(-2, 3) for x in range(-2, 3) if (y, x) != (0, 0) and (y == 0 or x == 0 or abs(y) == abs(x))}
stay = {(0, 0)}
# squares each piece can hit in ONE action (capture reach)
reach1 = {
    PING: diag(1, 2),
    DAEMON: king | {(s * d, 0) for d in (2, 3) for s in (-1, 1)} | {(0, s * d) for d in (2, 3) for s in (-1, 1)},
    EXPLOIT: knight,
    TRACE: diag(2, 3),
    ICE: ice,
    KERNEL: king,
}
# ... and in up to two actions (move then capture, approximated geometrically)
reach2 = {
    PING: reach1[PING] | compose(stay | orth(2), reach1[PING]),
    DAEMON: reach1[DAEMON] | compose(stay | orth(3), reach1[DAEMON]),
    EXPLOIT: knight | compose(knight, knight),
    TRACE: reach1[TRACE] | compose(stay | orth(1), reach1[TRACE]),
    ICE: ice | compose(ice, ice),
    KERNEL: {(y, x) for y in range(-2, 3) for x in range(-2, 3) if (y, x) != (0, 0)},
}

ring = np.array([[min(r, c, 7 - r, 7 - c) for c in range(8)] for r in range(8)], dtype=np.float32)
ring_oh = np.zeros((4, 64), dtype=np.float32)  # which ring each square is on
for r in range(8):
    for c in range(8):
        ring_oh[int(ring[r, c]), r * 8 + c] = 1.0

inits, nodes = [], []
_n = [0]


def const(arr, name=None):
    name = name or f"c{_n[0]}"
    _n[0] += 1
    arr = np.asarray(arr)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    inits.append(numpy_helper.from_array(arr, name))
    return name


def op(kind, ins, out=None, **kw):
    out = out or f"t{_n[0]}"
    _n[0] += 1
    nodes.append(helper.make_node(kind, ins, [out], **kw))
    return out


def shape_c(*s):
    return const(np.array(s, dtype=np.int64))


def gemm(x, W, b=None):
    return op("Gemm", [x, const(np.asarray(W, dtype=np.float32))] + ([const(np.asarray(b, dtype=np.float32))] if b is not None else []))


def mul(a, b):
    return op("Mul", [a, b])


def add(*xs):
    return op("Sum", list(xs))


def scale(x, s):
    return mul(x, const(np.array(s, dtype=np.float32)))


OBS = "observation"
counts = op("ReduceSum", [OBS, const(np.array([2, 3], dtype=np.int64))], keepdims=0)  # (B,16)


def col(idx_weights, bias=0.0):
    """Linear function of counts -> (B,1)."""
    W = np.zeros((16, 1), dtype=np.float32)
    for i, w in idx_weights.items():
        W[i, 0] = w
    return gemm(counts, W, np.array([bias], dtype=np.float32))


u = col({14: 1 / 64})                    # firewall urgency in [1/8, 1]
tm = col({15: 1 / 64})                   # 1 = Magenta to move
two = col({13: 1 / 32}, -1.0)            # 1 when 2 actions left
upow = u
for _ in range(int(P["fw_pow"]) - 1):
    upow = mul(upow, u)

# which ring locks next: L_j = ring j already locked (nested); N_j = L_{j-1} - L_j
thr = np.array([28, 48, 60, 64], dtype=np.float32)
Wl = np.zeros((16, 4), dtype=np.float32)
Wl[12, :] = 1.0
L = op("Clip", [gemm(counts, Wl, 1.0 - thr), const(np.array(0.0, dtype=np.float32)), const(np.array(1.0, dtype=np.float32))])
Msh = np.zeros((4, 4), dtype=np.float32)
for j in range(4):
    Msh[j, j] = -1.0
    if j >= 1:
        Msh[j - 1, j] = 1.0
Nnext = gemm(L, Msh, np.array([1.0, 0, 0, 0], dtype=np.float32))     # (B,4) one-hot next ring
Dmap = op("Reshape", [gemm(Nnext, ring_oh), shape_c(-1, 1, 8, 8)])    # 1 on squares the firewall locks next

# ---- reach / attack maps
Wc = np.zeros((4, 16, 2 * R + 1, 2 * R + 1), dtype=np.float32)
for p in range(6):
    Wc[0, p] = kern(reach1[p])
    Wc[1, p] = kern(reach2[p])
    Wc[2, 6 + p] = kern(reach1[p])
    Wc[3, 6 + p] = kern(reach2[p])
att = op("Min", [op("Conv", [OBS, const(Wc)], pads=[R, R, R, R]), const(np.array(1.0, dtype=np.float32))])  # (B,4,8,8)
Wk = np.zeros((4, 16, 1, 1), dtype=np.float32)
Wk[0, 6 + KERNEL] = Wk[1, 6 + KERNEL] = 1.0
Wk[2, KERNEL] = Wk[3, KERNEL] = 1.0
kp = op("Conv", [OBS, const(Wk)])                                     # kernel locations per feature
hit = mul(att, kp)
feat = op("ReduceSum", [hit, const(np.array([2, 3], dtype=np.int64))], keepdims=0)  # (B,4): t1 t2 d1 d2

# bodyguard: number of friendly non-kernel pieces adjacent to each kernel
Wg = np.zeros((2, 16, 3, 3), dtype=np.float32)
for p in range(5):
    Wg[0, p] = 1.0
    Wg[0, p, 1, 1] = 0.0
    Wg[1, 6 + p] = 1.0
    Wg[1, 6 + p, 1, 1] = 0.0
gmap = op("Conv", [OBS, const(Wg)], pads=[1, 1, 1, 1])                # (B,2,8,8)
kmask = op("Concat", [op("Slice", [OBS, const(np.array([5], dtype=np.int64)), const(np.array([6], dtype=np.int64)), const(np.array([1], dtype=np.int64))]),
                       op("Slice", [OBS, const(np.array([11], dtype=np.int64)), const(np.array([12], dtype=np.int64)), const(np.array([1], dtype=np.int64))])], axis=1)
gcnt = op("ReduceSum", [mul(gmap, kmask), const(np.array([2, 3], dtype=np.int64))], keepdims=0)  # (B,2): own, enemy guards
noguard = op("Relu", [gemm(gcnt, -np.eye(2, dtype=np.float32), np.ones(2, dtype=np.float32))])   # 1 when no guard
modk = add(scale(noguard, 1 - P["floor"]), const(np.full((1, 2), P["floor"], dtype=np.float32)))  # (B,2)
sel_own = np.array([[1.0, 0.0]], dtype=np.float32).T
sel_en = np.array([[0.0, 1.0]], dtype=np.float32).T
mod_own = gemm(modk, sel_own)   # (B,1): own kernel's guard modifier (enemy attacks on it)
mod_en = gemm(modk, sel_en)

# ---- value
SC = P["scale"]
w = np.zeros((16, 1), dtype=np.float32)
w[0:6, 0] = VAL * SC
w[6:12, 0] = -VAL * SC
mat = gemm(counts, w)

sp = np.zeros((16, 8, 8), dtype=np.float32)
for p in range(6):
    sp[p] = P["pc"] * ring
    sp[6 + p] = -P["pc"] * ring
sp[KERNEL] = P["kc"] * ring
sp[6 + KERNEL] = -P["kc"] * ring
pos = op("ReduceSum", [op("ReduceSum", [mul(OBS, const(sp[None])), const(np.array([2, 3], dtype=np.int64))], keepdims=0), const(np.array([1], dtype=np.int64))], keepdims=1)

# kernel-reach threats: [t1, t2, d1, d2]; t uses 1-action reach when 1 action left, 2-action when 2 left
Wf_b = np.array([[P["wt"]], [0.0], [0.0], [0.0]], dtype=np.float32)
Wf_t2 = np.array([[-P["wt"]], [P["wt"]], [0.0], [0.0]], dtype=np.float32)
Wd1 = np.array([[0.0], [0.0], [1.0], [0.0]], dtype=np.float32)
Wd2 = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float32)
threat = add(gemm(feat, Wf_b), mul(gemm(feat, Wf_t2), two))
threat = mul(threat, mod_en)
danger = add(scale(gemm(feat, Wd1), -P["wd"] * 0.5), scale(gemm(feat, Wd2), -P["wd"]))
danger = mul(danger, mod_own)

# hanging pieces (non-kernel): enemy pieces I attack vs. my pieces they attack
a_en1 = op("Slice", [att, const(np.array([0], dtype=np.int64)), const(np.array([1], dtype=np.int64)), const(np.array([1], dtype=np.int64))])
a_en2 = op("Slice", [att, const(np.array([1], dtype=np.int64)), const(np.array([2], dtype=np.int64)), const(np.array([1], dtype=np.int64))])
a_my1 = op("Slice", [att, const(np.array([2], dtype=np.int64)), const(np.array([3], dtype=np.int64)), const(np.array([1], dtype=np.int64))])
a_my2 = op("Slice", [att, const(np.array([3], dtype=np.int64)), const(np.array([4], dtype=np.int64)), const(np.array([1], dtype=np.int64))])
# a_en1: squares that MY pieces hit in 1 action; a_my1: squares THEIR pieces hit.  (names kept for clarity of use below)
red = const(np.array([2, 3], dtype=np.int64))
enemy_hit = op("ReduceSum", [mul(OBS, a_en1), red], keepdims=0)          # (B,16): enemy planes counted if I hit them
own_hit = op("ReduceSum", [mul(OBS, a_my1), red], keepdims=0)            # own planes counted if they hit them
Wh_en = np.zeros((16, 1), dtype=np.float32)
Wh_en[6:12, 0] = PVAL
Wh_own = np.zeros((16, 1), dtype=np.float32)
Wh_own[0:6, 0] = PVAL
hang = add(
    scale(gemm(enemy_hit, Wh_en), P["hang_en"]),
    scale(gemm(own_hit, Wh_own), -P["hang_own"]),
)

# firewall: material and kernels standing on the ring that locks next
fw_hit = op("ReduceSum", [mul(OBS, Dmap), red], keepdims=0)             # (B,16)
Wfm = np.zeros((16, 1), dtype=np.float32)
Wfm[0:5, 0] = -PVAL[:5] * P["fw_mat"]
Wfm[6:11, 0] = PVAL[:5] * P["fw_mat"]
fw_mat = mul(gemm(fw_hit, Wfm), upow)
k_own_d = gemm(fw_hit, np.eye(16, 1, -5, dtype=np.float32))     # own kernel on danger ring
k_en_d = gemm(fw_hit, np.eye(16, 1, -11, dtype=np.float32))
fw_k = add(scale(k_own_d, -P["fw_k_own"]),
           mul(k_en_d, add(const(np.array(P["fw_k_en"], dtype=np.float32)), scale(tm, P["fw_k_en_done"]))))
fw_k = mul(fw_k, upow)

v1 = add(mat, pos, threat, danger, hang, fw_mat, fw_k)
v2 = op("Tanh", [v1])
value = op("Reshape", [v2, shape_c(-1)], out="value")

# ---- policy:  logit[f,t] = A[f] + sum_p own_p[f] * S_p[t]
own = op("Slice", [OBS, const(np.array([0], dtype=np.int64)), const(np.array([6], dtype=np.int64)), const(np.array([1], dtype=np.int64))])  # (B,6,8,8)
enemy_pl = op("Slice", [OBS, const(np.array([6], dtype=np.int64)), const(np.array([12], dtype=np.int64)), const(np.array([1], dtype=np.int64))])
S1 = lambda x: op("Reshape", [x, shape_c(-1, 1, 64)])

# target-square scores shared by all pieces: capture gain, centrality, firewall
capw = np.zeros((1, 6, 1, 1), dtype=np.float32)
capw[0, :5, 0, 0] = PVAL[:5] / 0.3 * P["p_cap"] * 0.5
capw_k = P["p_kcap"]
cap_nonk = op("Conv", [enemy_pl, const(capw)])                          # (B,1,8,8)
ek = op("Slice", [OBS, const(np.array([11], dtype=np.int64)), const(np.array([12], dtype=np.int64)), const(np.array([1], dtype=np.int64))])
cap_k = mul(scale(ek, capw_k), op("Reshape", [mod_en, shape_c(-1, 1, 1, 1)]))
cap = add(cap_nonk, cap_k)
cen = const((P["p_cen"] * ring)[None, None])
fw_to = scale(mul(Dmap, op("Reshape", [upow, shape_c(-1, 1, 1, 1)])), -P["p_fw_to"])
base_to = add(cap, cen, fw_to)                                          # (B,1,8,8)

# attacked-square penalty depends on the mover's value:  -att_enemy[t] * pv_p
att_en_map = a_my1                                                      # squares their pieces hit
pen = mul(att_en_map, const(np.full((1, 6, 1, 1), 1.0, dtype=np.float32)))
penw = (np.array(list(PVAL[:5]) + [1.0], dtype=np.float32) / 0.3 * P["p_att"]).reshape(1, 6, 1, 1)
S = add(op("Mul", [pen, const(-penw)]), base_to)                        # (B,6,8,8) broadcast sum
S = op("Reshape", [S, shape_c(-1, 6, 64)])                              # (B,6,64) per mover-type to-score

# from-square score
fl = (np.array(list(PVAL[:5]) + [1.0], dtype=np.float32) / 0.3 * P["p_flee"]).reshape(6, 1, 1, 1)
flee = mul(own, mul(att_en_map, const(fl.reshape(1, 6, 1, 1))))          # own pieces under attack
fwf = np.array(list(PVAL[:5] / 0.3) + [0.0], dtype=np.float32).reshape(1, 6, 1, 1) * P["p_fw_from"]
fwf[0, 5] = P["p_kesc"]
evac = mul(mul(own, const(fwf.astype(np.float32))), mul(Dmap, op("Reshape", [upow, shape_c(-1, 1, 1, 1)])))
Aterm = op("ReduceSum", [add(flee, evac), const(np.array([1], dtype=np.int64))], keepdims=1)  # (B,1,8,8)
A = op("Reshape", [Aterm, shape_c(-1, 64, 1)])
own_f = op("Transpose", [op("Reshape", [own, shape_c(-1, 6, 64)])], perm=[0, 2, 1])               # (B,64,6)
pair = op("MatMul", [own_f, S])                                         # (B,64,64): own_p[f] * S_p[t]
logits = op("Reshape", [scale(add(pair, A), P["pol"]), shape_c(-1, 4096)])
policy = op("Identity", [logits], out="policy")

graph = helper.make_graph(
    nodes, "build_alt",
    [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 16, 8, 8])],
    [helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 4096]),
     helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, sys.argv[1])
print("saved", sys.argv[1])
