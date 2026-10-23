"""Spend more of the CPU budget without retraining: ensembles and symmetry test-time
augmentation (TTA) built by ONNX graph surgery.

Every member net is evaluated on V symmetric views of the board (the game is D4-symmetric).
The views are batched together, so one call evaluates V*B positions. Each view's policy is mapped
back to the original squares with a constant Gather, and the logits and values are averaged over
views and members. Cost is about V * sum(member cost). Only allow-listed ops are used: Slice with
constant negative steps, Transpose, Concat, Reshape/Split/Squeeze with constant shapes, Gather with
constant indices, Sum/Mul, ReduceMean.

usage: ens.py out.onnx VIEWS member1.onnx [member2.onnx ...]
       VIEWS: 1 (identity), 2 (+mirror), 4 (+row flip, +rot180), 8 (full D4)
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

out, V, members = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
assert V in (1, 2, 4, 8)
T = numpy_helper.from_array
N = helper.make_node
INT_MIN = np.iinfo(np.int64).min + 1

# square maps for each view: view[r', c'] = x[T(r', c')], and T is an involution for these
def sq_map(kind):
    m = np.zeros(64, dtype=np.int64)
    for r in range(8):
        for c in range(8):
            rr, cc = {"id": (r, c), "fc": (r, 7 - c), "fr": (7 - r, c), "r180": (7 - r, 7 - c),
                      "tr": (c, r), "tr_fc": (c, 7 - r), "tr_fr": (7 - c, r), "tr_r180": (7 - c, 7 - r)}[kind]
            m[r * 8 + c] = rr * 8 + cc
    return m


kinds = ["id", "fc", "fr", "r180", "tr", "tr_fc", "tr_fr", "tr_r180"][:V]
nodes, inits = [], []
inits += [T(np.array([-1], np.int64), "e_start"), T(np.array([INT_MIN], np.int64), "e_end"),
          T(np.array([-1], np.int64), "e_step"), T(np.array([2], np.int64), "e_ax2"),
          T(np.array([3], np.int64), "e_ax3"), T(np.array([0], np.int64), "e_ax0")]


def flip(src, axis_name, dst):
    nodes.append(N("Slice", [src, "e_start", "e_end", axis_name, "e_step"], [dst]))


view_names = []
for kind in kinds:
    x = "observation"
    if kind.startswith("tr"):
        nodes.append(N("Transpose", [x], [f"e_{kind}_t"], perm=[0, 1, 3, 2])); x = f"e_{kind}_t"
        kind_rest = kind[3:] if len(kind) > 2 else "id"
    else:
        kind_rest = kind
    if kind_rest in ("fc", "r180"):
        flip(x, "e_ax3", f"e_{kind}_c"); x = f"e_{kind}_c"
    if kind_rest in ("fr", "r180"):
        flip(x, "e_ax2", f"e_{kind}_r"); x = f"e_{kind}_r"
    view_names.append(x)
if V > 1:
    nodes.append(N("Concat", view_names, ["e_X"], axis=0))
else:
    nodes.append(N("Identity", ["observation"], ["e_X"]))

# per-view policy maps: original move (s1, s2) <- view move (T(s1), T(s2))
perms = []
for kind in kinds:
    if kind.startswith("tr"):
        # view = transpose then flip; the square map composes the same way (inverse order applies)
        base, rest = sq_map("tr"), kind[3:] if len(kind) > 2 else "id"
        rm = sq_map(rest if rest != "" else "id")
        smap = base[rm]  # view square of original square s: first flip-index, then transpose
    else:
        smap = sq_map(kind)
    perms.append((smap[:, None] * 64 + smap[None, :]).reshape(-1))

pol_terms, val_terms = [], []
for mi, path in enumerate(members):
    m = onnx.load(path)
    g = m.graph
    pre = f"m{mi}_"
    ren = lambda s: "e_X" if s == "observation" else ("" if s == "" else pre + s)
    for nd in g.node:
        nd.input[:] = [ren(s) for s in nd.input]
        nd.output[:] = [ren(s) for s in nd.output]
        nd.name = pre + nd.name
        nodes.append(nd)
    for it in g.initializer:
        it.name = pre + it.name
        inits.append(it)
    P, Vv = pre + g.output[0].name, pre + g.output[1].name
    if V > 1:
        inits += [T(np.array([V, -1, 4096], np.int64), pre + "e_shpP"), T(np.array([V, -1], np.int64), pre + "e_shpV"),
                  T(np.array([1] * V, np.int64), pre + "e_split")]
        nodes.append(N("Reshape", [P, pre + "e_shpP"], [pre + "e_P3"]))
        nodes.append(N("Split", [pre + "e_P3", pre + "e_split"], [f"{pre}e_Pv{k}" for k in range(V)], axis=0))
        for k in range(V):
            nodes.append(N("Squeeze", [f"{pre}e_Pv{k}", "e_ax0"], [f"{pre}e_Ps{k}"]))
            if kinds[k] == "id":
                pol_terms.append(f"{pre}e_Ps{k}")
            else:
                inits.append(T(perms[k], f"e_perm{k}") if mi == 0 else None)
                nodes.append(N("Gather", [f"{pre}e_Ps{k}", f"e_perm{k}"], [f"{pre}e_Pg{k}"], axis=1))
                pol_terms.append(f"{pre}e_Pg{k}")
        nodes.append(N("Reshape", [Vv, pre + "e_shpV"], [pre + "e_V2"]))
        nodes.append(N("ReduceMean", [pre + "e_V2"], [pre + "e_Vm"], axes=[0], keepdims=0))
        val_terms.append(pre + "e_Vm")
    else:
        pol_terms.append(P)
        val_terms.append(Vv)
inits = [i for i in inits if i is not None]
import os
W = [float(w) for w in os.environ["ENS_W"].split(",")] if os.environ.get("ENS_W") else [1.0] * len(members)
assert len(W) == len(members)
W = [w / sum(W) for w in W]
# per-member weights: each member contributes V policy terms (one per view) and one value term
per = len(pol_terms) // len(members)
wp_terms = []
for mi in range(len(members)):
    inits.append(T(np.array(W[mi] / per, np.float32), f"e_wp{mi}"))
    inits.append(T(np.array(W[mi], np.float32), f"e_wv{mi}"))
    nodes.append(N("Sum", pol_terms[mi * per:(mi + 1) * per], [f"e_Pm{mi}"]))
    nodes.append(N("Mul", [f"e_Pm{mi}", f"e_wp{mi}"], [f"e_Pw{mi}"]))
    nodes.append(N("Mul", [val_terms[mi], f"e_wv{mi}"], [f"e_Vw{mi}"]))
    wp_terms.append(mi)
nodes.append(N("Sum", [f"e_Pw{mi}" for mi in wp_terms], ["policy"]))
nodes.append(N("Sum", [f"e_Vw{mi}" for mi in wp_terms], ["value"]))
seen = set()
for i, nd in enumerate(nodes):  # onnxruntime rejects duplicate node names
    if not nd.name or nd.name in seen:
        nd.name = f"ens_n{i}"
    seen.add(nd.name)
graph = helper.make_graph(
    nodes, "ens",
    [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 16, 8, 8])],
    [helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 4096]),
     helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])],
    inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out)
print("saved", out, "views", kinds, "members", members)
