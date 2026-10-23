"""Inference-time knobs on a trained hybrid model, by graph surgery.

    policy' = policy * ps * (1 + os * opening)      opening = no locked square and urgency <= 0.25,
                                                     i.e. rounds 1-2 (the referee's sampled warmup)
    value'  = tanh(vs * pre_tanh_value)

ps sharpens or flattens the prior the referee's MCTS uses, os does so only in the opening (where
the referee samples moves from visit counts), and vs rescales the value the search backs up.
Needs a graph whose value is Reshape(Tanh(vsum)), as written by train2.py and arch.py.

usage: knob.py in.onnx out.onnx ps os vs
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst, ps, os_, vs = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
m = onnx.load(src)
g = m.graph
pol = [n for n in g.node if "policy" in n.output][0]
pol.output[:] = ["knob_policy_raw"]
tanh = [n for n in g.node if n.op_type == "Tanh" and n.input[0] == "vsum"]
assert tanh, "no Tanh(vsum) node: not a train2/arch hybrid"
tanh[0].input[0] = "knob_vsum_s"
T = numpy_helper.from_array
g.initializer.extend([
    T(np.array([14], dtype=np.int64), "knob_s14"), T(np.array([15], dtype=np.int64), "knob_e15"),
    T(np.array([12], dtype=np.int64), "knob_s12"), T(np.array([13], dtype=np.int64), "knob_e13"),
    T(np.array([1], dtype=np.int64), "knob_ax1"),
    T(np.array(0.3, dtype=np.float32), "knob_uthr"), T(np.array(0.5, dtype=np.float32), "knob_lthr"),
    T(np.array(ps, dtype=np.float32), "knob_ps"), T(np.array(os_, dtype=np.float32), "knob_os"),
    T(np.array(1.0, dtype=np.float32), "knob_one"), T(np.array(vs, dtype=np.float32), "knob_vs"),
])
N = helper.make_node
g.node.extend([
    N("Slice", ["observation", "knob_s14", "knob_e15", "knob_ax1"], ["knob_urg"]),
    N("ReduceMax", ["knob_urg"], ["knob_urgm"], axes=[1, 2, 3], keepdims=0),          # (B,)
    N("Slice", ["observation", "knob_s12", "knob_e13", "knob_ax1"], ["knob_lock"]),
    N("ReduceMax", ["knob_lock"], ["knob_lockm"], axes=[1, 2, 3], keepdims=0),        # (B,)
    N("Less", ["knob_urgm", "knob_uthr"], ["knob_early"]),
    N("Less", ["knob_lockm", "knob_lthr"], ["knob_nolock"]),
    N("And", ["knob_early", "knob_nolock"], ["knob_open_b"]),
    N("Cast", ["knob_open_b"], ["knob_open"], to=onnx.TensorProto.FLOAT),
    N("Mul", ["knob_open", "knob_os"], ["knob_o1"]),
    N("Add", ["knob_o1", "knob_one"], ["knob_o2"]),
    N("Mul", ["knob_o2", "knob_ps"], ["knob_scale"]),                                   # (B,)
    N("Unsqueeze", ["knob_scale", "knob_ax1"], ["knob_scale2"]),                        # (B,1)
    N("Mul", ["knob_policy_raw", "knob_scale2"], ["policy"]),
    N("Mul", ["vsum", "knob_vs"], ["knob_vsum_s"]),
])
# keep a topological order: move the policy consumers after their inputs
order = {n.output[0]: i for i, n in enumerate(g.node)}
nodes = list(g.node)
del g.node[:]
done, pending = set(i.name for i in g.initializer) | {"observation"}, nodes
while pending:
    rest = []
    for n in pending:
        if all(x in done or x == "" for x in n.input):
            g.node.append(n)
            done.update(n.output)
        else:
            rest.append(n)
    if len(rest) == len(pending):
        raise SystemExit("graph has a cycle or a missing input: " + str([n.name for n in rest][:5]))
    pending = rest
onnx.checker.check_model(m)
onnx.save(m, dst)
print("saved", dst, "ps", ps, "os", os_, "vs", vs)
