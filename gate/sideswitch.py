"""Side-switch net: member A answers when Cyan is to move, member B when Magenta
is to move. The switch reads observation plane 15 (1.0 = Magenta to move) and blends with Mul/Add
only, so both members always run (cost = A + B), with no control flow.

usage: sideswitch.py out.onnx cyan_model.onnx magenta_model.onnx
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

out, ma, mb = sys.argv[1:4]
T = numpy_helper.from_array
N = helper.make_node
nodes, inits = [], [T(np.array([15], np.int64), "s_s15"), T(np.array([16], np.int64), "s_e16"),
                    T(np.array([1], np.int64), "s_ax1")]
outs = []
for mi, path in enumerate((ma, mb)):
    m = onnx.load(path)
    g = m.graph
    pre = f"m{mi}_"
    ren = lambda s: s if s in ("observation", "") else pre + s
    for nd in g.node:
        nd.input[:] = [ren(s) for s in nd.input]
        nd.output[:] = [ren(s) for s in nd.output]
        nd.name = pre + (nd.name or "n")
        nodes.append(nd)
    for it in g.initializer:
        it.name = pre + it.name
        inits.append(it)
    outs.append((pre + g.output[0].name, pre + g.output[1].name))
(pa, va), (pb, vb) = outs
nodes += [
    N("Slice", ["observation", "s_s15", "s_e16", "s_ax1"], ["s_pl"]),
    N("ReduceMax", ["s_pl"], ["s_m"], axes=[1, 2, 3], keepdims=0),          # (B,) 1 = Magenta to move
    N("Unsqueeze", ["s_m", "s_ax1"], ["s_m2"]),                              # (B,1)
    N("Sub", [pb, pa], ["s_dp"]), N("Mul", ["s_dp", "s_m2"], ["s_dp2"]), N("Add", [pa, "s_dp2"], ["policy"]),
    N("Sub", [vb, va], ["s_dv"]), N("Mul", ["s_dv", "s_m"], ["s_dv2"]), N("Add", [va, "s_dv2"], ["value"]),
]
seen = set()
for i, nd in enumerate(nodes):
    if not nd.name or nd.name in seen:
        nd.name = f"sw_n{i}"
    seen.add(nd.name)
graph = helper.make_graph(
    nodes, "sideswitch",
    [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 16, 8, 8])],
    [helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 4096]),
     helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])],
    inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out)
print("saved", out, "cyan-to-move:", ma, "magenta-to-move:", mb)
