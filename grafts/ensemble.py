"""Average several policy/value ONNX models into one file.

The referee fixes the search at 100 simulations, so spare CPU budget can only buy strength through a
better evaluator; an ensemble of independently trained nets is the simplest way to spend it.

usage: ensemble.py OUT.onnx A.onnx[:w] B.onnx[:w] ...
  policy = sum_i w_i * logits_i (weights normalised to sum 1, i.e. a product of experts)
  value  = sum_i w_i * value_i
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def load_prefixed(path, pre):
    m = onnx.load(path)
    g = m.graph
    keep = {"observation", ""}
    ren = lambda s: s if s in keep else f"{pre}{s}"
    nodes = []
    for nd in g.node:
        if nd.op_type == "Constant":  # not on the allow-list: turn into an initializer
            t = helper.get_attribute_value({a.name: a for a in nd.attribute}["value"])
            g.initializer.append(numpy_helper.from_array(numpy_helper.to_array(t), nd.output[0]))
            continue
        nd.input[:] = [ren(x) for x in nd.input]
        nd.output[:] = [ren(x) for x in nd.output]
        nd.name = f"{pre}n{len(nodes)}_{nd.name}"
        nodes.append(nd)
    inits = []
    for t in g.initializer:
        t.name = ren(t.name)
        inits.append(t)
    opset = max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))
    return nodes, inits, ren("policy"), ren("value"), opset


def main():
    out = sys.argv[1]
    specs = []
    for s in sys.argv[2:]:
        p, _, w = s.partition(":")
        specs.append((p, float(w) if w else 1.0))
    tot = sum(w for _, w in specs)
    nodes, inits, pols, vals, opsets = [], [], [], [], []
    for i, (p, w) in enumerate(specs):
        n, ini, pol, val, ops = load_prefixed(p, f"m{i}_")
        nodes += n; inits += ini; opsets.append(ops)
        wn = f"m{i}_ensw"
        inits.append(numpy_helper.from_array(np.array(w / tot, dtype=np.float32), wn))
        nodes.append(helper.make_node("Mul", [pol, wn], [f"m{i}_wpol"]))
        nodes.append(helper.make_node("Mul", [val, wn], [f"m{i}_wval"]))
        pols.append(f"m{i}_wpol"); vals.append(f"m{i}_wval")
    nodes.append(helper.make_node("Sum", pols, ["policy"]))
    nodes.append(helper.make_node("Sum", vals, ["value"]))
    graph = helper.make_graph(
        nodes, "ensemble",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 16, 8, 8])],
        [helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 4096]),
         helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])],
        inits,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", max(opsets))])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, out)
    print("saved", out, "from", specs)


if __name__ == "__main__":
    main()
