"""Replace Constant nodes with initializers (Constant is not on the submission allow-list).

usage: fixconst.py model.onnx [more.onnx ...]   (rewrites in place)
"""
import sys

import onnx
from onnx import numpy_helper


def fix(path):
    m = onnx.load(path)
    g = m.graph
    keep = []
    for nd in g.node:
        if nd.op_type == "Constant":
            attr = {a.name: a for a in nd.attribute}
            t = onnx.helper.get_attribute_value(attr["value"])
            arr = numpy_helper.to_array(t)
            g.initializer.append(numpy_helper.from_array(arr, nd.output[0]))
        else:
            keep.append(nd)
    del g.node[:]
    g.node.extend(keep)
    onnx.checker.check_model(m)
    onnx.save(m, path)
    print("fixed", path, "ops", sorted({n.op_type for n in g.node}))


for p in sys.argv[1:]:
    fix(p)
