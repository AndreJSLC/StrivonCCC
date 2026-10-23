"""Store a model's large float32 weights as float16 initializers plus a Cast back to float32.
Cast is on the submission allow-list, and onnxruntime folds Cast(initializer)
into a float32 constant at session start, so inference cost is unchanged while the file roughly
halves. That lets about 2x more parameters fit under the 5 MiB upload limit.

usage: fp16w.py in.onnx out.onnx [min_elements=1024]
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
min_el = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
m = onnx.load(src)
g = m.graph
casts, keep, n_conv, saved = [], [], 0, 0
for init in g.initializer:
    arr = numpy_helper.to_array(init)
    if arr.dtype == np.float32 and arr.size >= min_el:
        h = arr.astype(np.float16)
        keep.append(numpy_helper.from_array(h, init.name + "__f16"))
        casts.append(helper.make_node("Cast", [init.name + "__f16"], [init.name], to=onnx.TensorProto.FLOAT,
                                      name="cast_" + init.name))
        n_conv += 1
        saved += arr.nbytes - h.nbytes
    else:
        keep.append(init)
del g.initializer[:]
g.initializer.extend(keep)
nodes = casts + list(g.node)
del g.node[:]
g.node.extend(nodes)
onnx.checker.check_model(m)
onnx.save(m, dst)
print(f"{dst}: {n_conv} initializers stored as float16, {saved / 1024:.0f} KB saved")
