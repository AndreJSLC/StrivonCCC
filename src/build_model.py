"""Build a hand-weighted material evaluator as an ONNX policy/value model."""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# planes 0-5 own Ping, Daemon, Exploit, Trace, Ice, Kernel; 6-11 enemy
VAL = np.array([0.15, 0.45, 0.25, 0.30, 0.50, 3.0], dtype=np.float32)
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5

w = np.zeros((16, 1), dtype=np.float32)
w[0:6, 0] = VAL * SCALE
w[6:12, 0] = -VAL * SCALE

KC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3  # kernel centrality per ring
PC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03  # piece centrality per ring
ring = np.array([[min(r, c, 7 - r, 7 - c) for c in range(8)] for r in range(8)], dtype=np.float32)
sp = np.zeros((16, 8, 8), dtype=np.float32)
for p in range(6):
    sp[p] = PC * ring
    sp[6 + p] = -PC * ring
sp[5] = KC * ring
sp[11] = -KC * ring

inits = [
    numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes"),
    numpy_helper.from_array(sp[None], "W_sp"),
    numpy_helper.from_array(np.array([1], dtype=np.int64), "ax1"),
    numpy_helper.from_array(w, "W_v"),
    numpy_helper.from_array(np.zeros((16, 4096), dtype=np.float32), "W_p"),
    numpy_helper.from_array(np.array([-1], dtype=np.int64), "flat"),
]
nodes = [
    helper.make_node("ReduceSum", ["observation", "axes"], ["counts"], keepdims=0),
    helper.make_node("Gemm", ["counts", "W_v"], ["v0"]),
    helper.make_node("Mul", ["observation", "W_sp"], ["xs"]),
    helper.make_node("ReduceSum", ["xs", "axes"], ["xs2"], keepdims=0),
    helper.make_node("ReduceSum", ["xs2", "ax1"], ["pos"], keepdims=1),
    helper.make_node("Add", ["v0", "pos"], ["v1"]),
    helper.make_node("Tanh", ["v1"], ["v2"]),
    helper.make_node("Reshape", ["v2", "flat"], ["value"]),
    helper.make_node("Gemm", ["counts", "W_p"], ["policy"]),
]
graph = helper.make_graph(
    nodes, "material",
    [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 16, 8, 8])],
    [helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 4096]),
     helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, sys.argv[1] if len(sys.argv) > 1 else "model.onnx")
print("saved")
