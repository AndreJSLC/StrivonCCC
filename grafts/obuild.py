"""Hand-built ONNX graph helpers and a grafting tool (the equation-graft kit).

Two ways to use it:

  * graft(base.onnx, build, out.onnx): keep a trained policy/value net and post-process its
    outputs. ``build(g, obs, net_policy, net_value)`` appends nodes to ``g`` and returns the names
    of the final policy ``(B, 4096)`` and value ``(B,)`` tensors.
  * standalone(build, out.onnx): a model with no net; ``build`` gets net_policy=net_value=None.

Allow-list reminders (tools/validate_submission.py enforces them): opset 13-18, no Constant node
(constants here are initializers), no Pad/Expand/Tile/ScatterND/If/Loop/OneHot/Range, and every
shape/axes/index argument of Reshape/Slice/Gather/Squeeze/Unsqueeze/Split must be a constant.
Allowed: Add Sub Mul Div Pow Neg Abs Sqrt Exp Log Min Max Sum Mean Clip Equal Greater
GreaterOrEqual Less LessOrEqual And Or Not Xor Where Relu LeakyRelu Sigmoid Tanh Softplus Softmax
LogSoftmax Conv Gemm MatMul BatchNormalization AveragePool MaxPool GlobalAveragePool GlobalMaxPool
Identity Cast Flatten Transpose Concat Reshape Squeeze Unsqueeze Split ReduceSum ReduceMean
ReduceMax ReduceMin ReduceProd Gather Slice (+ a few more, see docs/submission.md).

Square indexing is canonical: square = row * 8 + col, action = from_square * 64 + to_square.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17


class G:
    """Accumulates nodes and initializers; every helper returns the output tensor name."""

    def __init__(self, prefix: str = "eq"):
        self.nodes: list = []
        self.inits: list = []
        self.prefix = prefix
        self._n = 0

    # ---- plumbing
    def name(self, tag: str = "t") -> str:
        self._n += 1
        return f"{self.prefix}_{tag}{self._n}"

    def const(self, arr, dtype=None, tag: str = "c") -> str:
        arr = np.asarray(arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        elif arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif arr.dtype == np.int32:
            arr = arr.astype(np.int64)
        name = self.name(tag)
        self.inits.append(numpy_helper.from_array(arr, name))
        return name

    def f(self, x: float) -> str:
        """Scalar float32 constant."""
        return self.const(np.array(x, dtype=np.float32))

    def ints(self, *vals) -> str:
        return self.const(np.array(vals, dtype=np.int64))

    def op(self, kind: str, *inputs, n_out: int = 1, tag: str | None = None, **attrs):
        outs = [self.name(tag or kind.lower()) for _ in range(n_out)]
        self.nodes.append(helper.make_node(kind, list(inputs), outs, **attrs))
        return outs[0] if n_out == 1 else outs

    # ---- arithmetic (binary-float logic: AND = mul, OR = max, NOT = 1 - x)
    def add(self, *xs):
        return xs[0] if len(xs) == 1 else self.op("Sum", *xs)

    def sub(self, a, b):
        return self.op("Sub", a, b)

    def mul(self, a, b):
        return self.op("Mul", a, b)

    def scale(self, x, s: float):
        return self.op("Mul", x, self.f(s))

    def relu(self, x):
        return self.op("Relu", x)

    def clip01(self, x):
        return self.op("Clip", x, self.f(0.0), self.f(1.0))

    def minimum(self, *xs):
        return self.op("Min", *xs)

    def maximum(self, *xs):
        return self.op("Max", *xs)

    def not_(self, x):
        return self.op("Sub", self.f(1.0), x)

    def where(self, cond_float, a, b):
        """cond_float in {0,1}; returns a where cond else b (broadcasting)."""
        c = self.op("Greater", cond_float, self.f(0.5))
        return self.op("Where", c, a, b)

    # ---- shapes
    def reshape(self, x, *shape):
        return self.op("Reshape", x, self.ints(*shape))

    def planes(self, obs, lo: int, hi: int):
        """obs[:, lo:hi] -> (B, hi-lo, 8, 8)."""
        return self.op("Slice", obs, self.ints(lo), self.ints(hi), self.ints(1))

    def flat64(self, x):
        """(B, C, 8, 8) -> (B, C, 64)."""
        return self.op("Reshape", x, self.ints(0, 0, 64))

    def spatial(self, x64, M):
        """y[..., t] = sum_s x[..., s] * M[s, t] for x of shape (..., 64). Any fixed linear map of the
        board (shift, step along a ray, neighbourhood sum) is one 64x64 matrix."""
        return self.op("MatMul", x64, self.const(np.asarray(M, dtype=np.float32)))

    def sum_hw(self, x, keepdims: int = 0):
        """(B, C, 8, 8) -> (B, C) (or (B, C, 1, 1))."""
        return self.op("ReduceSum", x, self.ints(2, 3), keepdims=keepdims)

    def sum_axes(self, x, axes, keepdims: int = 0):
        return self.op("ReduceSum", x, self.ints(*axes), keepdims=keepdims)

    def max_axes(self, x, axes, keepdims: int = 0):
        return self.op("ReduceMax", x, axes=list(axes), keepdims=keepdims)

    def conv(self, x, W, pads=None, groups: int = 1):
        W = np.asarray(W, dtype=np.float32)
        kh, kw = W.shape[2], W.shape[3]
        if pads is None:
            pads = [kh // 2, kw // 2, kh // 2, kw // 2]
        return self.op("Conv", x, self.const(W), pads=list(pads), group=groups)


# ---- board geometry helpers (pure numpy, build-time only)
def sq(r: int, c: int) -> int:
    return r * 8 + c


def on_board(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def shift_matrix(dr: int, dc: int) -> np.ndarray:
    """M[s, t] = 1 when t = s + (dr, dc) on the board: y = x @ M moves every bit by (dr, dc)."""
    M = np.zeros((64, 64), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            if on_board(r + dr, c + dc):
                M[sq(r, c), sq(r + dr, c + dc)] = 1.0
    return M


RING = np.array([[min(r, c, 7 - r, 7 - c) for c in range(8)] for r in range(8)], dtype=np.float32)


# ---- model assembly
def _fix_constants(graph) -> None:
    keep = []
    for nd in graph.node:
        if nd.op_type == "Constant":
            attr = {a.name: a for a in nd.attribute}
            t = helper.get_attribute_value(attr["value"])
            graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(t), nd.output[0]))
        else:
            keep.append(nd)
    del graph.node[:]
    graph.node.extend(keep)


def graft(base_path: str, build, out_path: str, prefix: str = "eq") -> str:
    """Wrap a trained model: its outputs become ``net_policy`` (B,4096) and ``net_value`` (B,)."""
    m = onnx.load(base_path)
    g = m.graph
    ren = {"policy": "net_policy", "value": "net_value"}
    for nd in g.node:
        nd.output[:] = [ren.get(o, o) for o in nd.output]
        nd.input[:] = [ren.get(i, i) for i in nd.input]
    _fix_constants(g)
    b = G(prefix)
    pol, val = build(b, "observation", "net_policy", "net_value")
    b.nodes.append(helper.make_node("Identity", [pol], ["policy"]))
    b.nodes.append(helper.make_node("Identity", [val], ["value"]))
    g.node.extend(b.nodes)
    g.initializer.extend(b.inits)
    del g.output[:]
    g.output.extend([
        helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 4096]),
        helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"]),
    ])
    del g.value_info[:]
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    return out_path


def standalone(build, out_path: str, prefix: str = "eq") -> str:
    b = G(prefix)
    pol, val = build(b, "observation", None, None)
    b.nodes.append(helper.make_node("Identity", [pol], ["policy"]))
    b.nodes.append(helper.make_node("Identity", [val], ["value"]))
    graph = helper.make_graph(
        b.nodes, "standalone",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 16, 8, 8])],
        [helper.make_tensor_value_info("policy", TensorProto.FLOAT, ["batch", 4096]),
         helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])],
        b.inits,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    return out_path


def features_model(build_feats, out_path: str, prefix: str = "ft") -> str:
    """Export a feature extractor alone: ``build_feats(g, obs) -> {name: tensor}``; each becomes an
    output (shapes are left dynamic). Handy for checking features against the engine in numpy."""
    b = G(prefix)
    feats = build_feats(b, "observation")
    outs = []
    for k, t in feats.items():
        b.nodes.append(helper.make_node("Identity", [t], [k]))
        outs.append(helper.make_tensor_value_info(k, TensorProto.FLOAT, None))
    graph = helper.make_graph(
        b.nodes, "features",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 16, 8, 8])],
        outs, b.inits,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    return out_path
