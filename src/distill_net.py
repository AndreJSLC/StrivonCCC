"""Small residual conv policy/value net + ONNX export (dynamic batch, weights embedded)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self, c=48, blocks=4, d=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(16, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(c) for _ in range(blocks)])
        self.q = nn.Conv2d(c, d, 1)
        self.k = nn.Conv2d(c, d, 1)
        self.d = d
        self.scale = d ** -0.5
        self.vconv = nn.Sequential(nn.Conv2d(c, 8, 1), nn.ReLU())
        self.vfc1 = nn.Linear(8 * 64 + 4, 64)
        self.vfc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        q = self.q(h).reshape(-1, self.d, 64)  # (b,d,64) from
        k = self.k(h).reshape(-1, self.d, 64)  # (b,d,64) to
        logits = torch.matmul(q.transpose(1, 2), k) * self.scale  # (b,64,64)
        logits = logits.reshape(-1, 4096)
        sp = torch.cat([x[:, 13:16, 0, 0], x[:, 12:13].mean(dim=(2, 3))], 1)  # actions_left, urgency, side, locked frac
        v = self.vconv(h).reshape(-1, 8 * 64)
        v = F.relu(self.vfc1(torch.cat([v, sp], 1)))
        v = torch.tanh(self.vfc2(v)).reshape(-1)
        return logits, v


def export(net, path):
    net = net.eval().cpu()
    dummy = torch.zeros(2, 16, 8, 8)
    torch.onnx.export(
        net, dummy, path, input_names=["observation"], output_names=["policy", "value"],
        dynamic_axes={"observation": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    import onnx
    m = onnx.load(path)
    from onnx import numpy_helper
    keep = []
    for n in m.graph.node:  # Constant nodes are not on the allow-list -> initializers
        if n.op_type == "Constant":
            t = numpy_helper.to_array(n.attribute[0].t)
            m.graph.initializer.append(numpy_helper.from_array(t, n.output[0]))
        else:
            keep.append(n)
    del m.graph.node[:]
    m.graph.node.extend(keep)
    d1 = m.graph.output[0].type.tensor_type.shape.dim[1]
    d1.Clear()
    d1.dim_value = 4096
    onnx.save_model(m, path, save_as_external_data=False)
