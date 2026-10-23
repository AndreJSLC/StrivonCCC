"""Train a residual net on top of the heuristic ONNX model and export the merged model.

usage: train.py heuristic.onnx out.onnx epochs data_glob [data_glob ...]
"""
import glob
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper, numpy_helper

torch.set_num_threads(12)
HEUR, OUT, EPOCHS = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = sorted({f for g in sys.argv[4:] for f in glob.glob(g)})

# ---------------- data ----------------
obs, legal, pi, z = [], [], [], []
for f in files:
    d = np.load(f)
    obs.append(d["obs"]); legal.append(np.unpackbits(d["legal"], axis=1)[:, :4096].astype(bool))
    pi.append(d["pi"]); z.append(d["z"])
obs, legal, pi, z = map(np.concatenate, (obs, legal, pi, z))
print("positions", len(z), "files", len(files), flush=True)

# mirror augmentation (col -> 7-col), applied on the fly; the heuristic is mirror symmetric
sq = np.arange(64); r, c = sq // 8, sq % 8
msq = r * 8 + (7 - c)
perm = (msq[:, None] * 64 + msq[None, :]).reshape(-1)
inv = np.empty(4096, dtype=np.int64); inv[perm] = np.arange(4096)
inv_t = torch.from_numpy(inv)

# heuristic outputs (pre-tanh value and policy logits)
hm = onnx.load(HEUR)
hm.graph.output.append(helper.make_tensor_value_info("v1", onnx.TensorProto.FLOAT, ["batch", 1]))
sess = ort.InferenceSession(hm.SerializeToString(), providers=["CPUExecutionProvider"])
hp, hv = [], []
for i in range(0, len(obs), 1024):
    p, _, v = sess.run(["policy", "value", "v1"], {"observation": obs[i:i + 1024]})
    hp.append(p.astype(np.float16)); hv.append(v[:, 0])
hp, hv = np.concatenate(hp), np.concatenate(hv)
print("heuristic done", flush=True)

n = len(z)
rng = np.random.default_rng(0)
idx = rng.permutation(n)
nval = n // 20
val_idx, tr_idx = idx[:nval], idx[nval:]

T = lambda a: torch.from_numpy(np.ascontiguousarray(a))
obs_t, legal_t, pi_t, z_t, hp_t, hv_t = T(obs), T(legal), T(pi), T(z), T(hp), T(hv)


# ---------------- model ----------------
class Block(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(y)))


class Net(nn.Module):
    def __init__(self, ch=64, blocks=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(16, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.tower = nn.Sequential(*[Block(ch) for _ in range(blocks)])
        self.ph = nn.Conv2d(ch, 64, 1)  # channel = to-square, spatial = from-square
        self.vh = nn.Sequential(nn.Conv2d(ch, 4, 1), nn.ReLU(), nn.Flatten(), nn.Linear(256, 64), nn.ReLU())
        self.vo = nn.Linear(64, 1)
        nn.init.zeros_(self.ph.weight); nn.init.zeros_(self.ph.bias)
        nn.init.zeros_(self.vo.weight); nn.init.zeros_(self.vo.bias)

    def forward(self, x):
        h = self.tower(self.stem(x))
        p = self.ph(h).reshape(-1, 64, 64).transpose(1, 2).reshape(-1, 4096)  # [from, to]
        v = self.vo(self.vh(h))
        return p, v


net = Net()
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
BS = 512
steps = EPOCHS * (len(tr_idx) // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 1))


def batch_loss(b, flip=False):
    x, hpb, lg, target = obs_t[b], hp_t[b].float(), legal_t[b], pi_t[b].float()
    if flip:
        x = torch.flip(x, dims=[3])
        hpb, lg, target = hpb[:, inv_t], lg[:, inv_t], target[:, inv_t]
    p, v = net(x)
    logits = hpb + p
    logits = logits.masked_fill(~lg, -1e9)
    lp = F.log_softmax(logits, dim=1)
    pl = -(target * lp).sum(1).mean()
    vl = F.mse_loss(torch.tanh(hv_t[b] + v[:, 0]), z_t[b])
    return pl, vl


def evaluate():
    net.eval()
    with torch.no_grad():
        pl, vl = batch_loss(T(val_idx))
    net.train()
    return pl.item(), vl.item()


print("val before", evaluate(), flush=True)
t0 = time.time()
for ep in range(EPOCHS):
    perm_tr = rng.permutation(tr_idx)
    for i in range(0, len(perm_tr) - BS + 1, BS):
        b = T(perm_tr[i:i + BS])
        pl, vl = batch_loss(b, flip=bool(rng.integers(2)))
        loss = pl + vl
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    print(f"epoch {ep} val {evaluate()} {time.time() - t0:.0f}s", flush=True)

# ---------------- export + merge ----------------
net.eval()
tmp = OUT + ".net.onnx"
torch.onnx.export(net, torch.zeros(2, 16, 8, 8), tmp, input_names=["observation"],
                  output_names=["rp", "rv"], opset_version=17,
                  dynamic_axes={"observation": {0: "batch"}, "rp": {0: "batch"}, "rv": {0: "batch"}},
                  dynamo=False)
nm = onnx.load(tmp)
hm = onnx.load(HEUR)


def ren(name):
    return name if name in ("observation", "") else "n_" + name


for node in nm.graph.node:
    node.input[:] = [ren(x) for x in node.input]
    node.output[:] = [ren(x) for x in node.output]
    node.name = "n_" + node.name
for init in nm.graph.initializer:
    init.name = ren(init.name)

g = hm.graph
keep = [nd for nd in g.node if not (set(nd.output) & {"v2", "value", "policy"})]
pol_node = [nd for nd in g.node if "policy" in nd.output][0]
pol_node.output[:] = ["h_policy"]
keep.append(pol_node)
extra = [
    helper.make_node("Add", ["h_policy", "n_rp"], ["policy"]),
    helper.make_node("Add", ["v1", "n_rv"], ["vsum"]),
    helper.make_node("Tanh", ["vsum"], ["vt"]),
    helper.make_node("Reshape", ["vt", "flat"], ["value"]),
]
graph = helper.make_graph(
    keep + list(nm.graph.node) + extra, "hybrid", list(g.input), list(g.output),
    list(g.initializer) + list(nm.graph.initializer))
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, OUT)
torch.save(net.state_dict(), OUT + ".pt")
print("saved", OUT, flush=True)
