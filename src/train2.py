"""Train a residual (or weight-shared recurrent) net on top of the heuristic ONNX model, export merged.

usage: train2.py --heur h.onnx --out out.onnx [--ch 48] [--blocks 4] [--recurrent K] [--epochs 6]
                 [--init prev.pt] data_glob...
"""
import argparse
import glob
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper

ap = argparse.ArgumentParser()
ap.add_argument("--heur", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ch", type=int, default=48)
ap.add_argument("--blocks", type=int, default=4)
ap.add_argument("--recurrent", type=int, default=0, help="if >0: one shared block applied K times")
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--init", default=None)
ap.add_argument("data", nargs="+")
A = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
files = sorted({f for g in A.data for f in glob.glob(g)})

obs, legal, pi, z = [], [], [], []
for f in files:
    d = np.load(f)
    obs.append(d["obs"]); legal.append(np.unpackbits(d["legal"], axis=1)[:, :4096].astype(bool))
    pi.append(d["pi"])
    # value target: mean of game outcome and root search value when available (less overfitting)
    z.append(0.5 * d["z"] + 0.5 * d["q"] if "q" in d.files else d["z"])
obs, legal, pi, z = map(np.concatenate, (obs, legal, pi, z))
print("positions", len(z), "files", len(files), flush=True)

hm = onnx.load(A.heur)
hm.graph.output.append(helper.make_tensor_value_info("v1", onnx.TensorProto.FLOAT, ["batch", 1]))
sess = ort.InferenceSession(hm.SerializeToString(), providers=["CPUExecutionProvider"])
hp, hv = [], []
for i in range(0, len(obs), 2048):
    p, v = sess.run(["policy", "v1"], {"observation": obs[i:i + 2048]})
    hp.append(p.astype(np.float16)); hv.append(v[:, 0])
hp, hv = np.concatenate(hp), np.concatenate(hv)
print("heuristic done", flush=True)

# D4 symmetries: inverse action permutations (new index -> old index)
grid = np.arange(64).reshape(8, 8)
SYM = []
for k in range(4):
    for fl in (False, True):
        g = np.rot90(grid, k)
        if fl:
            g = g[:, ::-1]
        isq = g.reshape(-1)
        inv = (isq[:, None] * 64 + isq[None, :]).reshape(-1)
        SYM.append((k, fl, torch.from_numpy(inv.copy()).to(dev)))

n = len(z)
rng = np.random.default_rng(0)
idx = rng.permutation(n)
nval = max(n // 20, 1)
val_idx, tr_idx = idx[:nval], idx[nval:]
G = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
obs_t, legal_t, pi_t, z_t, hp_t, hv_t = G(obs), G(legal), G(pi), G(z), G(hp), G(hv)


class Block(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(y)))


class Net(nn.Module):
    def __init__(self, ch, blocks, recurrent):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(16, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.recurrent = recurrent
        if recurrent:
            # shared weights, per-iteration batchnorm (cheap, keeps statistics well defined)
            self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.inj = nn.Conv2d(16, ch, 1)  # re-inject the input every iteration
            self.b1 = nn.ModuleList([nn.BatchNorm2d(ch) for _ in range(recurrent)])
            self.b2 = nn.ModuleList([nn.BatchNorm2d(ch) for _ in range(recurrent)])
        else:
            self.tower = nn.Sequential(*[Block(ch) for _ in range(blocks)])
        self.ph = nn.Conv2d(ch, 64, 1)
        self.vh = nn.Sequential(nn.Conv2d(ch, 4, 1), nn.ReLU(), nn.Flatten(), nn.Linear(256, 64), nn.ReLU())
        self.vo = nn.Linear(64, 1)
        nn.init.zeros_(self.ph.weight); nn.init.zeros_(self.ph.bias)
        nn.init.zeros_(self.vo.weight); nn.init.zeros_(self.vo.bias)

    def forward(self, x):
        h = self.stem(x)
        if self.recurrent:
            e = self.inj(x)
            for i in range(self.recurrent):
                y = F.relu(self.b1[i](self.c1(h + e)))
                h = F.relu(h + self.b2[i](self.c2(y)))
        else:
            h = self.tower(h)
        p = self.ph(h).reshape(-1, 64, 64).transpose(1, 2).reshape(-1, 4096)
        v = self.vo(self.vh(h))
        return p, v


net = Net(A.ch, A.blocks, A.recurrent).to(dev)
if A.init:
    net.load_state_dict(torch.load(A.init, map_location=dev))
print("params", sum(p.numel() for p in net.parameters()), flush=True)
BS = 1024
steps = A.epochs * (len(tr_idx) // BS)
opt = torch.optim.AdamW(net.parameters(), lr=A.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=A.lr, total_steps=max(steps, 1))


def batch_loss(b, s=0):
    k, fl, inv = SYM[s]
    x, hpb, lg, target = obs_t[b], hp_t[b].float(), legal_t[b], pi_t[b].float()
    if s:
        x = torch.rot90(x, k, dims=(2, 3))
        if fl:
            x = torch.flip(x, dims=[3])
        hpb, lg, target = hpb[:, inv], lg[:, inv], target[:, inv]
    p, v = net(x)
    logits = (hpb + p).masked_fill(~lg, -1e9)
    pl = -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
    vl = F.mse_loss(torch.tanh(hv_t[b] + v[:, 0]), z_t[b])
    return pl, vl


def evaluate():
    net.eval()
    with torch.no_grad():
        tot = [0.0, 0.0]
        for i in range(0, len(val_idx), 4096):
            b = torch.from_numpy(val_idx[i:i + 4096]).to(dev)
            pl, vl = batch_loss(b)
            tot[0] += pl.item() * len(b); tot[1] += vl.item() * len(b)
    net.train()
    return round(tot[0] / len(val_idx), 4), round(tot[1] / len(val_idx), 4)


print("val before", evaluate(), flush=True)
t0 = time.time()
for ep in range(A.epochs):
    perm_tr = rng.permutation(tr_idx)
    for i in range(0, len(perm_tr) - BS + 1, BS):
        b = torch.from_numpy(perm_tr[i:i + BS]).to(dev)
        pl, vl = batch_loss(b, int(rng.integers(8)))
        opt.zero_grad(); (pl + vl).backward(); opt.step(); sched.step()
    print(f"epoch {ep} val {evaluate()} {time.time() - t0:.0f}s", flush=True)

net.eval().cpu()
torch.save(net.state_dict(), A.out + ".pt")
tmp = A.out + ".net.onnx"
torch.onnx.export(net, torch.zeros(2, 16, 8, 8), tmp, input_names=["observation"],
                  output_names=["rp", "rv"], opset_version=17,
                  dynamic_axes={"observation": {0: "batch"}, "rp": {0: "batch"}, "rv": {0: "batch"}},
                  dynamo=False)
nm, hm = onnx.load(tmp), onnx.load(A.heur)
ren = lambda s: s if s in ("observation", "") else "n_" + s
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
graph = helper.make_graph(keep + list(nm.graph.node) + extra, "hybrid", list(g.input), list(g.output),
                          list(g.initializer) + list(nm.graph.initializer))
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, A.out)
import subprocess, sys  # noqa: E401
subprocess.run([sys.executable, str(__import__("pathlib").Path(__file__).with_name("fixconst.py")), A.out], check=True)
ort.set_default_logger_severity(3)
s = ort.InferenceSession(A.out, providers=["CPUExecutionProvider"])
x8 = obs[:8]
t1 = time.perf_counter()
for _ in range(50):
    s.run(None, {"observation": x8})
print(f"saved {A.out}  batch-8 latency {(time.perf_counter() - t1) / 50 * 1000:.1f} ms", flush=True)
