"""Architecture search on top of the Level 4 heuristic. Trains on a GPU.

Every net is a residual on the hand-built heuristic (models/level4_heuristic.onnx): policy = heuristic logits + net logits,
value = tanh(heuristic pre-tanh value + net value). Heads are zero-initialised, so step 0
is exactly the heuristic. 8x D4 augmentation. Validation is split by data file (whole
self-play workers held out), so it is not leaked through positions of the same game.

usage: arch.py --arch {res,resgp,rec,tf} --out NAME.onnx [--ch 48] [--blocks 4]
                 [--k 6] [--kmin K] [--kmax K] [--gpool 1] [--recall 1]
                 [--d 64] [--layers 4] [--heads 4] [--epochs 6] [--lr 2e-3] [--seed 0]
                 data_glob [data_glob ...]
"""
import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HEUR = os.path.join(ROOT, "models", "level4_heuristic.onnx")

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True, choices=["res", "resgp", "rec", "tf"])
ap.add_argument("--out", required=True)
ap.add_argument("--ch", type=int, default=48)
ap.add_argument("--blocks", type=int, default=4)
ap.add_argument("--k", type=int, default=6, help="rec: iterations at export")
ap.add_argument("--kmin", type=int, default=0, help="rec: min iterations in training (0 = k)")
ap.add_argument("--kmax", type=int, default=0, help="rec: max iterations in training (0 = k)")
ap.add_argument("--gpool", type=int, default=0)
ap.add_argument("--se", type=int, default=0, help="res: squeeze-excitation in every block")
ap.add_argument("--mainheads", type=int, default=0, help="res: src/train2.py's head layout (to warm-start from its .pt)")
ap.add_argument("--block", default="res", choices=["res", "mb"], help="res: full 3x3 blocks or inverted residuals")
ap.add_argument("--t", type=int, default=3, help="mb: expansion factor")
ap.add_argument("--fp16w", type=int, default=0, help="store big weights as float16 + Cast (fits more under 5 MiB)")
ap.add_argument("--init", default=None, help="warm-start checkpoint (.pt); missing keys keep their init")
ap.add_argument("--bil", type=int, default=0, help="res: bilinear from-to policy head of this width")
ap.add_argument("--recall", type=int, default=1)
ap.add_argument("--norm", default="none", choices=["none", "bn", "ln"], help="rec: normalisation")
ap.add_argument("--pre", type=int, default=0, help="rec: unshared residual blocks before the loop")
ap.add_argument("--conv_ch", type=int, default=0, help="tf: conv stem width (0 = raw planes)")
ap.add_argument("--conv_blocks", type=int, default=0, help="tf: residual blocks in the conv stem")
ap.add_argument("--d", type=int, default=64)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--vw", type=float, default=1.0, help="value loss weight")
ap.add_argument("--qmix", type=float, default=0.0, help="value target = (1-qmix)*z + qmix*root q")
ap.add_argument("--bs", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--valfiles", type=int, default=2)
ap.add_argument("data", nargs="+")
A = ap.parse_args()
torch.manual_seed(A.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
files = sorted({f for g in A.data for f in glob.glob(g)})
t_start = time.time()

# ------------------------------------------------------------------ data + heuristic cache
key = hashlib.md5(("|".join(files) + str([os.path.getsize(f) for f in files])).encode()).hexdigest()[:12]
cache = os.path.join(ROOT, "cache", f"data_{key}.npz")
if os.path.exists(cache):
    d = np.load(cache)
    obs, legal, pi, z, hp, hv, fid = (d[k] for k in ("obs", "legal", "pi", "z", "hp", "hv", "fid"))
    q = d["q"] if "q" in d.files else z.copy()
    legal = np.unpackbits(legal, axis=1)[:, :4096].astype(bool)
else:
    O, L, P, Z, Q, FID = [], [], [], [], [], []
    for i, f in enumerate(files):
        d = np.load(f)
        O.append(d["obs"].astype(np.float32)); L.append(d["legal"]); P.append(d["pi"].astype(np.float16))
        Z.append(d["z"].astype(np.float32)); FID.append(np.full(len(d["z"]), i, np.int32))
        Q.append(d["q"].astype(np.float32) if "q" in d.files else d["z"].astype(np.float32))  # no q: use z
    obs, legal_p, pi, z, q, fid = map(np.concatenate, (O, L, P, Z, Q, FID))
    hm = onnx.load(HEUR)
    hm.graph.output.append(helper.make_tensor_value_info("v1", onnx.TensorProto.FLOAT, ["batch", 1]))
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(hm.SerializeToString(), so, providers=["CPUExecutionProvider"])
    HP, HV = [], []
    for i in range(0, len(obs), 2048):
        p, v = sess.run(["policy", "v1"], {"observation": obs[i:i + 2048]})
        HP.append(p.astype(np.float16)); HV.append(v[:, 0].astype(np.float32))
    hp, hv = np.concatenate(HP), np.concatenate(HV)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, obs=obs, legal=legal_p, pi=pi, z=z, q=q, hp=hp, hv=hv, fid=fid)
    legal = np.unpackbits(legal_p, axis=1)[:, :4096].astype(bool)
nfiles = int(fid.max()) + 1
rng = np.random.default_rng(1234)
val_files = set(rng.choice(nfiles, size=min(A.valfiles, max(nfiles - 1, 1)), replace=False).tolist())
is_val = np.isin(fid, list(val_files))
val_idx, tr_idx = np.nonzero(is_val)[0], np.nonzero(~is_val)[0]
print(f"positions {len(z)} files {nfiles} train {len(tr_idx)} val {len(val_idx)} "
      f"(held-out files {sorted(val_files)})  load {time.time() - t_start:.0f}s", flush=True)

G = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
# observation values are multiples of 1/8 in [0, 1], so float16 holds them exactly
obs_t, legal_t, pi_t, hp_t, hv_t = G(obs.astype(np.float16)), G(legal), G(pi), G(hp), G(hv)
# value target: (1 - qmix) * game result + qmix * root search value (q = z where the data has no q)
z_t = G(((1.0 - A.qmix) * z + A.qmix * q).astype(np.float32))
zr_t = G(z.astype(np.float32))  # validation always scores the value head against the real result
print(f"value target: qmix {A.qmix}  mean|z-q| {float(np.abs(z - q).mean()):.3f}", flush=True)

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


# ------------------------------------------------------------------ architectures
def zero_(m):
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class ConvHeads(nn.Module):
    def __init__(self, ch, bil=0):
        super().__init__()
        self.bil = bil
        if bil:  # Lc0-style bilinear "attention policy": logit[from, to] = q_from . k_to
            self.pq = nn.Conv2d(ch, bil, 1)
            self.pk = nn.Conv2d(ch, bil, 1)
            zero_(self.pq)
        else:
            self.ph = nn.Conv2d(ch, 64, 1)
            zero_(self.ph)
        self.vc = nn.Conv2d(ch, 8, 1)
        self.v1 = nn.Linear(512 + ch, 64)
        self.vo = nn.Linear(64, 1)
        zero_(self.vo)

    def forward(self, h):
        if self.bil:
            q = self.pq(h).reshape(-1, self.bil, 64).transpose(1, 2)            # (B,64from,d)
            k = self.pk(h).reshape(-1, self.bil, 64)                            # (B,d,64to)
            p = (torch.matmul(q, k) * (1.0 / math.sqrt(self.bil))).reshape(-1, 4096)
        else:
            p = self.ph(h).reshape(-1, 64, 64).transpose(1, 2).reshape(-1, 4096)  # [from, to]
        g = h.mean(dim=(2, 3))
        v = F.relu(self.vc(h)).reshape(-1, 512)
        v = self.vo(F.relu(self.v1(torch.cat([v, g], 1))))
        return p, v


class MainHeads(nn.Module):
    """src/train2.py's heads (ph, vh, vo), so its .pt checkpoints load for warm starts."""

    def __init__(self, ch, bil=0, keep_ph=False):
        super().__init__()
        self.bil = bil
        self.keep_ph = keep_ph or not bil
        if bil:  # bilinear policy on top of a warm-started trunk and value head
            self.pq = nn.Conv2d(ch, bil, 1)
            self.pk = nn.Conv2d(ch, bil, 1)
            zero_(self.pq)
        if self.keep_ph:  # with bil: keep the loaded 1x1 head and ADD the bilinear one (step 0 = loaded net)
            self.ph = nn.Conv2d(ch, 64, 1)
            zero_(self.ph)
        self.vh = nn.Sequential(nn.Conv2d(ch, 4, 1), nn.ReLU(), nn.Flatten(), nn.Linear(256, 64), nn.ReLU())
        self.vo = nn.Linear(64, 1)
        zero_(self.vo)

    def forward(self, h):
        p = 0
        if self.bil:
            q = self.pq(h).reshape(-1, self.bil, 64).transpose(1, 2)
            k = self.pk(h).reshape(-1, self.bil, 64)
            p = (torch.matmul(q, k) * (1.0 / math.sqrt(self.bil))).reshape(-1, 4096)
        if self.keep_ph:
            p = p + self.ph(h).reshape(-1, 64, 64).transpose(1, 2).reshape(-1, 4096)
        return p, self.vo(self.vh(h))


class GPool(nn.Module):
    """KataGo-style global pooling bias: mean and max over the board -> per-channel bias.
    Zero-initialised, so adding it to a trained net leaves the net unchanged at step 0."""

    def __init__(self, ch):
        super().__init__()
        self.fc = nn.Linear(2 * ch, ch)
        zero_(self.fc)
        self.ch = ch

    def forward(self, y):
        g = torch.cat([y.mean(dim=(2, 3)), y.amax(dim=(2, 3))], 1)
        return y + self.fc(g).reshape(-1, self.ch, 1, 1)


class SE(nn.Module):
    """Squeeze-and-excitation (as in Lc0's SE-ResNets): global mean -> per-channel gate and bias."""

    def __init__(self, ch, r=4):
        super().__init__()
        self.f1 = nn.Linear(ch, ch // r)
        self.f2 = nn.Linear(ch // r, 2 * ch)
        self.ch = ch

    def forward(self, y):
        s = self.f2(F.relu(self.f1(y.mean(dim=(2, 3)))))
        w, b = s[:, :self.ch], s[:, self.ch:]
        return y * torch.sigmoid(w).reshape(-1, self.ch, 1, 1) + b.reshape(-1, self.ch, 1, 1)


class Block(nn.Module):
    def __init__(self, ch, gpool, se=0):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(ch)
        self.gp = GPool(ch) if gpool else None
        self.se = SE(ch) if se else None

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        if self.gp is not None:
            y = self.gp(y)
        y = self.b2(self.c2(y))
        if self.se is not None:
            y = self.se(y)
        return F.relu(x + y)


class MBBlock(nn.Module):
    """MobileNetV2-style inverted residual: 1x1 expand to t*ch, 3x3 depthwise, 1x1 project, skip.
    Roughly half the multiply-adds of a full 3x3 residual block at the same width (t=3)."""

    def __init__(self, ch, t=3):
        super().__init__()
        e = ch * t
        self.pw1 = nn.Conv2d(ch, e, 1, bias=False); self.b1 = nn.BatchNorm2d(e)
        self.dw = nn.Conv2d(e, e, 3, padding=1, groups=e, bias=False); self.b2 = nn.BatchNorm2d(e)
        self.pw2 = nn.Conv2d(e, ch, 1, bias=False); self.b3 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.pw1(x)))
        y = F.relu(self.b2(self.dw(y)))
        return x + self.b3(self.pw2(y))


class ResNet(nn.Module):
    def __init__(self, ch, blocks, gpool, se=0, bil=0, mainheads=0, block="res", t=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(16, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        if block == "mb":
            self.tower = nn.Sequential(*[MBBlock(ch, t) for _ in range(blocks)], nn.ReLU())
        else:
            self.tower = nn.Sequential(*[Block(ch, gpool, se) for _ in range(blocks)])
        self.heads = MainHeads(ch, bil, keep_ph=(mainheads == 2)) if mainheads else ConvHeads(ch, bil)

    def forward(self, x, k=None):
        return self.heads(self.tower(self.stem(x)))


class RecNet(nn.Module):
    """One weight-shared block applied k times, with input recall (the board is concatenated back
    in at every iteration, as in Deep Thinking nets). Normalisation variants:
      none: ReZero-style learned step size, no norm; the same weights run at any depth.
      bn:   per-iteration BatchNorm (shared convs, iteration-specific statistics).
      ln:   one LayerNorm over (C, 8, 8) shared by all iterations; depth-agnostic.
    pre: unshared residual blocks before the recurrence (a hybrid trunk)."""

    def __init__(self, ch, recall, gpool, norm="none", kmax=8, pre=0, bil=0):
        super().__init__()
        self.norm = norm
        self.stem = nn.Sequential(nn.Conv2d(16, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.pre = nn.Sequential(*[Block(ch, 0) for _ in range(pre)])
        self.recall = recall
        self.c1 = nn.Conv2d(ch + (16 if recall else 0), ch, 3, padding=1, bias=(norm == "none"))
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=(norm == "none"))
        self.gp = GPool(ch) if gpool else None
        if norm == "bn":
            self.n1 = nn.ModuleList([nn.BatchNorm2d(ch) for _ in range(kmax)])
            self.n2 = nn.ModuleList([nn.BatchNorm2d(ch) for _ in range(kmax)])
        elif norm == "ln":
            self.n1 = nn.LayerNorm([ch, 8, 8])
            self.n2 = nn.LayerNorm([ch, 8, 8])
        # heads are zero-initialised, so a non-zero step size is safe at step 0
        self.alpha = nn.Parameter(torch.full((1,), 0.2))
        self.heads = ConvHeads(ch, bil)

    def forward(self, x, k):
        h = self.pre(self.stem(x))
        for i in range(k):
            u = torch.cat([h, x], 1) if self.recall else h
            if self.norm == "none":
                y = F.relu(self.c1(u))
                if self.gp is not None:
                    y = self.gp(y)
                h = h + self.alpha * self.c2(y)
            else:
                n1 = self.n1[i] if self.norm == "bn" else self.n1
                n2 = self.n2[i] if self.norm == "bn" else self.n2
                y = F.relu(n1(self.c1(u)))
                if self.gp is not None:
                    y = self.gp(y)
                h = F.relu(h + n2(self.c2(y)))
        return self.heads(F.relu(h) if self.norm == "none" else h)


class TfNet(nn.Module):
    """Small transformer over the 64 squares. Policy is a bilinear from x to score."""

    def __init__(self, d, layers, heads, conv_ch=0, conv_blocks=0):
        super().__init__()
        self.d, self.h = d, heads
        self.conv_ch = conv_ch
        if conv_ch:  # conv stem + residual blocks, then attention over the 64 squares (a hybrid)
            self.cstem = nn.Sequential(nn.Conv2d(16, conv_ch, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(conv_ch), nn.ReLU(),
                                       *[Block(conv_ch, 0) for _ in range(conv_blocks)])
        self.emb = nn.Linear(conv_ch or 16, d)
        self.pos = nn.Parameter(torch.randn(1, 64, d) * 0.02)
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.wq = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.wk = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.wv = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.proj = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.f1 = nn.ModuleList([nn.Linear(d, 2 * d) for _ in range(layers)])
        self.f2 = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.pq = nn.Linear(d, d)
        self.pk = nn.Linear(d, d)
        self.v1 = nn.Linear(d, 64)
        self.vo = nn.Linear(64, 1)
        zero_(self.pq); zero_(self.vo)

    def forward(self, x, k=None):
        d, H = self.d, self.h
        dh = d // H
        if self.conv_ch:
            t = self.emb(self.cstem(x).reshape(-1, self.conv_ch, 64).transpose(1, 2)) + self.pos
        else:
            t = self.emb(x.reshape(-1, 16, 64).transpose(1, 2)) + self.pos       # (B,64,d)
        for i in range(len(self.wq)):
            u = self.ln1[i](t)
            q = self.wq[i](u).reshape(-1, 64, H, dh).transpose(1, 2)               # (B,H,64,dh)
            kk = self.wk[i](u).reshape(-1, 64, H, dh).transpose(1, 2)
            v = self.wv[i](u).reshape(-1, 64, H, dh).transpose(1, 2)
            a = torch.softmax(torch.matmul(q, kk.transpose(2, 3)) * (1.0 / math.sqrt(dh)), dim=-1)
            o = torch.matmul(a, v).permute(0, 2, 1, 3).reshape(-1, 64, d)
            t = t + self.proj[i](o)
            t = t + self.f2[i](F.relu(self.f1[i](self.ln2[i](t))))
        t = self.lnf(t)
        p = torch.matmul(self.pq(t), self.pk(t).transpose(1, 2)) * (1.0 / math.sqrt(d))  # [from, to]
        v = self.vo(F.relu(self.v1(t.mean(dim=1))))
        return p.reshape(-1, 4096), v


if A.arch in ("res", "resgp"):
    net = ResNet(A.ch, A.blocks, A.arch == "resgp" or A.gpool, A.se, A.bil, A.mainheads, A.block, A.t)
elif A.arch == "rec":
    net = RecNet(A.ch, A.recall, A.gpool, A.norm, max(A.kmax, A.k), A.pre, A.bil)
else:
    net = TfNet(A.d, A.layers, A.heads, A.conv_ch, A.conv_blocks)
if A.init:
    sd = torch.load(A.init, map_location="cpu")
    sd = {("heads." + k if k.split(".")[0] in ("ph", "vh", "vo") else k): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print("warm start from", A.init, "missing", missing, "unexpected", unexpected, flush=True)
    if A.arch in ("res", "resgp"):
        # net growing: tower blocks absent from the checkpoint start as exact identities
        # (final BN gamma = beta = 0, so relu(x + 0) = x for the non-negative block input)
        grown = sorted({int(k.split(".")[1]) for k in missing if k.startswith("tower.") and ".b2." in k})
        for i in grown:
            nn.init.zeros_(net.tower[i].b2.weight); nn.init.zeros_(net.tower[i].b2.bias)
        if grown:
            print("grown blocks initialised as identity:", grown, flush=True)
net = net.to(dev)
kmin = A.kmin or A.k
kmax = A.kmax or A.k
print("arch", A.arch, "params", sum(p.numel() for p in net.parameters()), flush=True)

steps = A.epochs * (len(tr_idx) // A.bs)
opt = torch.optim.AdamW(net.parameters(), lr=A.lr, weight_decay=A.wd)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=A.lr, total_steps=max(steps, 1), pct_start=0.15)


def batch_loss(b, s=0, k=None, zt=None):
    zt = z_t if zt is None else zt
    kk_, fl, inv = SYM[s]
    x, hpb, lg, target = obs_t[b].float(), hp_t[b].float(), legal_t[b], pi_t[b].float()
    if s:
        x = torch.rot90(x, kk_, dims=(2, 3))
        if fl:
            x = torch.flip(x, dims=[3])
        hpb, lg, target = hpb[:, inv], lg[:, inv], target[:, inv]
    p, v = net(x, k if k is not None else A.k)
    logits = (hpb + p).masked_fill(~lg, -1e9)
    pl = -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
    vl = F.mse_loss(torch.tanh(hv_t[b] + v[:, 0]), zt[b])
    return pl, vl


def evaluate(k=None):
    net.eval()
    tot = [0.0, 0.0]
    with torch.no_grad():
        for i in range(0, len(val_idx), 4096):
            b = torch.from_numpy(val_idx[i:i + 4096]).to(dev)
            pl, vl = batch_loss(b, 0, k, zr_t)
            tot[0] += pl.item() * len(b); tot[1] += vl.item() * len(b)
    net.train()
    return round(tot[0] / len(val_idx), 4), round(tot[1] / len(val_idx), 4)


print("val before (policy CE, value MSE)", evaluate(), flush=True)
trng = np.random.default_rng(A.seed)
step = 0
for ep in range(A.epochs):
    perm = trng.permutation(tr_idx)
    for i in range(0, len(perm) - A.bs + 1, A.bs):
        b = torch.from_numpy(perm[i:i + A.bs]).to(dev)
        k = int(trng.integers(kmin, kmax + 1)) if A.arch == "rec" else None
        pl, vl = batch_loss(b, int(trng.integers(8)), k)
        opt.zero_grad(set_to_none=True)
        (pl + A.vw * vl).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step(); step += 1
    print(f"epoch {ep} val {evaluate()} {time.time() - t_start:.0f}s", flush=True)
final_val = evaluate()
val_by_k = {}
if A.arch == "rec":
    depth_cap = len(net.n1) if A.norm == "bn" else 10 ** 6  # per-iteration BN has no stats past kmax
    for k in sorted({kmin, A.k, kmax, kmax + 2, kmax + 4}):
        if k > depth_cap:
            continue
        val_by_k[k] = evaluate(k)
        print(f"val at k={k}: {val_by_k[k]}", flush=True)
    print("alpha", float(net.alpha), flush=True)

# ------------------------------------------------------------------ export + merge
net.eval().cpu()
torch.save(net.state_dict(), A.out + ".pt")


class Wrap(nn.Module):
    def __init__(self, m, k):
        super().__init__()
        self.m, self.k = m, k

    def forward(self, x):
        return self.m(x, self.k)


tmp = A.out + ".net.onnx"
torch.onnx.export(Wrap(net, A.k).eval(), torch.zeros(2, 16, 8, 8), tmp, input_names=["observation"],
                  output_names=["rp", "rv"], opset_version=17,
                  dynamic_axes={"observation": {0: "batch"}, "rp": {0: "batch"}, "rv": {0: "batch"}},
                  dynamo=False)
nm, hm = onnx.load(tmp), onnx.load(HEUR)
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
nodes, inits = [], list(g.initializer) + list(nm.graph.initializer)
for nd in keep + list(nm.graph.node) + extra:
    if nd.op_type == "Constant":  # Constant is not on the allow-list: turn it into an initializer
        attr = {a.name: a for a in nd.attribute}
        inits.append(numpy_helper.from_array(numpy_helper.to_array(helper.get_attribute_value(attr["value"])),
                                             nd.output[0]))
    else:
        nodes.append(nd)
graph = helper.make_graph(nodes, "hybrid37", list(g.input), list(g.output), inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, A.out)
os.remove(tmp)

# allow-list + contract check exactly as the referee loads it, then single-thread latency
sys.path.insert(0, os.path.join(ROOT, "src", "env"))
from colony_one.onnx_runner import OnnxEvaluator  # noqa: E402

ev = OnnxEvaluator(A.out, validate=True, num_threads=1)
x8 = obs[:8].astype(np.float32)
p_ref, v_ref = ev(x8)
net.eval()
with torch.no_grad():
    tp, tv = net(torch.from_numpy(x8), A.k)
tp = tp.numpy() + hp[:8].astype(np.float32)
tv = np.tanh(tv.numpy()[:, 0] + hv[:8])
err_p = float(np.abs(p_ref - tp).max())
err_v = float(np.abs(v_ref - tv).max())
for _ in range(20):
    ev(x8)
c0 = time.process_time()
for _ in range(200):
    ev(x8)
lat = (time.process_time() - c0) / 200 * 1000
size_kb = os.path.getsize(A.out) / 1024
print(f"saved {A.out}  {size_kb:.0f} KB  batch-8 CPU {lat:.2f} ms (1 thread)  "
      f"onnx-vs-torch max err policy {err_p:.2e} value {err_v:.2e}  total {time.time() - t_start:.0f}s",
      flush=True)
with open(A.out + ".json", "w") as fh:
    json.dump(dict(args=vars(A), val=final_val, val_by_k={str(k): v for k, v in val_by_k.items()},
                   lat_ms=lat, size_kb=size_kb, err_p=err_p, err_v=err_v), fh)
