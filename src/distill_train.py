"""Supervised / expert-iteration training on stored (obs, visit-dist, z) chunks. Needs PyTorch.

usage: distill_train.py out.onnx steps [glob[,glob...]] [init.pt]
env: DISTILL_LAMBDA (weight of root search value q in the value target, default 0.3), DISTILL_THREADS, DISTILL_LR, DISTILL_C/DISTILL_B (net size)
"""
import os
import glob
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
import torch.nn.functional as F

import distill_net as pn

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(os.environ.get("DISTILL_THREADS", "8")))
LAM = float(os.environ.get("DISTILL_LAMBDA", "0.3"))
K = 64
sq = np.arange(64).reshape(8, 8)
_perms = []
for k in range(4):
    for fl in (False, True):
        g = np.rot90(sq, k)
        g = g[:, ::-1] if fl else g
        _perms.append(g.reshape(64))
PERM = np.stack(_perms)                    # new[s] = old[PERM[g][s]]
INV = np.argsort(PERM, axis=1)             # old square s -> new square INV[g][s]


def load(files):
    O, I, W, Z = [], [], [], []
    for f in files:
        d = np.load(f)
        O.append(d["obs"].astype(np.float16))
        q = d["q"] if "q" in d.files else d["z"]
        Z.append((1 - LAM) * d["z"] + LAM * q)
        pi = d["pi"]
        idx = np.zeros((len(pi), K), np.int16); w = np.zeros((len(pi), K), np.float16)
        for r in range(len(pi)):
            nz = np.nonzero(pi[r])[0]
            if len(nz) > K:
                nz = nz[np.argsort(-pi[r][nz])[:K]]
            idx[r, :len(nz)] = nz; w[r, :len(nz)] = pi[r][nz]
        I.append(idx); W.append(w)
    return np.concatenate(O), np.concatenate(I), np.concatenate(W), np.concatenate(Z)


def batch(O, I, W, Z, ids, perm, inv, aug=True):
    b = len(ids)
    o = O[ids].float(); i = I[ids].long(); w = W[ids].float(); z = Z[ids]
    if aug:
        g = torch.randint(0, 8, (b,), device=dev)
        o = o.reshape(b, 16, 64).gather(2, perm[g][:, None, :].expand(-1, 16, -1)).reshape(b, 16, 8, 8)
        iv = inv[g]                                  # (b,64)
        i = iv.gather(1, i // 64) * 64 + iv.gather(1, i % 64)
    p = torch.zeros(b, 4096, device=dev)
    p.scatter_add_(1, i, w)
    return o, p / p.sum(1, keepdim=True).clamp_min(1e-6), z


def main():
    out, steps = sys.argv[1], int(sys.argv[2])
    pat = sys.argv[3] if len(sys.argv) > 3 else str(HERE / "data" / "teacher_*.npz")
    files = sorted(f for g in pat.split(",") for f in glob.glob(g))
    print(len(files), "files", flush=True)
    O, I, W, Z = load(files)
    nval = min(8000, len(Z) // 10)
    O, I, W, Z = (torch.tensor(a, device=dev) for a in (O, I, W, Z))
    perm = torch.tensor(PERM, device=dev); inv = torch.tensor(INV, device=dev)
    n = len(Z) - nval
    print("train", n, "val", nval, flush=True)
    net = pn.Net(int(os.environ.get("DISTILL_C", "48")), int(os.environ.get("DISTILL_B", "4"))).to(dev)
    if len(sys.argv) > 4:
        net.load_state_dict(torch.load(sys.argv[4]))
    print("params", sum(p.numel() for p in net.parameters()))
    BS = 512
    lr = float(os.environ.get("DISTILL_LR", "2e-3"))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0 = time.perf_counter()
    acc = np.zeros(2)
    for s in range(1, steps + 1):
        net.train()
        ids = torch.randint(0, n, (BS,), device=dev)
        o, p, z = batch(O, I, W, Z, ids, perm, inv)
        logits, v = net(o)
        pl = -(p * F.log_softmax(logits, 1)).sum(1).mean()
        vl = F.mse_loss(v, z)
        loss = pl + vl
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc += (pl.item(), vl.item())
        if s % 1000 == 0 or s == steps:
            net.eval()
            with torch.no_grad():
                vids = torch.arange(n, n + nval, device=dev)
                vp = vv = 0
                for c in range(0, nval, 1000):
                    o, p, z = batch(O, I, W, Z, vids[c:c + 1000], perm, inv, aug=False)
                    lg, v = net(o)
                    vp += -(p * F.log_softmax(lg, 1)).sum(1).sum().item(); vv += ((v - z) ** 2).sum().item()
            print(f"step {s} train pl {acc[0]/1000:.3f} vl {acc[1]/1000:.3f} | val pl {vp/nval:.3f} vl {vv/nval:.3f} "
                  f"| {time.perf_counter()-t0:.0f}s", flush=True)
            acc[:] = 0
    torch.save(net.state_dict(), out + ".pt")
    pn.export(net, out)
    print("exported", out)


if __name__ == "__main__":
    main()
