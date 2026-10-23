"""Continuous self-play on the Level 4 scene, feeding the training loop.

Same .npz keys and move-selection schedule as selfplay.py (obs, legal bits, visit policy pi,
outcome z, root search value q), with these changes:
  * every worker seeds from fresh OS entropy, so no two workers, runs or pods replay the same
    games;
  * workers never stop: they pick up a new net as soon as models/current.txt names one, and
    write a file every `--games` games;
  * playout-cap randomization (KataGo): with --pcr P, each move is searched with --sims and
    recorded with probability P, otherwise searched with --fast sims and not recorded;
  * --gpu evaluates each worker's leaf batches with onnxruntime's CUDA provider.

usage: selfplay_pcr.py models_dir out_dir --workers 15 --games 2 --sims 300
                          [--pcr 0.25 --fast 60] [--gpu]
"""
import argparse
import json
import os
import secrets
import sys
import time
from multiprocessing import Process
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "env"))
SCENE = HERE.parent / "scenes" / "level-4_scene.json"


class CudaEvaluator:
    """Same contract as colony_one.onnx_runner.OnnxEvaluator, on the GPU."""

    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def __call__(self, planes):
        import numpy as np
        planes = np.asarray(planes, dtype=np.float32)
        if planes.ndim == 3:
            planes = planes[None]
        p, v = self.sess.run(None, {"observation": planes})
        p = np.nan_to_num(np.asarray(p, dtype=np.float32).reshape(len(planes), -1), nan=0.0, posinf=0.0, neginf=0.0)
        v = np.clip(np.nan_to_num(np.asarray(v, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=-1.0), -1, 1)
        return p, v


def worker(a, wid, run):
    import numpy as np
    from colony_one import encoding, mcts
    from colony_one.env import ColonyOneEnv
    from colony_one.onnx_runner import OnnxEvaluator
    models, out = Path(a.models), Path(a.out)
    scene = np.asarray(json.load(open(SCENE)), dtype=np.float32)
    full_cfg = mcts.MCTSConfig(simulations=a.sims)
    fast_cfg = mcts.MCTSConfig(simulations=a.fast)
    rng = np.random.default_rng()  # fresh OS entropy
    name, ev, k = None, None, 0
    while True:
        try:
            cur = (models / "current.txt").read_text().strip()
            if cur != name:
                ev = CudaEvaluator(models / f"{cur}.onnx") if a.gpu else OnnxEvaluator(models / f"{cur}.onnx")
                name = cur
        except Exception as e:  # net not pushed yet, or mid-copy
            print(f"worker {wid}: waiting for net ({e})", flush=True)
            time.sleep(10)
            continue
        O, L, Pi, Z, Q = [], [], [], [], []
        for _ in range(a.games):
            env = ColonyOneEnv(scene=scene)
            obs, legal, pis, teams = [], [], [], []
            while not env.terminated:
                team = env.current_team
                full = a.pcr >= 1.0 or rng.random() < a.pcr
                root = mcts.run(env.board, team, env.actions_left, env.round_number, ev,
                                full_cfg if full else fast_cfg, rng=rng)
                if full:
                    Q.append(sum(root.W.values()) / max(1, sum(root.N.values())))
                    pis.append(mcts.policy_target(root).astype(np.float32))
                    obs.append(env.observation().astype(np.float32))
                    legal.append(np.packbits(env.legal_mask()))
                    teams.append(team)
                mv = mcts.move_number(team, env.actions_left, env.round_number)
                temp = 1.0 if mv <= 8 else (0.5 if mv <= 16 else 0.15)
                idx = mcts.select_action_index(root, temp, rng=rng)
                env.step(encoding.decode_action(idx, team, env.board))
            for o, l, p, t in zip(obs, legal, pis, teams):
                O.append(o)
                L.append(l)
                Pi.append(p)
                Z.append(1.0 if env.winner == t else -1.0)
        if not Z:
            continue
        k += 1
        dst = out / f"f53_{name}_{run}_{wid}_{k}.npz"
        tmp = out / f".{dst.name}.tmp"
        with open(tmp, "wb") as fh:  # file object: numpy must not append its own suffix
            np.savez_compressed(fh, obs=np.array(O), legal=np.array(L),
                                pi=np.array(Pi, dtype=np.float16), z=np.array(Z, dtype=np.float32),
                                q=np.array(Q, dtype=np.float32))
        os.replace(tmp, dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--games", type=int, default=2, help="games per output file")
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--pcr", type=float, default=1.0, help="share of moves searched with --sims and recorded")
    ap.add_argument("--fast", type=int, default=60)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    run = secrets.token_hex(3)
    print(f"run {run}: {vars(a)}", flush=True)
    procs = [Process(target=worker, args=(a, w, run)) for w in range(a.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
