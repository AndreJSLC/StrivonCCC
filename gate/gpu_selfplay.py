"""GPU-batched self-play.

The referee search is unchanged (colony_one.mcts.run with MCTSConfig(simulations=S)). Only the
evaluator moves: each process runs K games at once in threads. Every search thread hands its leaf
batch (up to 8 positions) to one batcher thread, which stacks all waiting requests from all K games
into one big batch and runs it on the GPU with onnxruntime's CUDA provider. So the GPU, otherwise
idle during CPU self-play, does the network work for many games at once, and the CPU cores only do
the tree search and the rules.

Output: the same format as selfplay.py (obs, legal packbits, pi fp16, z, q), one file per process,
written under a temp name and then renamed.

usage: gpu_selfplay.py model.onnx out_prefix procs games_per_proc concurrent_games sims [seed]
"""
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "env"))
SCENE = ROOT / "scenes" / "level-4_scene.json"


class GpuBatcher:
    def __init__(self, model_path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_path), so,
                                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.provider = self.sess.get_providers()[0]
        self.cv = threading.Condition()
        self.pending = []
        self.active = 0           # live game threads; a batch runs once all of them are waiting
        self.calls = 0
        self.positions = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def register(self, delta):
        with self.cv:
            self.active += delta
            self.cv.notify_all()

    def __call__(self, planes):
        import numpy as np
        planes = np.asarray(planes, dtype=np.float32)
        if planes.ndim == 3:
            planes = planes[None]
        slot = [planes, threading.Event(), None]
        with self.cv:
            self.pending.append(slot)
            self.cv.notify_all()
        slot[1].wait()
        return slot[2]

    def _loop(self):
        import numpy as np
        while True:
            with self.cv:
                # block (no busy-wait, so the GIL stays with the search threads) until every live
                # game is waiting, or 20 ms pass with something queued
                self.cv.wait_for(lambda: self.pending and len(self.pending) >= self.active, timeout=0.02)
                if not self.pending:
                    continue
                items, self.pending = self.pending, []
            x = np.concatenate([it[0] for it in items])
            p, v = self.sess.run(None, {"observation": x})
            self.calls += 1
            self.positions += len(x)
            o = 0
            for it in items:
                n = len(it[0])
                it[2] = (p[o:o + n], v[o:o + n].reshape(-1))
                o += n
                it[1].set()


def proc_main(args):
    import numpy as np
    from colony_one import encoding, mcts
    from colony_one.env import ColonyOneEnv
    model, out, pid, games, conc, sims, seed = args
    ev = GpuBatcher(model)
    scene = np.asarray(json.load(open(SCENE)), dtype=np.float32)
    cfg = mcts.MCTSConfig(simulations=sims)
    lock = threading.Lock()
    O, L, Pi, Z, Q = [], [], [], [], []
    state = {"started": 0, "done": 0}

    def game_thread(tid):
        try:
            _game_thread(tid)
        finally:
            ev.register(-1)

    def _game_thread(tid):
        rng = np.random.default_rng(seed * 1000003 + pid * 10007 + tid)
        while True:
            with lock:
                if state["started"] >= games:
                    return
                state["started"] += 1
            env = ColonyOneEnv(scene=scene)
            obs, legal, pis, teams, qs = [], [], [], [], []
            while not env.terminated:
                team = env.current_team
                root = mcts.run(env.board, team, env.actions_left, env.round_number, ev, cfg, rng=rng)
                n = sum(root.N.values())
                qs.append(sum(root.W.values()) / n if n else 0.0)
                obs.append(env.observation().astype(np.float32))
                legal.append(np.packbits(env.legal_mask()))
                pis.append(mcts.policy_target(root).astype(np.float16))
                teams.append(team)
                mv = mcts.move_number(team, env.actions_left, env.round_number)
                temp = 1.0 if mv <= 8 else (0.5 if mv <= 16 else 0.15)
                idx = mcts.select_action_index(root, temp, rng=rng)
                env.step(encoding.decode_action(idx, team, env.board))
            with lock:
                for o, l, p, t, q in zip(obs, legal, pis, teams, qs):
                    O.append(o); L.append(l); Pi.append(p); Q.append(q)
                    Z.append(1.0 if env.winner == t else -1.0)
                state["done"] += 1

    t0 = time.time()
    ev.register(conc)
    ths = [threading.Thread(target=game_thread, args=(i,)) for i in range(conc)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    tmp = f"{out}_{pid}.tmp.npz"
    np.savez_compressed(tmp, obs=np.array(O), legal=np.array(L), pi=np.array(Pi, dtype=np.float16),
                        z=np.array(Z, dtype=np.float32), q=np.array(Q, dtype=np.float32))
    os.replace(tmp, f"{out}_{pid}.npz")
    return dict(pid=pid, games=state["done"], positions=len(Z), secs=time.time() - t0,
                provider=ev.provider, calls=ev.calls, avg_batch=ev.positions / max(ev.calls, 1))


if __name__ == "__main__":
    model, out = sys.argv[1], sys.argv[2]
    procs, games, conc, sims = (int(a) for a in sys.argv[3:7])
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=procs) as ex:
        res = list(ex.map(proc_main, [(model, out, p, games, conc, sims, seed) for p in range(procs)]))
    g = sum(r["games"] for r in res)
    wall = time.time() - t0
    print(f"{g} games, {sum(r['positions'] for r in res)} positions in {wall:.0f} s "
          f"= {g / wall * 60:.0f} games/min; provider {res[0]['provider']}, "
          f"avg GPU batch {sum(r['avg_batch'] for r in res) / len(res):.0f}", flush=True)
