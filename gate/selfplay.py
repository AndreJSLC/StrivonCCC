"""Self-play data on the real Level 4 scene. Same format as src/selfplay.py,
plus q = root search value for the side to move (for value-target mixing).

usage: selfplay.py model.onnx out_prefix workers games_per_worker sims [seed]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "env"))
SCENE = ROOT / "scenes" / "level-4_scene.json"


def worker(args):
    import numpy as np
    from colony_one import encoding, mcts
    from colony_one.env import ColonyOneEnv
    from colony_one.onnx_runner import OnnxEvaluator
    model, out, wid, games, sims, seed = args
    ev = OnnxEvaluator(Path(model), num_threads=1)
    scene = np.asarray(json.load(open(SCENE)), dtype=np.float32)
    cfg = mcts.MCTSConfig(simulations=sims)
    rng = np.random.default_rng(seed * 100003 + wid * 7777 + 13)
    O, L, Pi, Z, Q = [], [], [], [], []
    for g in range(games):
        env = ColonyOneEnv(scene=scene)
        obs, legal, pis, teams, qs = [], [], [], [], []
        while not env.terminated:
            team = env.current_team
            root = mcts.run(env.board, team, env.actions_left, env.round_number, ev, cfg, rng=rng)
            n = sum(root.N.values())
            q = sum(root.W.values()) / n if n else 0.0
            obs.append(env.observation().astype(np.float32))
            legal.append(np.packbits(env.legal_mask()))
            pis.append(mcts.policy_target(root).astype(np.float32))
            teams.append(team)
            qs.append(q)
            mv = mcts.move_number(team, env.actions_left, env.round_number)
            temp = 1.0 if mv <= 8 else (0.5 if mv <= 16 else 0.15)
            idx = mcts.select_action_index(root, temp, rng=rng)
            env.step(encoding.decode_action(idx, team, env.board))
        for o, l, p, t, q in zip(obs, legal, pis, teams, qs):
            O.append(o); L.append(l); Pi.append(p); Q.append(q)
            Z.append(1.0 if env.winner == t else -1.0)
    import os
    tmp = f"{out}_{wid}.tmp.npz"  # write then rename, so collectors never see a partial file
    np.savez_compressed(tmp, obs=np.array(O), legal=np.array(L),
                        pi=np.array(Pi, dtype=np.float16), z=np.array(Z, dtype=np.float32),
                        q=np.array(Q, dtype=np.float32))
    os.replace(tmp, f"{out}_{wid}.npz")
    return len(Z)


if __name__ == "__main__":
    model, out, workers, games, sims = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        n = sum(ex.map(worker, [(model, out, w, games, sims, seed) for w in range(workers)]))
    print("positions", n)
