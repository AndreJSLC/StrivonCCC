"""Self-play data generation on the real L4 scene with the referee MCTS.

usage: distill_selfplay.py model.onnx out.npz games [sims] [procs] [seed]
Each record: observation, visit distribution (float16), result z and root search value q (side to move).
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "env"))
import numpy as np  # noqa: E402

from colony_one import constants, mcts  # noqa: E402
from colony_one.env import ColonyOneEnv  # noqa: E402
from colony_one.onnx_runner import OnnxEvaluator  # noqa: E402

SCENE = np.asarray(json.load(open(HERE.parent / "scenes/level-4_scene.json")), dtype=np.float32)

_ev = None


def _init(model):
    global _ev
    import onnxruntime as ort  # noqa: F401
    _ev = OnnxEvaluator(Path(model))


def play_game(args):
    seed, sims, temps = args
    env = ColonyOneEnv(scene=SCENE)
    cfg = mcts.MCTSConfig(simulations=sims, warmup_temperatures=temps)
    rng = np.random.default_rng(seed)
    obs, pis, sides, qs = [], [], [], []
    while not env.terminated:
        team, al, rn = env.current_team, env.actions_left, env.round_number
        o = env.observation()
        root = mcts.run(env.board, team, al, rn, _ev, cfg, rng=rng)
        t = mcts.warmup_temperature(mcts.move_number(team, al, rn), cfg.warmup_temperatures,
                                    cfg.warmup_step_actions)
        a = mcts.select_action_index(root, t, rng=rng)
        obs.append(o)
        pis.append(mcts.policy_target(root).astype(np.float16))
        sides.append(team)
        qs.append(sum(root.W.values()) / max(1, sum(root.N.values())))
        env.step(a)
    z = np.array([1.0 if s == env.winner else -1.0 for s in sides], dtype=np.float32)
    return np.stack(obs), np.stack(pis), z, env.winner, np.array(qs, dtype=np.float32)


def generate(model, games, sims=100, procs=14, seed=0, temps=(1.0, 1.0, 0.5, 0.5)):
    jobs = [(seed * 100003 + g, sims, temps) for g in range(games)]
    with mp.Pool(procs, initializer=_init, initargs=(str(model),)) as pool:
        res = pool.map(play_game, jobs, chunksize=1)
    O = np.concatenate([r[0] for r in res])
    P = np.concatenate([r[1] for r in res])
    Z = np.concatenate([r[2] for r in res])
    Q = np.concatenate([r[4] for r in res])
    cyan = sum(r[3] == constants.CYAN for r in res)
    return O, P, Z, cyan, Q


if __name__ == "__main__":
    model, out, games = sys.argv[1], sys.argv[2], int(sys.argv[3])
    sims = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 14
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    t0 = time.perf_counter()
    O, P, Z, cyan, Q = generate(model, games, sims, procs, seed)
    np.savez_compressed(out, obs=O, pi=P, z=Z, q=Q)
    print(f"{games} games, {len(Z)} positions, cyan wins {cyan}/{games}, {time.perf_counter() - t0:.1f}s")
