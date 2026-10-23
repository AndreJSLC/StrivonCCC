"""Head-to-head on the real L4 scene with referee settings (A and B each play both colours).

usage: distill_eval.py A.onnx B.onnx pairs [sims] [procs]   -> prints A's win rate
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "env"))
import numpy as np  # noqa: E402

from colony_one import constants, mcts  # noqa: E402
from colony_one.env import ColonyOneEnv  # noqa: E402
from colony_one.onnx_runner import OnnxEvaluator  # noqa: E402

SCENE = np.asarray(json.load(open(HERE.parent / "scenes/level-4_scene.json")), dtype=np.float32)
_evs = None


def _init(a, b):
    global _evs
    _evs = (OnnxEvaluator(Path(a)), OnnxEvaluator(Path(b)))


def play(job):
    seed, a_is_cyan, sims = job
    env = ColonyOneEnv(scene=SCENE)
    cfg = mcts.MCTSConfig(simulations=sims)
    rng = np.random.default_rng(seed)
    while not env.terminated:
        a_turn = (env.current_team == constants.CYAN) == bool(a_is_cyan)
        ev = _evs[0] if a_turn else _evs[1]
        env.step(mcts.search_action(env.board, env.current_team, env.actions_left,
                                    env.round_number, ev, cfg, rng=rng))
    return (env.winner == constants.CYAN) == bool(a_is_cyan), a_is_cyan


def evaluate(a, b, pairs, sims=100, procs=14, seed=12345):
    jobs = [(seed + g, c, sims) for g in range(pairs) for c in (1, 0)]
    with mp.Pool(procs, initializer=_init, initargs=(str(a), str(b))) as pool:
        res = pool.map(play, jobs, chunksize=1)
    wins = sum(r[0] for r in res)
    wc = sum(r[0] for r in res if r[1])
    return wins, len(res), wc


if __name__ == "__main__":
    a, b, pairs = sys.argv[1], sys.argv[2], int(sys.argv[3])
    sims = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 14
    w, n, wc = evaluate(a, b, pairs, sims, procs)
    print(f"{a} vs {b}: {w}/{n} = {w / n:.1%} (as cyan {wc}/{n // 2})")
