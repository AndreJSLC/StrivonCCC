"""Head-to-head on the real Level 4 scene, parallel, both seats.

usage: bench.py games sims workers new.onnx old.onnx
(games = total; half with `new` as Cyan, half as Magenta)
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "env"))
import numpy as np  # noqa: E402

from colony_one import constants, mcts  # noqa: E402
from colony_one.env import ColonyOneEnv  # noqa: E402
from colony_one.onnx_runner import OnnxEvaluator  # noqa: E402

SCENE = np.asarray(json.load(open(ROOT.parent / "scenes/level-4_scene.json")), dtype=np.float32)
_ev = {}


def ev_for(path):
    if path not in _ev:
        _ev[path] = OnnxEvaluator(Path(path))
    return _ev[path]


def play(args):
    new, old, sims, g = args
    new_cyan = g % 2 == 0
    ev_c, ev_m = (ev_for(new), ev_for(old)) if new_cyan else (ev_for(old), ev_for(new))
    env = ColonyOneEnv(scene=SCENE)
    cfg = mcts.MCTSConfig(simulations=sims)
    rng = np.random.default_rng(int(__import__("os").environ.get("SEEDBASE", "1000")) + g)
    while not env.terminated:
        ev = ev_c if env.current_team == constants.CYAN else ev_m
        a = mcts.search_action(env.board, env.current_team, env.actions_left,
                               env.round_number, ev, cfg, rng=rng)
        env.step(a)
    cyan_won = env.winner == constants.CYAN
    return g, int(cyan_won == new_cyan), new_cyan


if __name__ == "__main__":
    games, sims, workers = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    new, old = sys.argv[4], sys.argv[5]
    t0 = time.perf_counter()
    wins = [0, 0]  # as cyan, as magenta
    n = [0, 0]
    with Pool(workers) as pool:
        for g, w, nc in pool.imap_unordered(play, [(new, old, sims, g) for g in range(games)]):
            wins[0 if nc else 1] += w
            n[0 if nc else 1] += 1
    tot = sum(wins)
    print(f"{new} vs {old}: {tot}/{games} = {tot / games:.3f}  "
          f"(cyan {wins[0]}/{n[0]}, magenta {wins[1]}/{n[1]})  {time.perf_counter() - t0:.0f}s")
