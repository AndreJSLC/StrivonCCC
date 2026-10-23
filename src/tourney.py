"""Round-robin between ONNX models on random mirrored 6-piece scenes (Level 4 proxy).

usage: tourney.py games sims model1.onnx model2.onnx ...
"""
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "env"))
import numpy as np  # noqa: E402

from colony_one import constants, mcts  # noqa: E402
from colony_one.env import ColonyOneEnv  # noqa: E402
from colony_one.onnx_runner import OnnxEvaluator  # noqa: E402

BASE = np.asarray(json.load(open(Path(__file__).parent / "env/colony_one/scene.json")), dtype=np.float32)


def random_scene(rng):
    s = np.zeros_like(BASE)
    s[12:] = BASE[12:]
    cells = rng.choice(24, size=6, replace=False)  # rows 5-7
    for p, c in enumerate(cells):
        r, col = 5 + c // 8, c % 8
        s[p, r, col] = 1.0
        s[6 + p, 7 - r, 7 - col] = 1.0
    return s


def play(ev_c, ev_m, scene, sims, seed):
    env = ColonyOneEnv(scene=scene)
    cfg = mcts.MCTSConfig(simulations=sims)
    rng = np.random.default_rng(seed)
    while not env.terminated:
        ev = ev_c if env.current_team == constants.CYAN else ev_m
        a = mcts.search_action(env.board, env.current_team, env.actions_left,
                               env.round_number, ev, cfg, rng=rng)
        env.step(a)
    return env.winner


games, sims = int(sys.argv[1]), int(sys.argv[2])
paths = sys.argv[3:]
evs = [OnnxEvaluator(Path(p)) for p in paths]
score = {p: 0 for p in paths}
t0 = time.perf_counter()
for i, j in itertools.combinations(range(len(paths)), 2):
    w = 0
    for g in range(games):
        rng = np.random.default_rng(g)
        scene = random_scene(rng)
        for swap in (0, 1):
            c, m = (j, i) if swap else (i, j)
            win = play(evs[c], evs[m], scene, sims, 7919 * g + swap)
            winner = c if win == constants.CYAN else m
            score[paths[winner]] += 1
            w += winner == i
    print(f"{paths[i]} vs {paths[j]}: {w}/{2 * games}", flush=True)
for p, s in sorted(score.items(), key=lambda kv: -kv[1]):
    print(f"{s:4d}  {p}")
print(f"{time.perf_counter() - t0:.1f}s")
