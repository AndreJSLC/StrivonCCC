"""CPU seconds each side spends per Level 4 game at the referee's settings (100 sims, 1 thread).

usage: cputime.py games model.onnx [model.onnx ...]
Each model plays itself; prints mean / max CPU seconds per side per game and actions per side.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "env"))
import numpy as np  # noqa: E402

from colony_one import constants, mcts  # noqa: E402
from colony_one.env import ColonyOneEnv  # noqa: E402
from colony_one.onnx_runner import OnnxEvaluator  # noqa: E402

SCENE = np.asarray(json.load(open(ROOT.parent / "scenes/level-4_scene.json")), dtype=np.float32)
games = int(sys.argv[1])
for path in sys.argv[2:]:
    ev = OnnxEvaluator(Path(path))
    cfg = mcts.MCTSConfig(simulations=100)
    per_side, acts = [], []
    for g in range(games):
        env = ColonyOneEnv(scene=SCENE)
        rng = np.random.default_rng(g)
        cpu = {constants.CYAN: 0.0, constants.MAGENTA: 0.0}
        n = {constants.CYAN: 0, constants.MAGENTA: 0}
        while not env.terminated:
            team = env.current_team
            t = time.process_time()
            a = mcts.search_action(env.board, team, env.actions_left, env.round_number, ev, cfg, rng=rng)
            cpu[team] += time.process_time() - t
            n[team] += 1
            env.step(a)
        per_side += list(cpu.values())
        acts += list(n.values())
    print(f"{Path(path).name}: CPU s/side/game mean {np.mean(per_side):.1f} max {max(per_side):.1f}; "
          f"actions/side mean {np.mean(acts):.0f} max {max(acts)}", flush=True)
