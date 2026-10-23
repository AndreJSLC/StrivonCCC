"""A vs B on the real Level 4 scene, both sides, 100 sims, parallel.

usage: duel.py a.onnx b.onnx [pairs] [sims]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "env"))

SCENE = HERE.parent / "scenes" / "level-4_scene.json"


def play(args):
    import numpy as np
    from colony_one import constants, mcts
    from colony_one.env import ColonyOneEnv
    from colony_one.onnx_runner import OnnxEvaluator
    a, b, a_is_cyan, seed, sims = args
    scene = np.asarray(json.load(open(SCENE)), dtype=np.float32)
    ea, eb = OnnxEvaluator(Path(a)), OnnxEvaluator(Path(b))
    env = ColonyOneEnv(scene=scene)
    cfg = mcts.MCTSConfig(simulations=sims)
    rng = np.random.default_rng(seed)
    while not env.terminated:
        cyan_turn = env.current_team == constants.CYAN
        ev = ea if cyan_turn == a_is_cyan else eb
        act = mcts.search_action(env.board, env.current_team, env.actions_left,
                                 env.round_number, ev, cfg, rng=rng)
        env.step(act)
    return (env.winner == constants.CYAN) == a_is_cyan, a_is_cyan, env.action


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    sims = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    import os
    off = int(os.environ.get("DUEL_SEED", "0"))
    jobs = [(a, b, side == 0, off + 1000 * g + side, sims) for g in range(pairs) for side in (0, 1)]
    import os
    with ProcessPoolExecutor(max_workers=int(os.environ.get("DUEL_WORKERS", "8"))) as ex:
        res = list(ex.map(play, jobs))
    wins = sum(r[0] for r in res)
    wc = sum(r[0] for r in res if r[1])
    wm = sum(r[0] for r in res if not r[1])
    avg = sum(r[2] for r in res) / len(res)
    print(f"{Path(a).name} vs {Path(b).name}: {wins}/{len(res)} = {wins / len(res):.0%}"
          f"  (as cyan {wc}/{pairs}, as magenta {wm}/{pairs}, avg {avg:.0f} actions)")
