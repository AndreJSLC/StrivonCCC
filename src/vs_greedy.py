"""Play model (Cyan, referee MCTS 60 sims) vs GreedyAgent (Magenta)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "env"))
import numpy as np  # noqa: E402

from colony_one import constants, mcts  # noqa: E402
from colony_one.baselines import GreedyAgent  # noqa: E402
from colony_one.env import ColonyOneEnv  # noqa: E402
from colony_one.onnx_runner import OnnxEvaluator  # noqa: E402

model = Path(sys.argv[1])
games = int(sys.argv[2]) if len(sys.argv) > 2 else 10
sims = int(sys.argv[3]) if len(sys.argv) > 3 else 60
ev = OnnxEvaluator(model)
cfg = mcts.MCTSConfig(simulations=sims)
wins = 0
t0 = time.perf_counter()
for g in range(games):
    rng = np.random.default_rng(1000 + g)
    greedy = GreedyAgent(seed=2000 + g)
    env = ColonyOneEnv()
    while not env.terminated:
        if env.current_team == constants.CYAN:
            a = mcts.search_action(env.board, env.current_team, env.actions_left,
                                   env.round_number, ev, cfg, rng=rng)
        else:
            a = greedy.select(env.board, env.current_team, env.actions_left, env.round_number)
        env.step(a)
    wins += env.winner == constants.CYAN
    print(f"game {g}: winner {env.winner} after {env.action} actions", flush=True)
print(f"cyan wins {wins}/{games}  ({time.perf_counter() - t0:.1f}s)")
