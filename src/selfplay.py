"""Self-play data on the Level 4 scene: (obs, legal bits, visit policy, outcome, team).

usage: selfplay.py model.onnx out_prefix workers games_per_worker sims
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "env"))
SCENE = HERE.parent / "scenes" / "level-4_scene.json"


def worker(args):
    import numpy as np
    from colony_one import constants, encoding, mcts
    from colony_one.env import ColonyOneEnv
    from colony_one.onnx_runner import OnnxEvaluator
    model, out, wid, games, sims = args
    ev = OnnxEvaluator(Path(model))
    scene = np.asarray(json.load(open(SCENE)), dtype=np.float32)
    cfg = mcts.MCTSConfig(simulations=sims)
    rng = np.random.default_rng()  # OS entropy: repeated runs with the same net must not replay identical games
    O, L, Pi, Z, Q = [], [], [], [], []
    for g in range(games):
        env = ColonyOneEnv(scene=scene)
        obs, legal, pis, teams = [], [], [], []
        while not env.terminated:
            team = env.current_team
            root = mcts.run(env.board, team, env.actions_left, env.round_number, ev, cfg, rng=rng)
            Q.append(sum(root.W.values()) / max(1, sum(root.N.values())))
            pi = mcts.policy_target(root).astype(np.float32)
            obs.append(env.observation().astype(np.float32))
            legal.append(np.packbits(env.legal_mask()))
            pis.append(pi)
            teams.append(team)
            mv = mcts.move_number(team, env.actions_left, env.round_number)
            temp = 1.0 if mv <= 8 else (0.5 if mv <= 16 else 0.15)
            idx = mcts.select_action_index(root, temp, rng=rng)
            env.step(encoding.decode_action(idx, team, env.board))
        for o, l, p, t in zip(obs, legal, pis, teams):
            O.append(o); L.append(l); Pi.append(p)
            Z.append(1.0 if env.winner == t else -1.0)
    np.savez_compressed(f"{out}_{wid}.npz", obs=np.array(O), legal=np.array(L),
                        pi=np.array(Pi, dtype=np.float16), z=np.array(Z, dtype=np.float32),
                        q=np.array(Q, dtype=np.float32))
    return len(Z)


if __name__ == "__main__":
    model, out, workers, games, sims = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        n = sum(ex.map(worker, [(model, out, w, games, sims) for w in range(workers)]))
    print("positions", n)
