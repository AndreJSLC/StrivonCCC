"""Gatekeeper duels on the real Level 4 scene. Meant for a many-core machine.

usage:
  gate.py CAND.onnx OPP.onnx [OPP2.onnx ...] [--pairs 60] [--sims 100] [--workers N] [--seed0 0]
          [--out results.jsonl]

Each pair is two games with the same seed: CAND as Cyan and CAND as Magenta. Every game uses the
referee's search exactly (mcts.search_action with MCTSConfig(simulations=sims), default warmup
temperatures and root noise, one ORT thread per evaluator). The same seeds are used for every
candidate, so candidates are compared on identical openings.

Also reports, for the candidate's side only: CPU seconds per action (time.process_time around
each search_action call) and peak RSS per worker. A game with no legal action for the side to
move counts as a loss for that side.
"""
import argparse
import json
import math
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "env"))
SCENE = ROOT / "scenes" / "level-4_scene.json"

_EV = {}
_SCENE = None


def _evaluator(path):
    from colony_one.onnx_runner import OnnxEvaluator
    if path not in _EV:
        _EV[path] = OnnxEvaluator(Path(path), num_threads=1)
    return _EV[path]


def play(job):
    global _SCENE
    import numpy as np
    from colony_one import constants, mcts
    from colony_one.env import ColonyOneEnv

    cand, opp, cand_cyan, seed, sims = job
    if _SCENE is None:
        _SCENE = np.asarray(json.load(open(SCENE)), dtype=np.float32)
    env = ColonyOneEnv(scene=_SCENE)
    cfg = mcts.MCTSConfig(simulations=sims)
    rng = np.random.default_rng(seed)
    ec, eo = _evaluator(cand), _evaluator(opp)
    cpu_c, n_c, stuck = 0.0, 0, None
    while not env.terminated:
        mine = (env.current_team == constants.CYAN) == cand_cyan
        t0 = time.process_time()
        act = mcts.search_action(env.board, env.current_team, env.actions_left,
                                 env.round_number, ec if mine else eo, cfg, rng=rng)
        if mine:
            cpu_c += time.process_time() - t0
            n_c += 1
        if act is None:
            stuck = env.current_team
            break
        env.step(act)
    if stuck is not None:
        cand_won = not mine
    else:
        cand_won = (env.winner == constants.CYAN) == cand_cyan
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return dict(opp=Path(opp).name, cyan=cand_cyan, seed=seed, win=bool(cand_won),
                actions=env.action, rounds=env.round_number, stuck=stuck,
                cand_cpu=cpu_c, cand_actions=n_c, rss_mb=rss_mb)


def effective_cpus():
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("opps", nargs="+")
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results.jsonl"))
    a = ap.parse_args()
    workers = a.workers or effective_cpus()
    cand = str(Path(a.cand).resolve())
    opps = [str(Path(o).resolve()) for o in a.opps]
    jobs = [(cand, o, side == 0, a.seed0 + 1000 * g + side, a.sims)
            for o in opps for g in range(a.pairs) for side in (0, 1)]
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(play, j) for j in jobs]
        for f in as_completed(futs):
            res.append(f.result())
    wall = time.time() - t0
    cname = Path(cand).name
    summary, lines = [], []
    for o in opps:
        on = Path(o).name
        r = [x for x in res if x["opp"] == on]
        k = sum(x["win"] for x in r)
        p, lo, hi = wilson(k, len(r))
        kc = sum(x["win"] for x in r if x["cyan"])
        km = sum(x["win"] for x in r if not x["cyan"])
        nc = sum(1 for x in r if x["cyan"])
        lines.append(f"{cname} vs {on}: {k}/{len(r)} = {p:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  "
                     f"as cyan {kc}/{nc}, as magenta {km}/{len(r) - nc}, "
                     f"avg {sum(x['actions'] for x in r) / len(r):.0f} actions, "
                     f"stuck {sum(1 for x in r if x['stuck'])}")
        summary.append(dict(cand=cname, opp=on, wins=k, games=len(r), p=p, lo=lo, hi=hi,
                            cyan_wins=kc, magenta_wins=km))
    cpu = sum(x["cand_cpu"] for x in res)
    acts = sum(x["cand_actions"] for x in res)
    per_act = cpu / max(acts, 1)
    worst_game = max(x["cand_cpu"] for x in res)
    lines.append(f"{cname} CPU per action at {a.sims} sims: {per_act * 1000:.0f} ms "
                 f"(worst game {worst_game:.1f} s over {max(x['cand_actions'] for x in res)} actions; "
                 f"52-action side projects to {52 * per_act:.1f} s), "
                 f"peak RSS {max(x['rss_mb'] for x in res):.0f} MB, "
                 f"{len(res)} games in {wall:.0f} s on {workers} workers")
    # write the record first: a dropped ssh session must not lose the result
    with open(a.out, "a") as fh:
        fh.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), cand=cname,
                                 sims=a.sims, pairs=a.pairs, seed0=a.seed0, workers=workers,
                                 summary=summary, cpu_per_action=per_act,
                                 worst_game_cpu=worst_game, wall=wall, lines=lines)) + "\n")
    try:
        print("\n".join(lines), flush=True)
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
