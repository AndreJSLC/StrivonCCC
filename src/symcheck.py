"""Check that Colony: One rules are invariant under the 8 board symmetries (D4).

For positions taken from self-play files, and for every symmetry g:
  legal(g(board)) == g(legal(board))   for both teams, including action kinds,
  g(apply(board, a)) == apply(g(board), g(a))   for every legal action a
  (bodyguard redirects included).

usage: symcheck.py file.npz [file.npz ...]
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "env"))
import numpy as np  # noqa: E402

from colony_one import constants  # noqa: E402
from colony_one.board import Board  # noqa: E402
from colony_one.rules import legal_actions_for_team  # noqa: E402

N = 7
SYMS = {
    "id": lambda r, c: (r, c),
    "rot90": lambda r, c: (c, N - r),
    "rot180": lambda r, c: (N - r, N - c),
    "rot270": lambda r, c: (N - c, r),
    "transpose": lambda r, c: (c, r),
    "antitranspose": lambda r, c: (N - c, N - r),
    "flip_rows": lambda r, c: (N - r, c),
    "flip_cols": lambda r, c: (r, N - c),
}


def transform_obs(obs, g):
    out = np.zeros_like(obs)
    for r in range(8):
        for c in range(8):
            rr, cc = g(r, c)
            out[:, rr, cc] = obs[:, r, c]
    return out


def snapshot(board):
    return tuple(
        tuple((p.piece_type.value, p.team) if p is not None else None for p in row) for row in board.grid
    )


def transform_snapshot(snap, g):
    out = [[None] * 8 for _ in range(8)]
    for r in range(8):
        for c in range(8):
            rr, cc = g(r, c)
            out[rr][cc] = snap[r][c]
    return tuple(tuple(row) for row in out)


def actions(board, team):
    return {(a.from_pos, a.to_pos, a.kind) for a in legal_actions_for_team(board, team)}


positions = []
for f in sys.argv[1:]:
    positions.extend(np.load(f)["obs"])
print("positions", len(positions))
checked = bad = 0
for obs in positions:
    base = Board.from_observation(obs, constants.CYAN)
    for name, g in SYMS.items():
        tb = Board.from_observation(transform_obs(obs, g), constants.CYAN)
        for team in (constants.CYAN, constants.MAGENTA):
            la = actions(base, team)
            lg = actions(tb, team)
            mapped = {(g(*f), g(*t), k) for f, t, k in la}
            if mapped != lg:
                bad += 1
                print("legal mismatch", name, team)
                continue
            for a in legal_actions_for_team(base, team):
                b1 = copy.deepcopy(base)
                b1.apply_action(a)
                b2 = copy.deepcopy(tb)
                ga = next(x for x in legal_actions_for_team(b2, team)
                          if x.from_pos == g(*a.from_pos) and x.to_pos == g(*a.to_pos))
                b2.apply_action(ga)
                checked += 1
                if transform_snapshot(snapshot(b1), g) != snapshot(b2):
                    bad += 1
                    print("apply mismatch", name, team, a)
print(f"checked {checked} action applications, mismatches {bad}")
