"""Engine reference helpers for checking hand-built features against the real rules.

Positions in grafts/positions.npz (a sample of self-play positions, not included) are canonical observations (planes 0-5 = side to move). board_from_obs
rebuilds them with the side to move as CYAN and no rotation, so squares are canonical squares and
encoding.action_index(a, CYAN) equals the policy index the model sees. The enemy's actions on that
board are listed with the same (unrotated) squares.

Run with the local env:  python -I grafts/ref.py   (prints a demo)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "env"))

import numpy as np  # noqa: E402

from colony_one import constants  # noqa: E402
from colony_one.board import Board  # noqa: E402
from colony_one.pieces import PieceType  # noqa: E402
from colony_one.rules import legal_actions_for_team  # noqa: E402

OWN, ENEMY = constants.CYAN, constants.MAGENTA
PIECES = list(PieceType)  # plane order: PING DAEMON EXPLOIT TRACE ICE KERNEL
SHRINK = (8, 16, 24, 26)


def load_positions(path=None):
    """(obs (N,16,8,8) float32, legal (N,4096) bool, z (N,) game result for the side to move,
    pi (N,4096) float16 300-sim visit distribution)."""
    d = np.load(path or ROOT / "grafts" / "positions.npz")
    legal = np.unpackbits(d["legal"], axis=1)[:, :4096].astype(bool)
    return d["obs"], legal, d["z"], d["pi"]


def board_from_obs(obs) -> Board:
    return Board.from_observation(obs, OWN)


def actions_left(obs) -> int:
    return 2 if obs[13, 0, 0] > 0.75 else 1


def magenta_to_move(obs) -> bool:
    return bool(obs[15, 0, 0] > 0.5)


def round_number(obs) -> int:
    """Recovered from the locked rings (plane 12) and the urgency scalar (plane 14)."""
    rings = 0
    for r in range(4):
        if obs[12, r, r] > 0.5:
            rings = r + 1
    if rings >= 4:
        return 27
    nxt = SHRINK[rings]
    u = float(obs[14, 0, 0])
    return int(round(nxt - 8 * (1.0 - u)))


def sq(pos) -> int:
    return pos[0] * 8 + pos[1]


def actions(board: Board, team: str = OWN):
    """[(from_sq, to_sq, kind)] for every legal action of team (canonical squares)."""
    return [(sq(a.from_pos), sq(a.to_pos), a.kind) for a in legal_actions_for_team(board, team)]


def kernel_sq(board: Board, team: str):
    for r in range(8):
        for c in range(8):
            p = board.grid[r][c]
            if p is not None and p.team == team and p.piece_type is PieceType.KERNEL:
                return r * 8 + c
    return None


def captures_kernel_within(board: Board, team: str, n_actions: int) -> bool:
    """Brute force: can `team` destroy the enemy Kernel within n_actions of its own (bodyguard rule
    included, the opponent does not move in between)?"""
    enemy = ENEMY if team == OWN else OWN
    for a in legal_actions_for_team(board, team):
        rec = board.apply_action(a)
        try:
            if rec.captured_piece is not None and rec.captured_piece.piece_type is PieceType.KERNEL \
                    and rec.captured_piece.team == enemy:
                return True
            if n_actions > 1 and captures_kernel_within(board, team, n_actions - 1):
                return True
        finally:
            board.undo_action(rec)
    return False


def attack_targets(board: Board, team: str):
    """Squares holding an enemy piece that `team` can remove with one action (move-capture or shot).
    Note: a hit on a guarded Kernel kills a guard instead -- see Board._pick_guard."""
    out = set()
    for a in legal_actions_for_team(board, team):
        t = board.grid[a.to_pos[0]][a.to_pos[1]]
        if t is not None and t.team != team:
            out.add(sq(a.to_pos))
    return out


if __name__ == "__main__":
    obs, legal, z, pi = load_positions()
    i = 0
    b = board_from_obs(obs[i])
    acts = actions(b, OWN)
    idx = sorted(f * 64 + t for f, t, _ in acts)
    assert idx == sorted(np.nonzero(legal[i])[0].tolist()), "legal mask mismatch"
    print("positions", len(obs), "| demo position 0: round", round_number(obs[i]),
          "actions_left", actions_left(obs[i]), "magenta_to_move", magenta_to_move(obs[i]))
    print("own legal actions", len(acts), "enemy legal actions", len(actions(b, ENEMY)))
    print("own can take enemy kernel this turn:", captures_kernel_within(b, OWN, actions_left(obs[i])))
    print("enemy can take own kernel in its next turn (2 actions):", captures_kernel_within(b, ENEMY, 2))
