"""Colony: One solver. State: dict {(x,y): (team, kind)}, x,y in 1..8; team 'C'/'M'; kind 'K'/'P'."""
import sys
from functools import lru_cache

N = 8
DIRS8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
ORTH = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def inb(x, y):
    return 1 <= x <= N and 1 <= y <= N


def actions(board, team):
    """Yield (src, dst, new_board) for every legal action of team."""
    for (x, y), (t, k) in list(board.items()):
        if t != team:
            continue
        if k == 'K':
            for dx, dy in DIRS8:
                nx, ny = x + dx, y + dy
                if not inb(nx, ny):
                    continue
                o = board.get((nx, ny))
                if o and o[0] == team:
                    continue
                nb = dict(board)
                del nb[(x, y)]
                nb[(nx, ny)] = (t, k)
                yield (x, y), (nx, ny), nb
        else:
            for dx, dy in ORTH:
                for s in (1, 2):
                    nx, ny = x + dx * s, y + dy * s
                    if not inb(nx, ny) or (nx, ny) in board:
                        break
                    nb = dict(board)
                    del nb[(x, y)]
                    nb[(nx, ny)] = (t, k)
                    yield (x, y), (nx, ny), nb
            for dx, dy in DIAG:
                for s in (1, 2):
                    nx, ny = x + dx * s, y + dy * s
                    if not inb(nx, ny):
                        break
                    o = board.get((nx, ny))
                    if o is None:
                        continue
                    if o[0] != team:
                        nb = dict(board)
                        del nb[(x, y)]
                        nb[(nx, ny)] = (t, k)
                        yield (x, y), (nx, ny), nb
                    break


def has_k(board, team):
    return any(v == (team, 'K') for v in board.values())


def turns(board, team):
    """All 2-action sequences -> list of (seq, final_board); stops early if a kernel is captured."""
    out = []
    for a1 in actions(board, team):
        b1 = a1[2]
        if not has_k(b1, 'C' if team == 'M' else 'M'):
            out.append(((a1[:2],), b1))
            continue
        for a2 in actions(b1, team):
            out.append(((a1[:2], a2[:2]), a2[2]))
    return out


def key(b):
    return tuple(sorted(b.items()))


def win_now(board, team):
    """Sequence by team capturing enemy kernel this turn, else None."""
    en = 'C' if team == 'M' else 'M'
    for seq, b in turns(board, team):
        if not has_k(b, en):
            return seq
    return None


def solve(board, depth_turns):
    """Cyan to move; find a sequence that wins within depth_turns cyan turns vs best magenta play."""
    @lru_cache(None)
    def cwin(kb, d):
        b = dict(kb)
        res = None
        for seq, nb in turns(b, 'C'):
            if not has_k(nb, 'M'):
                return seq
        if d <= 1:
            return None
        for seq, nb in turns(b, 'C'):
            if not has_k(nb, 'C'):
                continue
            ok = True
            for mseq, mb in turns(nb, 'M'):
                if not has_k(mb, 'C'):
                    ok = False
                    break
                if cwin(key(mb), d - 1) is None:
                    ok = False
                    break
            if ok:
                return seq
        return None
    return cwin(key(board), depth_turns)


def name(p):
    return 'ABCDEFGH'[p[0] - 1] + str(p[1])


def parse(rows):
    b = {}
    for r, line in enumerate(rows):
        for c, ch in enumerate(line.split()):
            if ch == '.':
                continue
            b[(c + 1, 8 - r)] = (ch[0], ch[1])
    return b


if __name__ == '__main__':
    # CK=cyan kernel, CP=cyan ping, MK/MP magenta. rows listed top (8) to bottom (1)
    spec = sys.argv[1]  # e.g. "CK:H2,CP:E2,MK:B5"
    board = {}
    for item in spec.split(','):
        t, pos = item.split(':')
        board[('ABCDEFGH'.index(pos[0]) + 1, int(pos[1]))] = (t[0], t[1])
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    for dd in range(1, d + 1):
        s = solve(board, dd)
        if s:
            print('win in', dd, 'turn(s):', [(name(a), name(b)) for a, b in s])
            break
    else:
        print('no forced win within', d)


def advise(board, top=5):
    """Rank cyan turns: never lose K; maximise fraction of magenta replies after which cyan wins at once; penalise losing P."""
    scored = []
    seen = set()
    for seq, nb in turns(board, 'C'):
        if not has_k(nb, 'M'):
            scored.append((10 ** 9, seq))
            continue
        k = key(nb)
        if k in seen:
            continue
        seen.add(k)
        total = lost_k = lost_p = wins = 0
        for mseq, mb in turns(nb, 'M'):
            total += 1
            if not has_k(mb, 'C'):
                lost_k += 1
                continue
            if not any(v == ('C', 'P') for v in mb.values()):
                lost_p += 1
            if win_now(mb, 'C'):
                wins += 1
        if lost_k:
            continue
        scored.append((wins / total * 100 - lost_p / total * 50, seq, wins, lost_p, total))
    scored.sort(key=lambda s: -s[0])
    return scored[:top]


if __name__ == '__main__' and len(sys.argv) > 3 and sys.argv[3] == 'advise':
    for s in advise(board):
        print(round(s[0], 1), [(name(a), name(b)) for a, b in s[1]], s[2:])
