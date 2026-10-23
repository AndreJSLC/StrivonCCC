"""Play one Level 4 game between two ONNX models with the referee search and render it as an animated GIF.

Every frame shows the board after an action, the last move, the mover's top candidates from its
100-simulation search (visit shares) and the search's evaluation as a win-chance bar.

usage: showcase.py cyan.onnx magenta.onnx out.gif [--seed 0] [--sims 100] [--scene scenes/level-4_scene.json]
                   [--names "final net,heuristic"] [--still out.png] [--still-at N]
--still saves frame N as a PNG (default: the frame before the last action).
Needs Pillow.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "env"))
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

from colony_one import constants, encoding, mcts  # noqa: E402
from colony_one.env import ColonyOneEnv  # noqa: E402
from colony_one.onnx_runner import OnnxEvaluator  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cyan")
ap.add_argument("magenta")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sims", type=int, default=100)
ap.add_argument("--scene", default=str(HERE.parent / "scenes" / "level-4_scene.json"))
ap.add_argument("--names", default="final net,heuristic")
ap.add_argument("--still", default=None)
ap.add_argument("--still-at", type=int, default=-1)
A = ap.parse_args()

# ------------------------------------------------------------------ play
NAMES = {"PING": "Ping", "DAEMON": "Daemon", "EXPLOIT": "Exploit", "TRACE": "Trace", "ICE": "Ice", "KERNEL": "Kernel"}
LETTER = {"PING": "P", "DAEMON": "D", "EXPLOIT": "E", "TRACE": "T", "ICE": "I", "KERNEL": "K"}


def sq(pos):
    return f"{chr(ord('A') + pos[1])}{8 - pos[0]}"


def snapshot(board):
    grid = [[None if p is None else (p.team, p.piece_type.name) for p in row] for row in board.grid]
    return grid, [row[:] for row in board.locked]


scene = np.asarray(json.load(open(A.scene)), dtype=np.float32)
evs = {constants.CYAN: OnnxEvaluator(Path(A.cyan)), constants.MAGENTA: OnnxEvaluator(Path(A.magenta))}
env = ColonyOneEnv(scene=scene)
cfg = mcts.MCTSConfig(simulations=A.sims)
rng = np.random.default_rng(A.seed)
grid, locked = snapshot(env.board)
frames = [dict(grid=grid, locked=locked, round=1, action=0, last=None, cands=[], value=None,
               to_move=constants.CYAN, lost={constants.CYAN: [], constants.MAGENTA: []}, fw=[], end=None)]
lost = {constants.CYAN: [], constants.MAGENTA: []}
while not env.terminated:
    team, rnd = env.current_team, env.round_number
    root = mcts.run(env.board, team, env.actions_left, rnd, evs[team], cfg, rng=rng)
    temp = mcts.warmup_temperature(mcts.move_number(team, env.actions_left, rnd),
                                   cfg.warmup_temperatures, cfg.warmup_step_actions)
    idx = mcts.select_action_index(root, temp, rng=rng)
    total = sum(root.N.values())
    q = sum(root.W.values()) / max(1, total)
    cands = []
    for a, n in sorted(root.N.items(), key=lambda kv: -kv[1])[:3]:
        act = encoding.decode_action(a, team, env.board)
        cands.append((env.board.piece_at(*act.from_pos).piece_type.name, sq(act.from_pos), sq(act.to_pos),
                      n / total, a == idx))
    action = encoding.decode_action(idx, team, env.board)
    piece = env.board.piece_at(*action.from_pos).piece_type.name
    after = copy.deepcopy(env.board)  # the board after this action, before any firewall lock
    rec = after.apply_action(action)
    captured = None
    if rec.captured_piece is not None:
        captured = (rec.captured_piece.piece_type.name, rec.actual_target_pos)
        lost[rec.captured_piece.team].append(rec.captured_piece.piece_type.name)
    before_locked = [row[:] for row in env.board.locked]
    num = env.action  # 1-based number of this action
    env.step(action)
    grid, locked = snapshot(env.board)
    fw = [(r, c) for r in range(8) for c in range(8) if locked[r][c] and not before_locked[r][c]]
    burned = [after.grid[r][c] for r, c in fw if after.grid[r][c] is not None]
    for p in burned:
        lost[p.team].append(p.piece_type.name)
    end = None
    if env.terminated:
        if captured and captured[0] == "KERNEL":
            why = "Kernel captured"
        elif any(p.piece_type.name == "KERNEL" for p in burned):
            why = "Kernel destroyed by the firewall"
        else:
            why = "bare-Kernel endgame rule"
        end = (env.winner, why)
    frames.append(dict(grid=grid, locked=locked, round=rnd, action=num,
                       last=dict(team=team, piece=piece, frm=action.from_pos, to=action.to_pos,
                                 kind=action.kind, captured=captured),
                       cands=cands, value=q if team == constants.CYAN else -q, to_move=env.current_team,
                       lost={k: v[:] for k, v in lost.items()}, fw=fw, end=end))
print(f"{len(frames) - 1} actions, winner {env.winner}")

# ------------------------------------------------------------------ render
S, M, PW = 64, 30, 300                      # square, board margin, panel width
W, H = 2 * M + 8 * S + PW, 2 * M + 8 * S + 24
BG, PANEL = (11, 15, 25), (17, 23, 37)
SQ = [(31, 39, 58), (24, 31, 47)]
LOCK, HATCH = (46, 18, 28), (88, 28, 44)
COL = {constants.CYAN: (34, 211, 238), constants.MAGENTA: (232, 121, 249)}
TEXT, MUTED, HI, RED, AMBER = (226, 232, 240), (120, 136, 165), (250, 204, 21), (248, 113, 113), (245, 158, 11)
FONTS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/Library/Fonts")]
DEJAVU = {"segoeuib.ttf": "DejaVuSans-Bold.ttf", "arialbd.ttf": "DejaVuSans-Bold.ttf",
          "segoeui.ttf": "DejaVuSans.ttf", "arial.ttf": "DejaVuSans.ttf", "consola.ttf": "DejaVuSansMono.ttf"}


def font(name, size, fallback="arial.ttf"):
    for n in (name, fallback, DEJAVU.get(name, "DejaVuSans.ttf")):
        for d in FONTS:
            try:
                return ImageFont.truetype(str(d / n), size)
            except OSError:
                continue
    return ImageFont.load_default(size)


F_PIECE, F_KERN = font("segoeuib.ttf", 26, "arialbd.ttf"), font("segoeuib.ttf", 30, "arialbd.ttf")
F_COORD, F_SMALL = font("consola.ttf", 14, "arial.ttf"), font("segoeui.ttf", 15)
F_BODY, F_BOLD = font("segoeui.ttf", 17), font("segoeuib.ttf", 17, "arialbd.ttf")
F_TITLE, F_BIG = font("segoeuib.ttf", 21, "arialbd.ttf"), font("segoeuib.ttf", 46, "arialbd.ttf")


def center(r, c):
    return M + c * S + S // 2, M + r * S + S // 2


def arrow(d, a, b, color, width=6, dashed=False, gap=24):
    (x0, y0), (x1, y1) = a, b
    v = np.array([x1 - x0, y1 - y0], dtype=float)
    n = np.linalg.norm(v)
    if n < 1:
        return
    u = v / n
    x0, y0 = x0 + u[0] * 8, y0 + u[1] * 8
    n -= 8
    tip = np.array([x1, y1]) - u * gap
    base = tip - u * 18
    if dashed:
        for t in np.arange(0, n - gap - 18, 14):
            p, e = np.array([x0, y0]) + u * t, np.array([x0, y0]) + u * min(t + 8, n - gap - 18)
            d.line([tuple(p), tuple(e)], fill=color, width=width)
    else:
        d.line([(x0, y0), tuple(base)], fill=color, width=width)
    perp = np.array([-u[1], u[0]])
    d.polygon([tuple(tip), tuple(base + perp * 11), tuple(base - perp * 11)], fill=color)


def draw_piece(d, r, c, team, kind, glow=None, ghost=False):
    x, y = center(r, c)
    rad = 25 if kind == "KERNEL" else 21
    col = COL[team]
    if glow is not None:
        glow.ellipse([x - rad - 6, y - rad - 6, x + rad + 6, y + rad + 6], fill=col + (150,))
    if ghost:
        d.ellipse([x - rad, y - rad, x + rad, y + rad], outline=col, width=3)
        d.line([(x - 16, y - 16), (x + 16, y + 16)], fill=RED, width=5)
        d.line([(x - 16, y + 16), (x + 16, y - 16)], fill=RED, width=5)
        return
    d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=col)
    if kind == "KERNEL":
        d.ellipse([x - rad + 4, y - rad + 4, x + rad - 4, y + rad - 4], outline=BG, width=3)
    f = F_KERN if kind == "KERNEL" else F_PIECE
    d.text((x, y + 1), LETTER[kind], font=f, fill=BG, anchor="mm")


def render(fr):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    last = fr["last"]
    for r in range(8):
        for c in range(8):
            x0, y0 = M + c * S, M + r * S
            if fr["locked"][r][c]:
                d.rectangle([x0, y0, x0 + S - 1, y0 + S - 1], fill=LOCK)
                for k in range(-S, S, 12):
                    d.line([(x0 + max(k, 0), y0 + max(-k, 0)), (x0 + min(S + k, S) - 1, y0 + min(S - k, S) - 1)],
                           fill=HATCH, width=2)
            else:
                d.rectangle([x0, y0, x0 + S - 1, y0 + S - 1], fill=SQ[(r + c) % 2])
            if (r, c) in fr["fw"]:
                d.rectangle([x0 + 2, y0 + 2, x0 + S - 3, y0 + S - 3], outline=AMBER, width=3)
    if last:
        for pos in (last["frm"], last["to"]):
            x0, y0 = M + pos[1] * S, M + pos[0] * S
            d.rectangle([x0 + 1, y0 + 1, x0 + S - 2, y0 + S - 2], outline=HI, width=3)
    for i in range(8):
        d.text((M + i * S + S // 2, M + 8 * S + 12), chr(ord("A") + i), font=F_COORD, fill=MUTED, anchor="mm")
        d.text((M // 2, M + i * S + S // 2), str(8 - i), font=F_COORD, fill=MUTED, anchor="mm")
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for r in range(8):
        for c in range(8):
            p = fr["grid"][r][c]
            if p:
                draw_piece(gd, r, c, p[0], p[1], glow=gd)
    img.paste(glow.filter(ImageFilter.GaussianBlur(9)), (0, 0), glow.filter(ImageFilter.GaussianBlur(9)))
    d = ImageDraw.Draw(img)
    for r in range(8):
        for c in range(8):
            p = fr["grid"][r][c]
            if p:
                draw_piece(d, r, c, p[0], p[1])
    if last:
        if last["captured"] and last["kind"] == "shoot":
            tr, tc = last["captured"][1]
            enemy = constants.MAGENTA if last["team"] == constants.CYAN else constants.CYAN
            draw_piece(d, tr, tc, enemy, last["captured"][0], ghost=True)
        elif last["captured"] and last["captured"][1] != last["to"]:  # bodyguard took the hit
            tr, tc = last["captured"][1]
            enemy = constants.MAGENTA if last["team"] == constants.CYAN else constants.CYAN
            draw_piece(d, tr, tc, enemy, last["captured"][0], ghost=True)
        arrow(d, center(*last["frm"]), center(*last["to"]), RED if last["kind"] == "shoot" else HI,
              dashed=last["kind"] == "shoot")
    if fr["end"]:
        ov = Image.new("RGBA", (8 * S, 8 * S), (11, 15, 25, 170))
        img.paste(ov, (M, M), ov)
        d = ImageDraw.Draw(img)
        winner, why = fr["end"]
        d.text((M + 4 * S, M + 4 * S - 22), f"{winner} WINS", font=F_BIG, fill=COL[winner], anchor="mm")
        d.text((M + 4 * S, M + 4 * S + 26), why, font=F_BODY, fill=TEXT, anchor="mm")

    # ---- side panel
    px = 2 * M + 8 * S
    d.rectangle([px, 0, W, H], fill=PANEL)
    x, y = px + 22, 26
    d.text((x, y), "COLONY: ONE  ·  LEVEL 4", font=F_TITLE, fill=TEXT)
    y += 34
    d.text((x, y), f"Round {fr['round']} / 26", font=F_BODY, fill=MUTED)
    d.text((W - 22, y), f"Action {fr['action']}", font=F_BODY, fill=MUTED, anchor="ra")
    y += 38
    for team, name in ((constants.CYAN, NAMES_T[0]), (constants.MAGENTA, NAMES_T[1])):
        d.ellipse([x, y + 3, x + 14, y + 17], fill=COL[team])
        d.text((x + 24, y), f"{team.title()}", font=F_BOLD, fill=TEXT)
        d.text((x + 112, y), name, font=F_BODY, fill=MUTED)
        y += 28
    y += 14
    d.text((x, y), "Search evaluation (Cyan win chance)", font=F_SMALL, fill=MUTED)
    y += 24
    bw = W - 22 - x
    v = fr["value"]
    share = 0.5 if v is None else (v + 1) / 2
    d.rounded_rectangle([x, y, x + bw, y + 18], radius=9, fill=COL[constants.MAGENTA])
    d.rounded_rectangle([x, y, x + max(18, int(bw * share)), y + 18], radius=9, fill=COL[constants.CYAN])
    d.text((x + bw // 2, y + 9), "-" if v is None else f"{share:.0%}", font=F_BOLD, fill=BG, anchor="mm")
    y += 40
    if last:
        who = last["team"].title()
        d.text((x, y), f"{who}'s search ({A.sims} simulations)", font=F_SMALL, fill=MUTED)
        y += 26
        for kind, f_, t_, s, played in fr["cands"]:
            col = HI if played else TEXT
            d.text((x, y), f"{NAMES[kind]} {f_} > {t_}", font=F_BOLD if played else F_BODY, fill=col)
            d.text((W - 22, y), f"{s:.0%}", font=F_BOLD if played else F_BODY, fill=col, anchor="ra")
            y += 26
        y += 8
        verb = "shoots" if last["kind"] == "shoot" else "moves"
        line = f"{who} {NAMES[last['piece']]} {verb} {sq(last['frm'])} > {sq(last['to'])}"
        d.text((x, y), line, font=F_BODY, fill=TEXT)
        y += 26
        if last["captured"]:
            d.text((x, y), f"captures {NAMES[last['captured'][0]]}", font=F_BOLD, fill=RED)
        y += 26
    else:
        d.text((x, y), "Starting position", font=F_BODY, fill=TEXT)
        y += 104
    if fr["fw"]:
        d.text((x, y), "Firewall locked the outer ring", font=F_BOLD, fill=AMBER)
    y += 34
    d.text((x, y), "Lost pieces", font=F_SMALL, fill=MUTED)
    y += 24
    for team in (constants.CYAN, constants.MAGENTA):
        cx = x
        d.ellipse([cx, y + 4, cx + 10, y + 14], fill=COL[team])
        cx += 20
        for k in fr["lost"][team]:
            d.text((cx, y), LETTER[k], font=F_BOLD, fill=COL[team])
            cx += 18
        y += 24
    return img


NAMES_T = [s.strip() for s in A.names.split(",")]
imgs = [render(fr) for fr in frames]
dur = [1600] + [900] * (len(imgs) - 2) + [4000]
imgs[0].save(A.out, save_all=True, append_images=imgs[1:], duration=dur, loop=0, optimize=True, disposal=2)
print("saved", A.out)
if A.still:
    imgs[A.still_at if A.still_at >= 0 else len(imgs) - 2].save(A.still)
    print("saved", A.still)
