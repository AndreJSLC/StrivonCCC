"""Count duplicate self-play games across .npz files (games split at the start position).

usage: dupcheck.py "glob1" ["glob2" ...]
"""
import sys, glob, hashlib, numpy as np, collections, os
def games(f):
    obs = np.load(f)['obs']
    start = obs[0]
    idx = [i for i in range(len(obs)) if np.array_equal(obs[i], start)] + [len(obs)]
    for a, b in zip(idx[:-1], idx[1:]):
        yield hashlib.md5(obs[a:b].tobytes()).hexdigest(), b - a, obs[a:b]
for pat in sys.argv[1:]:
    files = sorted(glob.glob(pat))
    seen = collections.Counter(); n = 0; firstk = collections.Counter()
    for f in files:
        for h, L, o in games(f):
            seen[h] += 1; n += 1
            firstk[hashlib.md5(o[:8].tobytes()).hexdigest()] += 1
    print(f"{pat}: files {len(files)} games {n} unique {len(seen)} ({len(seen)/max(n,1):.0%}); unique first-8-action prefixes {len(firstk)}; most common game x{seen.most_common(1)[0][1] if seen else 0}")
