# -*- coding: utf-8 -*-
# Plot trend figures from enhanced sbox_lns per-round logs (fields c52d6/nld6/nw).
import re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse(path):
    rounds, umax, c52, c52d6, nld6, nw, du6, nw8 = [], [], [], [], [], [], [], []
    pat = re.compile(
        r"round\s+(\d+) \| pool\s+(\d+) u max\s+([-\d.]+) \| best_c52 (\d+)"
        r" \| c52d6 (\d+) nld6 ([\d.]+) nw (\d+)(?: du6 (\d+) nw8 (\d+))?")
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                rounds.append(int(m.group(1)))
                umax.append(float(m.group(3)))
                c52.append(int(m.group(4)))
                v = int(m.group(5))
                c52d6.append(v if v < (1 << 29) else float("nan"))
                nld6.append(float(m.group(6)))
                nw.append(int(m.group(7)))
                du6.append(int(m.group(8)) if m.group(8) else 0)
                nw8.append(int(m.group(9)) if m.group(9) else 0)
    champ = -1
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"champ_round=(\d+)", line)
            if m:
                champ = int(m.group(1))
    return rounds, umax, c52, c52d6, nld6, nw, du6, nw8, champ

mode = sys.argv[1]
log_path = sys.argv[2]
out_path = sys.argv[3]
rounds, umax, c52, c52d6, nld6, nw, du6, nw8, champ = parse(log_path)

if mode == "climb":
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    ax.plot(rounds, c52, color="#1f77b4", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Round")
    ax.set_ylabel(r"best $c_{52}$ in pool (log scale)")
    ax.grid(alpha=0.3)
    ax.set_title("(a)")
    ax = axes[1]
    ax.plot(rounds, nw8, color="#2ca02c", lw=1.2)
    ax.set_xlabel("Round")
    ax.set_ylabel(r"DU$\leq$6, $1\leq c_{52}\leq 8$ S-boxes in pool")
    ax.grid(alpha=0.3)
    ax.set_title("(b)")
else:  # finish
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(rounds, nw, color="#2ca02c", lw=1.2)
    if champ > 0:
        ax.axvline(champ, color="gray", ls="--", lw=1)
    ax.set_xlabel("Round")
    ax.set_ylabel("near-optimal S-boxes in pool")
    ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(out_path)
print("saved", out_path, "rounds:", len(rounds), "champ:", champ)
