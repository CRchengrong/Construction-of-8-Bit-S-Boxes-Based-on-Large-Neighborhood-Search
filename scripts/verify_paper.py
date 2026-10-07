# -*- coding: utf-8 -*-
# Verifies that the data and logs in this package are consistent with the
# numbers reported in the paper "Construction of 8-Bit S-Boxes Based on
# Large Neighborhood Search". Run from anywhere:  python scripts/verify_paper.py
# Prints ALL_OK if every check passes.
import numpy as np, re, os, statistics

N = 256
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # supplementary/

# parity table P[a][x] = (-1)^{popcount(a&x)}
aa = np.arange(N, dtype=np.uint64)
M = aa[:, None] & aa[None, :]
try:
    pc = np.bitwise_count(M)
except AttributeError:
    pc = np.vectorize(lambda v: bin(int(v)).count("1"))(M)
P = 1 - 2 * (pc % 2).astype(np.int64)   # +1 / -1

def fwht_rows(A):
    A = A.astype(np.int64).copy()
    h = 1
    while h < N:
        A = A.reshape(N, -1, 2, h)
        u, v = A[:, :, 0, :], A[:, :, 1, :]
        A = np.stack([u + v, u - v], axis=2)
        A = A.reshape(N, -1)
        h <<= 1
    return A

def eval_box(S, need_deg=False):
    S = np.asarray(S, dtype=np.int64)
    W = fwht_rows(P[:, S])            # W[b][a]
    Wa = np.abs(W[1:])
    mxw = int(Wa.max())
    nl = (N - mxw) // 2
    c52 = int((Wa >= 52).sum()); c48 = int((Wa >= 48).sum())
    idx = np.arange(N)
    du = 0
    for a_ in range(1, N):
        d = S ^ S[idx ^ a_]
        cnt = np.bincount(d, minlength=N)
        du = max(du, int(cnt.max()))
    deg = None
    if need_deg:
        deg = 0
        for bit in range(8):
            A = ((S >> bit) & 1).astype(np.int64)
            for i in range(8):
                step = 1 << i
                A = A.reshape(-1, 2, step)
                A[:, 1, :] ^= A[:, 0, :]
                A = A.reshape(-1)
            degs = np.bitwise_count(np.arange(N, dtype=np.uint64))[A != 0]
            deg = max(deg, int(degs.max()))
    return du, nl, deg, c52, c48

def load(fn):
    rows = []
    with open(fn) as f:
        for line in f:
            v = [int(t) for t in re.findall(r"\d+", line)]
            if len(v) >= N:
                rows.append(v[:N])
    return rows

ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

# ---------- 1. Appendix S-boxes (Appendix A of the paper) ----------
champs = []
for k in range(1, 6):
    S = load(os.path.join(base, "data/champions/sbox%d.txt" % k))[0]
    champs.append(S)
    du, nl, deg, c52, c48 = eval_box(S, need_deg=True)
    check(f"Sbox{k}: delta=6, NL=104 (paper Appendix A)",
          du == 6 and nl == 104, f"(du={du} nl={nl} deg={deg} c52={c52})")
dmin = min(int((np.array(a) != np.array(b)).sum())
           for i, a in enumerate(champs) for b in champs[i+1:])
check("five appendix S-boxes pairwise distinct", dmin > 0,
      f"(min pairwise Hamming distance {dmin})")

# consistency with the manuscript's appendix, if a tex file is found nearby
for texpath in [os.path.join(base, "..", "MDPI_submission", "manuscript.tex"),
                os.path.join(base, "..", "main.tex")]:
    if os.path.exists(texpath):
        tex = open(texpath, encoding="utf-8").read()
        for k in range(1, 6):
            m = re.search(r"(?:Sbox|S盒)%d = \[([0-9a-f, ]+)\]" % k, tex)
            if not m:
                check(f"Sbox{k}: found in {os.path.basename(texpath)}", False)
                continue
            S_tex = [int(h, 16) for h in m.group(1).split(",")]
            check(f"Sbox{k}: appendix == data/champions/sbox{k}.txt", S_tex == champs[k-1])
        break

# ---------- 2. Pool file counts (Section 4.2 of the paper) ----------
def stat(fn):
    rows = load(fn)
    n_du6 = n_g = n_c1 = n_c58 = 0
    for S in rows:
        du, nl, _, c52, _ = eval_box(S)
        if du <= 6:
            n_du6 += 1
            if 1 <= c52 <= 4: n_g += 1
            if c52 == 1: n_c1 += 1
            if 5 <= c52 <= 8: n_c58 += 1
    return len(rows), n_du6, n_g, n_c1, n_c58

n, n_du6, n_g, n_c1, n_c58 = stat(os.path.join(base, "data/sources.txt"))
check("sources.txt: 139 starting points, delta<=6, 9 in G, 130 with c52 in [5,8] (paper Section 4.2)",
      n == 139 and n_du6 == 139 and n_g == 9 and n_c58 == 130,
      f"(n={n}, du6={n_du6}, G={n_g}, c52 in [5,8]={n_c58})")

n, n_du6, n_g, n_c1, _ = stat(os.path.join(base, "data/enriched.txt"))
check("enriched.txt: 1022 elements of G, 47 with c52=1 (paper Section 4.2)",
      n == 1022 and n_g == 1022 and n_c1 == 47,
      f"(n={n}, G={n_g}, c52=1: {n_c1})")

n, n_du6, _, n_c1, _ = stat(os.path.join(base, "data/pool_c1.txt"))
check("pool_c1.txt: 47 boxes, all with c52=1 and delta<=6 (paper Section 4.2)",
      n == 47 and n_c1 == 47 and n_du6 == 47,
      f"(n={n}, c52=1: {n_c1}, du6={n_du6})")

# ---------- 3. Stage-3 success statistics (Sections 4.2/4.3) ----------
fin = open(os.path.join(base, "logs/finish_summary.log")).read()
rounds = [int(x) for x in re.findall(r"champ_round=(-?\d+)", fin)]
succ = sorted(r for r in rounds if r > 0)
check("Stage 3: 19/24 runs successful, rounds in [7,531], median 67 (paper Table 2, Section 4.2)",
      len(rounds) == 24 and len(succ) == 19 and succ[0] == 7 and succ[-1] == 531
      and statistics.median(succ) == 67,
      f"({len(succ)}/{len(rounds)}, min={succ[0]}, max={succ[-1]}, median={statistics.median(succ)})")

# ---------- 4. Figure data (Figures 1 and 2 of the paper) ----------
def parse_rounds(path):
    recs = {}
    pat = re.compile(r"round\s+(\d+) \| .*?best_c52 (\d+).*? nw (\d+) du6 (\d+) nw8 (\d+)")
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if m:
            recs[int(m.group(1))] = dict(best_c52=int(m.group(2)), nw=int(m.group(3)),
                                         du6=int(m.group(4)), nw8=int(m.group(5)))
    return recs

c3 = parse_rounds(os.path.join(base, "logs/climb_3.log"))
first1 = min(r for r, d in c3.items() if d["best_c52"] == 1)
nw8 = {r: d["nw8"] for r, d in c3.items()}
first_nw8 = min(r for r in sorted(nw8) if nw8[r] > 0)
check("Figure 1 (climb_3.log): best_c52 reaches 1 at round 183",
      first1 == 183, f"(first at round {first1})")
check("Figure 1 (climb_3.log): count of delta<=6, 1<=c52<=8 boxes nonzero from round 222, 79 at round 400",
      first_nw8 == 222 and nw8[400] == 79,
      f"(first at round {first_nw8}, round 400: {nw8.get(400)})")

f12txt = open(os.path.join(base, "logs/finish_12.log")).read()
champ = int(re.search(r"champ_round=(-?\d+)", f12txt).group(1))
f12 = parse_rounds(os.path.join(base, "logs/finish_12.log"))
nw_series = [f12[r]["nw"] for r in sorted(f12)]
check("Figure 2 (finish_12.log): G count rises 47 -> 144, champion at round 404",
      nw_series[0] == 47 and nw_series[-1] == 144 and champ == 404,
      f"(first={nw_series[0]}, last={nw_series[-1]}, champ_round={champ})")

print("ALL_OK" if ok else "MISMATCH")
