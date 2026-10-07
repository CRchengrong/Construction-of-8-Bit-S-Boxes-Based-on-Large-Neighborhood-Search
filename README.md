# Supplementary Materials

Supplementary code, data, and logs for the paper:

> **Construction of 8-Bit S-Boxes Based on Large Neighborhood Search**

This package contains everything needed to verify every number reported in
the paper and to reproduce the experiments. The algorithm is stochastic, so
each run produces different S-boxes with statistically equivalent results.

## Directory structure

```
supplementary/
├── code/
│   ├── sbox_lns.cpp     Stages 1/3: constraint-based repair operator + destroy-repair LNS;
│   │                    also provides the pool filtering used between stages
│   └── enrich_at.cpp    Stage 2: exchange-neighborhood local search (pool enrichment)
├── scripts/
│   ├── plot_trend.py    regenerates Figures 1 and 2 from the per-round logs
│   └── verify_paper.py  checks every number reported in the paper against this package
├── data/
│   ├── champions/       sbox1.txt..sbox5.txt: the five S-boxes of Appendix A
│   │                    (decimal, comma-separated, 256 values each)
│   ├── climb_pool_1..4.txt  final elite pools of the four Stage-1 runs (8192 boxes each)
│   ├── sources.txt      Stage-2 starting points: 139 boxes with delta<=6 and c52<=8
│   ├── enriched.txt     Stage-2 output: 1022 near-optimal boxes (47 with c52=1)
│   └── pool_c1.txt      Stage-3 initial pool: the 47 boxes with c52=1
├── logs/
│   ├── climb_1..4.log   per-round logs of the four Stage-1 runs
│   ├── enrich.log       Stage-2 log
│   ├── finish_1..24.log per-round logs of the 24 Stage-3 runs
│   └── finish_summary.log   one-line summary per Stage-3 run (champion round, wall time)
└── figures/
    ├── fig_climb.pdf    Figure 1 of the paper (from logs/climb_3.log)
    └── fig_finish.pdf   Figure 2 of the paper (from logs/finish_12.log)
```

## Build

Windows (MinGW g++, static link):

```bash
g++ -O3 -march=native -fopenmp -static-libstdc++ -static-libgcc \
    -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic code/sbox_lns.cpp -o sbox_lns.exe
g++ -O3 -march=native -fopenmp -static-libstdc++ -static-libgcc \
    -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic code/enrich_at.cpp -o enrich_at.exe
```

Linux: drop `-Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic`.

## Parameters

The programs take **no command-line arguments**. All parameters are declared
as constants in a clearly marked block at the top of `main()`, with the
paper's values as defaults (Tables 1 and 2 of the paper):

- `sbox_lns.cpp`: `MODE` ("lns" for the Stages 1/3 search, "filter" for the
  pool filtering between stages), `POOL_IN`, `ROUNDS` (B1 = 400 for Stage 1,
  B3 = 600 for Stage 3), `POOL_CAPACITY` M = 8192, `PARENTS` n_p = 64,
  `COPIES` n_c = 8, `K_MIN`/`K_MAX` = 2/16, `DESTROY_TB` T_b = 3,
  `THREADS` = 24, and the filter file settings.
- `enrich_at.cpp`: `SRC_FILE`, `OUT_FILE`, `STARTS` = 12, `STEPS` s = 300,
  `STALL` q = 30, `RESTARTS` r = 25, `TOP_M` m_J = 256, `THREADS` = 24.

## Reproducing the pipeline (about 1 hour on 24 threads)

1. **Stage 1** (four independent runs): `MODE="lns"`, `POOL_IN=""`,
   `ROUNDS=400`. Run four times (e.g., in four separate working
   directories), collecting `lns_pool.txt` from each run
   (archived as `data/climb_pool_1..4.txt`, logs as `logs/climb_1..4.log`).
2. **Filter**: `MODE="filter"`, `FILTER_IN` = the four Stage-1 pools
   concatenated, `C52_LO=1`, `C52_HI=8`, `FILTER_OUT="sources.txt"`
   (139 boxes).
3. **Stage 2**: run `enrich_at` with `SRC_FILE="data/sources.txt"`
   (output `enriched.txt`, 1022 boxes).
4. **Filter**: `MODE="filter"`, `FILTER_IN="enriched.txt"`,
   `C52_LO=1`, `C52_HI=1`, `FILTER_OUT="pool_c1.txt"` (47 boxes).
5. **Stage 3** (24 independent runs): `MODE="lns"`,
   `POOL_IN="data/pool_c1.txt"`, `ROUNDS=600`
   (logs archived as `logs/finish_1..24.log`). A target S-box is written to
   `champion.txt` when found (19 of the 24 archived runs succeeded; the
   successful runs needed 7–531 rounds, median 67).

## Figures

```bash
python scripts/plot_trend.py climb  logs/climb_3.log    figures/fig_climb.pdf    # Figure 1
python scripts/plot_trend.py finish logs/finish_12.log  figures/fig_finish.pdf   # Figure 2
```

## Verification

```bash
python scripts/verify_paper.py
```

checks, directly from the files in this package: the five appendix S-boxes
(delta = 6, NL = 104, pairwise distinct, and identical to the manuscript's
Appendix A when the .tex file is found next to this folder); the pool counts
(139 starting points with 9 in G and 130 with c52 in [5,8]; 1022 enriched
boxes with 47 at c52 = 1; the 47-box Stage-3 pool); the Stage-3 statistics
(19/24 successes, rounds in [7, 531], median 67); and the figure data
(Figure 1: best c52 = 1 at round 183, count first nonzero at round 222, 79 at
round 400; Figure 2: 47 -> 144, champion at round 404). The output ends with
`ALL_OK` if every check passes.

Because the algorithm is stochastic, exact individual S-boxes and round
counts vary between reproductions; the archived logs and data files in this
package are the ones behind the numbers in the paper.
