// Exchange-neighborhood local search for expanding the near-optimal S-box
// pool (Stage 2 of the paper's pipeline, Algorithm 6).
//
// From the starting boxes, greedy swap trajectories are run with the
// lexicographic objective J = (peak |W|, c52, h40); boxes that refresh the
// trajectory minimum and satisfy DU<=6 with 1 <= c52 <= 4 are kept.
// Parallel over (source, perturbed state) pairs; per-thread workspaces only,
// results merged at the end.
//
// All parameters are collected in the Params block at the top of main();
// no command-line arguments are used. The defaults are exactly the values
// reported in the paper (Table 1).
//
// Build (Windows/MinGW, static link):
//   g++ -O3 -march=native -fopenmp -static-libstdc++ -static-libgcc \
//       -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic enrich_at.cpp -o enrich_at.exe
// Build (Linux):
//   g++ -O3 -march=native -fopenmp enrich_at.cpp -o enrich_at
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// The algorithm is stochastic: every run can produce different boxes.
static uint64_t entropy_seed() {
    std::random_device rd;
    uint64_t s = (uint64_t)rd() << 32 | (uint64_t)rd();
    s ^= (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return splitmix64(s);
}

static const int N = 256;
using Box = std::array<uint8_t, N>;
static int8_t P[N][N];

static void init_tables() {
    for (int a = 0; a < N; a++)
        for (int x = 0; x < N; x++)
            P[a][x] = (__builtin_popcount(a & x) & 1) ? -1 : 1;
}

static void fwht(int16_t* v) {
    for (int h = 1; h < N; h <<= 1)
        for (int i = 0; i < N; i += 2 * h)
            for (int j = i; j < i + h; j++) {
                int16_t a = v[j], b = v[j + h];
                v[j] = a + b; v[j + h] = a - b;
            }
}

static void compute_lat(const Box& S, int16_t W[N][N]) {
    for (int b = 0; b < N; b++) {
        for (int x = 0; x < N; x++) W[b][x] = P[b][S[x]];
        fwht(W[b]);
    }
}

static int ddt_max(const Box& S) {
    int du = 0;
    for (int a = 1; a < N; a++) {
        uint8_t cnt[N] = {0};
        int mx = 0;
        for (int x = 0; x < N; x++) {
            uint8_t d = S[x] ^ S[x ^ a];
            cnt[d]++;
            if (cnt[d] > mx) mx = cnt[d];
        }
        du = std::max(du, mx);
    }
    return du;
}

struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t s) : g(s) {}
    float uni() { return std::uniform_real_distribution<float>(0, 1)(g); }
    int randint(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(g); }
};

struct Lex { int maxw; int c52; double h40; };
static inline double lexval(const Lex& l) { return l.maxw * 1e6 + l.c52 * 1e3 + l.h40; }

static Lex lex_of(const int16_t W[N][N]) {
    Lex l{0, 0, 0};
    for (int b = 1; b < N; b++)
        for (int a = 0; a < N; a++) {
            int w = W[b][a] < 0 ? -W[b][a] : W[b][a];
            l.maxw = std::max(l.maxw, w);
            l.c52 += (w >= 52);
            float d = std::max(0, w - 40);
            l.h40 += d * d;
        }
    l.h40 /= (255.0 * 256.0);
    return l;
}

struct Swap { int x1, x2; };

// One greedy trajectory (Algorithm 6 of the paper).
// max_steps = s (step limit), stall_lim = q (non-improving step limit),
// restart_lim = r (restart limit), top_m = m_J (candidates kept per step).
// Kept near-optimal boxes are appended to `kept`.
static void trajectory(Box S, int max_steps, int stall_lim, int restart_lim,
                       int top_m, uint64_t seed, std::vector<Box>& kept) {
    static thread_local int16_t W[N][N];
    compute_lat(S, W);
    Rng rng(seed);
    Lex cur = lex_of(W);
    double cur_cost = lexval(cur), best = cur_cost;
    Box S_best = S;
    int stall = 0, restarts = 0;

    // boundary entries (|W| >= 44): J is estimated only on these
    std::vector<int> crit;
    auto rebuild_crit = [&]() {
        crit.clear();
        for (int b = 1; b < N; b++)
            for (int a = 0; a < N; a++) {
                int w = W[b][a] < 0 ? -W[b][a] : W[b][a];
                if (w >= 44) crit.push_back(b * N + a);
            }
    };
    rebuild_crit();

    for (int it = 0; it < max_steps; it++) {
        // evaluate all swaps on the boundary set, keep the top_m cheapest
        std::vector<Swap> tops;
        std::vector<double> topc;
        double worst = 1e30;
        for (int x1 = 0; x1 < N; x1++)
            for (int x2 = x1 + 1; x2 < N; x2++) {
                int v1 = S[x1], v2 = S[x2];
                int maxw = 0, c52 = 0;
                double h40 = 0;
                for (int idx : crit) {
                    int b = idx / N, a = idx % N;
                    int d = (P[a][x1] - P[a][x2]) * (P[b][v2] - P[b][v1]);
                    int nw = W[b][a] + d;
                    int aw = nw < 0 ? -nw : nw;
                    maxw = std::max(maxw, aw);
                    c52 += (aw >= 52);
                    float dd = std::max(0, aw - 40);
                    h40 += dd * dd;
                }
                double c = maxw * 1e6 + c52 * 1e3 + h40 / (255.0 * 256.0);
                if ((int)tops.size() < top_m) {
                    tops.push_back({x1, x2}); topc.push_back(c);
                    if ((int)tops.size() == top_m)
                        worst = *std::max_element(topc.begin(), topc.end());
                } else if (c < worst) {
                    // replace current worst
                    int wi = int(std::max_element(topc.begin(), topc.end()) - topc.begin());
                    tops.at(wi) = {x1, x2}; topc.at(wi) = c;
                    worst = *std::max_element(topc.begin(), topc.end());
                }
            }
        // DU<=6 filter among the cheapest
        double bc = 1e30;
        Box bcand;
        bool have = false;
        for (size_t t = 0; t < tops.size(); t++) {
            Box cand = S;
            std::swap(cand.at(tops.at(t).x1), cand.at(tops.at(t).x2));
            if (ddt_max(cand) <= 6 && topc.at(t) < bc) { bc = topc.at(t); bcand = cand; have = true; }
        }
        if (!have) { stall++; }
        else if (bc < cur_cost - 1e-9) {
            S = bcand;
            compute_lat(S, W);
            rebuild_crit();
            cur_cost = bc;
            cur = lex_of(W);
            if (bc < best - 1e-9) {
                best = bc; S_best = S;
                int c52 = cur.c52;
                if (c52 >= 1 && c52 <= 4 && ddt_max(S) <= 6) kept.push_back(S);
                if (cur.maxw <= 48) break;  // reached NL104: stop this trajectory
            }
            stall = 0;
        } else {
            // sideways move: random feasible swap with cost <= cur
            std::vector<int> side;
            for (size_t t = 0; t < tops.size(); t++) {
                Box cand = S;
                std::swap(cand.at(tops.at(t).x1), cand.at(tops.at(t).x2));
                if (ddt_max(cand) <= 6 && topc.at(t) <= cur_cost + 1e-9) side.push_back((int)t);
            }
            if (!side.empty()) {
                int t = side.at(rng.randint(0, (int)side.size() - 1));
                Box cand = S;
                std::swap(cand.at(tops.at(t).x1), cand.at(tops.at(t).x2));
                S = cand;
                compute_lat(S, W);
                rebuild_crit();
                cur_cost = topc.at(t);
                stall = 0;
            } else stall++;
        }
        if (stall >= stall_lim) {
            // restart from the best box of this trajectory with a random feasible swap
            S = S_best;
            for (int r = 0; r < 2; r++) {
                for (int t = 0; t < 64; t++) {
                    int x1 = rng.randint(0, N - 1), x2 = rng.randint(0, N - 1);
                    if (x1 == x2) continue;
                    Box cand = S;
                    std::swap(cand[x1], cand[x2]);
                    if (ddt_max(cand) <= 6) { S = cand; break; }
                }
            }
            compute_lat(S, W);
            rebuild_crit();
            cur = lex_of(W); cur_cost = lexval(cur);
            stall = 0;
            if (++restarts >= restart_lim) break;
        }
    }
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    init_tables();

    // ======================================================================
    // Parameters (edit here; defaults are the values reported in the paper)
    // ======================================================================
    const std::string SRC_FILE = "data/sources.txt";  // Stage-2 starting points (139 boxes)
    const std::string OUT_FILE = "enriched.txt";      // kept near-optimal boxes (1022)
    const int STARTS   = 12;    // random perturbed states per starting box
    const int STEPS    = 300;   // s: step limit per search
    const int STALL    = 30;    // q: non-improving step limit before restart
    const int RESTARTS = 25;    // r: restart limit per search
    const int TOP_M    = 256;   // m_J: candidates kept per step
    const int THREADS  = 24;
    // ======================================================================

    std::vector<Box> srcs;
    {
        std::ifstream f(SRC_FILE);
        if (!f) { printf("cannot open %s\n", SRC_FILE.c_str()); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            Box b{}; int v, j = 0;
            while (j < N && (ss >> v)) b[j++] = (uint8_t)v;
            if (j == N) srcs.push_back(b);
        }
    }
    printf("sources %llu\n", (unsigned long long)srcs.size());

    int njobs = (int)srcs.size() * STARTS;
    std::vector<std::vector<Box>> all(njobs);
    uint64_t master = entropy_seed();
    double t0 = omp_get_wtime();
    #pragma omp parallel for num_threads(THREADS) schedule(dynamic, 1)
    for (int job = 0; job < njobs; job++) {
        uint64_t jseed = splitmix64(master + 0x9E3779B97F4A7C15ull * (uint64_t)job);
        int bi = job / STARTS, st = job % STARTS;
        Box S = srcs.at(bi);
        if (st > 0) {
            // perturbed state: two random feasible swaps
            Rng prng(jseed);
            for (int r = 0; r < 2; r++) {
                for (int t = 0; t < 64; t++) {
                    int x1 = prng.randint(0, N - 1), x2 = prng.randint(0, N - 1);
                    if (x1 == x2) continue;
                    Box cand = S;
                    std::swap(cand[x1], cand[x2]);
                    if (ddt_max(cand) <= 6) { S = cand; break; }
                }
            }
        }
        trajectory(S, STEPS, STALL, RESTARTS, TOP_M, splitmix64(jseed), all[job]);
    }

    std::set<Box> uniq;
    for (auto& v : all) for (auto& b : v) uniq.insert(b);
    std::ofstream out(OUT_FILE);
    for (auto& b : uniq) {
        for (int i = 0; i < N; i++) out << (int)b[i] << (i + 1 < N ? " " : "\n");
    }
    printf("kept %llu near-win boxes -> %s | %.1fs\n",
           (unsigned long long)uniq.size(), OUT_FILE.c_str(), omp_get_wtime() - t0);
    return 0;
}
