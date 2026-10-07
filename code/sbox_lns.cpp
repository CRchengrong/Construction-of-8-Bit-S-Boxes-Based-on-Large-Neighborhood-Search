// Constraint-guided destroy-repair large neighborhood search (LNS) for
// constructing 8-bit bijective S-boxes with differential uniformity 6 and
// nonlinearity 104 (Stages 1 and 3 of the paper's pipeline).
//
// All parameters are collected in the Params block at the top of main();
// no command-line arguments are used. The defaults are exactly the values
// reported in the paper (Tables 1 and 2).
//
// The algorithm is stochastic: every run produces different S-boxes.
//
// Build (Windows/MinGW, static link):
//   g++ -O3 -march=native -fopenmp -static-libstdc++ -static-libgcc \
//       -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic sbox_lns.cpp -o sbox_lns.exe
// Build (Linux):
//   g++ -O3 -march=native -fopenmp sbox_lns.cpp -o sbox_lns
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

static uint64_t entropy_seed() {
    std::random_device rd;
    uint64_t s = (uint64_t)rd() << 32 | (uint64_t)rd();
    s ^= (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return splitmix64(s);
}

static const int N = 256;
using Box = std::array<uint8_t, N>;

static int8_t P[N][N];        // P[a][x] = (-1)^{a.x}
static bool H_init = false;

static void init_tables() {
    if (H_init) return;
    for (int a = 0; a < N; a++)
        for (int x = 0; x < N; x++)
            P[a][x] = (__builtin_popcount(a & x) & 1) ? -1 : 1;
    H_init = true;
}

struct Feats {
    float nl = 0, h40 = 0, h32 = 0;
    int du = 0, c52 = 0, c48 = 0, d8 = 0;
};

static void fwht(int16_t* v) {
    for (int h = 1; h < N; h <<= 1)
        for (int i = 0; i < N; i += 2 * h)
            for (int j = i; j < i + h; j++) {
                int16_t a = v[j], b = v[j + h];
                v[j] = a + b; v[j + h] = a - b;
            }
}

// full LAT: W[beta][alpha], excludes beta=0 in metrics
static void compute_lat(const Box& S, int16_t W[N][N]) {
    for (int b = 0; b < N; b++) {
        for (int x = 0; x < N; x++) W[b][x] = P[b][S[x]];
        fwht(W[b]);
    }
}

static void compute_ddt_max(const Box& S, int& du, int& d8) {
    du = 0; d8 = 0;
    for (int a = 1; a < N; a++) {
        uint8_t cnt[N] = {0};
        int mx = 0;
        for (int x = 0; x < N; x++) {
            uint8_t d = S[x] ^ S[x ^ a];
            cnt[d]++;
            if (cnt[d] > mx) mx = cnt[d];
        }
        du = std::max(du, mx);
        for (int b = 0; b < N; b++) d8 += (cnt[b] >= 8);
    }
}

static Feats eval_feats(const Box& S, int16_t W[N][N]) {
    compute_lat(S, W);
    Feats f;
    int mxw = 0;
    for (int b = 1; b < N; b++)
        for (int a = 0; a < N; a++) {
            int w = W[b][a] < 0 ? -W[b][a] : W[b][a];
            mxw = std::max(mxw, w);
            f.c52 += (w >= 52); f.c48 += (w >= 48);
            float d40 = std::max(0, w - 40), d32 = std::max(0, w - 32);
            f.h40 += d40 * d40; f.h32 += d32 * d32;
        }
    f.nl = (N - mxw) / 2.0f;
    f.h40 /= (255.0f * 256.0f); f.h32 /= (255.0f * 256.0f);
    compute_ddt_max(S, f.du, f.d8);
    return f;
}

// removal score u(S), Eq. (utility) of the paper
static inline float ref_utility(const Feats& f) {
    return f.nl - 60.0f * std::max(0, f.du - 6) - 0.05f * f.c52 - 0.5f * f.d8;
}
// objective function r(S), Eq. (reward) of the paper
static inline float arm_reward(const Feats& f) {
    return 2.0f * f.nl - 1000.0f * f.h40 - f.c52 - 3.0f * f.d8
           - 100.0f * std::max(0, f.du - 6);
}

struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t s) : g(s) {}
    float uni() { return std::uniform_real_distribution<float>(0, 1)(g); }
    int randint(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(g); }
    float gumbel() {
        float u = std::max(uni(), 1e-9f);
        return -std::log(-std::log(u));
    }
};

// Constraint-based repair operator (Algorithm 4 of the paper).
// masked[i]=true marks a position to be refilled.
static void guided_repair(Box& x, const std::array<bool, N>& masked, Rng& rng) {
    int16_t cnt[N][N];                       // incremental DDT counts (a,b)
    memset(cnt, 0, sizeof(cnt));
    int16_t Wt[N][N];                        // incremental Walsh partial sums
    memset(Wt, 0, sizeof(Wt));
    bool m[N]; memcpy(m, masked.data(), N);
    bool avail_v[N]; memset(avail_v, 0, N);
    for (int i = 0; i < N; i++) if (masked[i]) avail_v[x[i]] = true;
    for (int i = 0; i < N; i++) if (masked[i]) x[i] = 255;      // mask token

    // init incremental tables from fixed positions
    for (int a = 0; a < N; a++)
        for (int i = 0; i < N; i++) if (!m[i] && !m[i ^ a])
            cnt[a][x[i] ^ x[i ^ a]]++;
    for (int i = 0; i < N; i++) if (!m[i])
        for (int b = 0; b < N; b++)
            for (int a = 0; a < N; a++)
                Wt[b][a] += P[b][x[i]] * P[a][i];

    auto commit = [&](int i, int v) {
        for (int a = 0; a < N; a++) if (!m[i ^ a])
            cnt[a][v ^ x[i ^ a]] += 2;
        for (int b = 0; b < N; b++)
            for (int a = 0; a < N; a++)
                Wt[b][a] += P[b][v] * P[a][i];
        x[i] = (uint8_t)v; m[i] = false; avail_v[v] = false;
    };

    int steps = 8, step = 0;                 // 8 assignment groups (Table 1)
    for (;;) {
        int remaining = 0;
        for (int i = 0; i < N; i++) remaining += m[i];
        if (!remaining) break;
        step++;
        int quota = step < steps ? std::max(1, (int)std::ceil(remaining / (double)(steps - step + 1))) : remaining;
        // boundary set C for Walsh penalties (|W| >= 44), rebuilt once per group
        std::vector<int> crit; crit.reserve(1024);
        for (int b = 1; b < N; b++)
            for (int a = 0; a < N; a++) {
                int w = Wt[b][a]; if (w < 0) w = -w;
                if (w >= 44) crit.push_back(b * N + a);
            }
        // candidate positions: masked indices in increasing index order
        std::vector<int> posl;
        for (int i = 0; i < N; i++) if (m[i]) posl.push_back(i);
        for (int q = 0; q < quota && !posl.empty(); q++) {
            int i = posl.front(); posl.erase(posl.begin());
            // score all available values directly
            std::vector<std::pair<float, int>> sc;
            for (int v = 0; v < N; v++) if (avail_v[v]) sc.push_back({rng.gumbel(), v});
            float best = -1e30f; int bv = -1;
            for (auto& [gn, v] : sc) {
                // pen_ddt
                int pen_du = 0;
                for (int a = 0; a < N; a++) if (!m[i ^ a])
                    pen_du += (cnt[a][v ^ x[i ^ a]] >= 6);
                // pen_lat + penE over the boundary set: delta of Wt entry = P[b][v]*P[a][i]
                int pen_lat = 0, penE = 0;
                for (int idx : crit) {
                    int b = idx / N, a = idx % N;
                    int d = P[b][v] * P[a][i];
                    int nw = Wt[b][a] + d, ow = Wt[b][a];
                    int anw = nw < 0 ? -nw : nw, aow = ow < 0 ? -ow : ow;
                    pen_lat += (anw >= 48 && aow < 48);
                    if (anw >= 44) penE += anw * anw - aow * aow;
                }
                bool forb = (pen_du > 0) || (pen_lat > 0);   // hard constraints
                float score = gn - 30.0f * pen_du - 3.0f * pen_lat - 8.0f * penE / 256.0f;
                if (forb) score = -1e29f;
                if (score > best) { best = score; bv = v; }
            }
            if (bv < 0) {
                // degenerate case: every candidate violates; take the value with
                // the largest Gumbel noise among the minimum-penalty values
                float bestp = 1e30f;
                for (auto& [gn, v] : sc) {
                    int pen_du = 0;
                    for (int a = 0; a < N; a++) if (!m[i ^ a])
                        pen_du += (cnt[a][v ^ x[i ^ a]] >= 6);
                    int pen_lat = 0;
                    for (int idx : crit) {
                        int b = idx / N, a = idx % N;
                        int nw = Wt[b][a] + P[b][v] * P[a][i];
                        nw = nw < 0 ? -nw : nw;
                        pen_lat += (nw >= 48);
                    }
                    float p = 30.0f * pen_du + 3.0f * pen_lat;
                    if (p < bestp) { bestp = p; bv = v; }
                }
            }
            commit(i, bv);
        }
    }
}

// Destroy operator (Algorithm 3 of the paper): LAT-attribution destroy score,
// Gumbel-perturbed top-k. Tb is the temperature T_b of Eq. (brk).
static void destroy_latmix(const Box& S, Rng& rng, int k, float Tb, std::array<bool, N>& masked) {
    static thread_local int16_t W[N][N];
    compute_lat(S, W);
    float score[N];
    for (int x = 0; x < N; x++) score[x] = 0;
    for (int b = 1; b < N; b++)
        for (int a = 0; a < N; a++) {
            int w = W[b][a]; if (w < 0) w = -w;
            if (w >= 48) {
                int sgn = W[b][a] >= 0 ? 1 : -1;
                for (int x = 0; x < N; x++)
                    score[x] += sgn * P[a][x] * P[b][S[x]];
            }
        }
    masked.fill(false);
    // gumbel-perturbed top-k on score/Tb + g
    std::vector<std::pair<float, int>> sc;
    for (int x = 0; x < N; x++) sc.push_back({score[x] / Tb + rng.gumbel(), x});
    std::partial_sort(sc.begin(), sc.begin() + k, sc.end(),
                      [](auto& p, auto& q) { return p.first > q.first; });
    for (int i = 0; i < k; i++) masked[sc[i].second] = true;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    init_tables();

    // ======================================================================
    // Parameters (edit here; defaults are the values reported in the paper)
    // ======================================================================
    const std::string MODE = "lns";   // "lns": Stages 1/3 search (Algorithm 2 of the paper)
                                      // "filter": select boxes with DU<=6 and c52 in [C52_LO,C52_HI]

    // --- MODE == "lns" ---
    const std::string POOL_IN   = "";               // initial pool file; "" -> Stage 1 starts
                                                    // from 16 random permutations.
                                                    // Stage 3: "data/pool_c1.txt"
    const int         ROUNDS    = 400;              // B1 = 400 (Stage 1); B3 = 600 (Stage 3)
    const std::string POOL_OUT  = "lns_pool.txt";   // final pool of this run
    const std::string CHAMP_OUT = "champion.txt";   // target S-box when found
    const int   POOL_CAPACITY = 8192;               // M, pool capacity
    const int   PARENTS       = 64;                 // n_p, parents sampled per round
    const int   COPIES        = 8;                  // n_c, copies per parent
    const int   K_MIN = 2, K_MAX = 16;              // destroy size k ~ U[K_MIN, K_MAX]
    const float DESTROY_TB    = 3.0f;               // T_b, temperature of the destroy operator

    // --- MODE == "filter" ---
    const std::string FILTER_IN  = "lns_pool.txt";  // merged Stage-1 pools -> sources.txt;
                                                    // enriched.txt -> pool_c1.txt
    const std::string FILTER_OUT = "sources.txt";
    const int C52_LO = 1, C52_HI = 8;               // [1,8] -> sources.txt; [1,1] -> pool_c1.txt

    // --- common ---
    const int THREADS = 24;
    // ======================================================================

    if (MODE == "filter") {
        // keep boxes in FILTER_IN with DU<=6 and c52 in [C52_LO,C52_HI]
        std::ifstream pf2(FILTER_IN);
        if (!pf2) { printf("cannot open %s\n", FILTER_IN.c_str()); return 1; }
        std::string line;
        std::vector<Box> srcs;
        while (std::getline(pf2, line)) {
            std::istringstream ss(line);
            Box b{}; int v, j = 0;
            while (j < N && (ss >> v)) b[j++] = (uint8_t)v;
            if (j == N) srcs.push_back(b);
        }
        std::vector<char> keep(srcs.size(), 0);
        #pragma omp parallel for num_threads(THREADS) schedule(static)
        for (size_t i = 0; i < srcs.size(); i++) {
            static thread_local int16_t W[N][N];
            Feats f = eval_feats(srcs[i], W);
            if (f.du <= 6 && f.c52 >= C52_LO && f.c52 <= C52_HI) keep[i] = 1;
        }
        std::ofstream out(FILTER_OUT);
        size_t nk = 0;
        for (size_t i = 0; i < srcs.size(); i++)
            if (keep[i]) {
                nk++;
                for (int j = 0; j < N; j++) out << (int)srcs[i][j] << (j + 1 < N ? " " : "\n");
            }
        printf("filter: kept %zu / %zu (c52 in [%d,%d], DU<=6) -> %s\n",
               nk, srcs.size(), C52_LO, C52_HI, FILTER_OUT.c_str());
        return 0;
    }

    // MODE == "lns": one destroy-repair LNS run (Algorithm 2 of the paper)
    Rng rng(entropy_seed());

    std::vector<Box> pool;
    {
        std::ifstream f(POOL_IN);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            Box b{}; int v, j = 0;
            while (j < N && (ss >> v)) b[j++] = (uint8_t)v;
            if (j == N) pool.push_back(b);
        }
    }
    printf("init pool %zu\n", pool.size());
    if (pool.empty()) {
        // Stage 1: start from 16 random permutations
        for (int i = 0; i < 16; i++) {
            Box b{}; std::iota(b.begin(), b.end(), 0);
            std::shuffle(b.begin(), b.end(), rng.g);
            pool.push_back(b);
        }
    }
    std::vector<Feats> pf(pool.size());
    {
        static thread_local int16_t W[N][N];
        for (size_t i = 0; i < pool.size(); i++) pf[i] = eval_feats(pool[i], W);
    }

    const int NCAND = PARENTS * COPIES;
    std::atomic<bool> found{false};
    double t0 = omp_get_wtime();
    int champ_round = -1;
    for (int rnd = 1; rnd <= ROUNDS && !found.load(); rnd++) {
        // parent sampling: normalized softmax of the objective r(S)
        std::vector<float> rw(pool.size());
        float mean = 0;
        for (size_t i = 0; i < pool.size(); i++) { rw[i] = arm_reward(pf[i]); mean += rw[i]; }
        mean /= pool.size();
        float var = 0;
        for (float v : rw) var += (v - mean) * (v - mean);
        float sd = std::sqrt(var / pool.size()) + 1e-6f;
        float mx = *std::max_element(rw.begin(), rw.end());
        std::vector<double> w(pool.size());
        double wsum = 0;
        for (size_t i = 0; i < pool.size(); i++) {
            w[i] = std::exp(2.0 * (rw[i] - mx) / sd); wsum += w[i];
        }
        std::vector<Box> states(PARENTS);
        for (int s = 0; s < PARENTS; s++) {
            double u = rng.uni() * wsum, acc = 0;
            size_t pick = pool.size() - 1;
            for (size_t i = 0; i < pool.size(); i++) { acc += w[i]; if (u <= acc) { pick = i; break; } }
            states[s] = pool[pick];
        }

        std::vector<Box> cands(NCAND);
        std::vector<Feats> fc(NCAND);
        uint64_t rbase = rng.g();   // per-round entropy from the master stream
        #pragma omp parallel for num_threads(THREADS) schedule(static)
        for (int s = 0; s < NCAND; s++) {
            Rng trng(splitmix64(rbase + 0x9E3779B97F4A7C15ull * (uint64_t)s));
            Box st = states[s / COPIES];
            int k = trng.randint(K_MIN, K_MAX);
            std::array<bool, N> masked;
            destroy_latmix(st, trng, k, DESTROY_TB, masked);
            Box cand = st;
            guided_repair(cand, masked, trng);
            cands[s] = cand;
            static thread_local int16_t W[N][N];
            fc[s] = eval_feats(cands[s], W);
        }
        // merge into pool with dedup + trim (Algorithm 5 of the paper)
        for (int s = 0; s < NCAND; s++) { pool.push_back(cands[s]); pf.push_back(fc[s]); }
        {
            std::set<Box> seen;
            std::vector<Box> np; std::vector<Feats> nf;
            for (size_t i = 0; i < pool.size(); i++)
                if (seen.insert(pool[i]).second) { np.push_back(pool[i]); nf.push_back(pf[i]); }
            pool.swap(np); pf.swap(nf);
            if (pool.size() > (size_t)POOL_CAPACITY) {
                std::vector<int> ord(pool.size());
                std::iota(ord.begin(), ord.end(), 0);
                std::partial_sort(ord.begin(), ord.begin() + POOL_CAPACITY, ord.end(),
                                  [&](int a, int b) { return ref_utility(pf[a]) > ref_utility(pf[b]); });
                std::vector<Box> np(POOL_CAPACITY); std::vector<Feats> nf(POOL_CAPACITY);
                for (int i = 0; i < POOL_CAPACITY; i++) { np[i] = pool[ord[i]]; nf[i] = pf[ord[i]]; }
                pool.swap(np); pf.swap(nf);
            }
        }
        // champion check
        for (int s = 0; s < NCAND && !found.load(); s++) {
            if (fc[s].du <= 6 && fc[s].nl >= 104.0f) {
                found.store(true);
                champ_round = rnd;
                std::ofstream out(CHAMP_OUT);
                for (int i = 0; i < N; i++) out << (int)cands[s][i] << (i + 1 < N ? "," : "\n");
            }
        }
        float best_u = -1e18f; int best_c52 = 1 << 30;
        for (size_t i = 0; i < pool.size(); i++) {
            best_u = std::max(best_u, ref_utility(pf[i]));
            best_c52 = std::min(best_c52, pf[i].c52);
        }
        float best_nl6 = 0; int best_c52_6 = 1 << 30; int n_nw = 0, n_du6 = 0, n_nw8 = 0;
        for (size_t i = 0; i < pool.size(); i++) {
            if (pf[i].du <= 6) {
                n_du6++;
                best_nl6 = std::max(best_nl6, pf[i].nl);
                best_c52_6 = std::min(best_c52_6, pf[i].c52);
                if (pf[i].c52 >= 1 && pf[i].c52 <= 4) n_nw++;
                if (pf[i].c52 >= 1 && pf[i].c52 <= 8) n_nw8++;
            }
        }
        double el = omp_get_wtime() - t0;
        printf("round %4d | pool %5zu u max %7.2f | best_c52 %d | c52d6 %d nld6 %.0f nw %d du6 %d nw8 %d | %.2fs/it\n",
               rnd, (unsigned long long)pool.size(), best_u, best_c52,
               best_c52_6, best_nl6, n_nw, n_du6, n_nw8, el / rnd);
        fflush(stdout);
    }
    printf("DONE champ_round=%d wall=%.1fs\n", champ_round, omp_get_wtime() - t0);
    {
        std::ofstream pf_out(POOL_OUT);
        for (auto& b : pool)
            for (int i = 0; i < N; i++) pf_out << (int)b[i] << (i + 1 < N ? " " : "\n");
    }
    return 0;
}
