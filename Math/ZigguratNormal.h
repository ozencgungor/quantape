#pragma once
/**
 * @file ZigguratNormal.h
 * @brief Ziggurat method for normal random variates — 256-layer implementation
 *
 * Uses the Marsaglia & Tsang (2000) decomposition with N = 256 equal-area layers:
 *
 *   Layer 0 (base): rectangle [0, v/y₀] × [0, y₀] + exponential tail
 *   Layer i (1..N-2): rectangle [0, x[i-1]] × [y[i-1], y[i]]
 *   Layer N-1 (top): rectangle [0, x[N-2]] × [y[N-2], 1]
 *
 * Each layer has area v = r·f(r) + ∫_r^∞ f(t)dt, where r is the tail cutoff.
 *
 * Table layout:
 *   x[0..N-1] — N right boundaries  (x[0]=r, x[N-1] close to 0)
 *   y[0..N-1] — N heights           (y[0]=f(r), y[N-1] close to 1)
 *   wtab[0..N-1], ktab[0..N-1] — fast sampling tables
 *
 * Sampling:
 *   1. Draw 64-bit integer u, layer i = u & 0xFF
 *   2. x = signed(u) * wtab[i]
 *   3. If |u| < ktab[i]: ACCEPT (fast path — ~98% of the time)
 *   4. Layer 0 slow path: if |x| < r accept, else tail sampling
 *   5. Interior slow path: rejection test between y-bounds
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace mc {

// ═══════════════════════════════════════════════════════════════════════════
// TABLE GENERATION
// ═══════════════════════════════════════════════════════════════════════════

struct ZigguratTables {
    static constexpr int N = 256;

    // x[0] = r (tail cutoff), x[1]..x[N-2] from recurrence, x[N-1] sentinel
    double x[N];
    // y[0] = f(r), y[1]..y[N-2] from recurrence, y[N-1] close to 1
    double y[N];
    double A;             // v = common layer area
    double r;             // tail cutoff

    // Precomputed for fast sampling (Marsaglia convention):
    //   Layer 0: wtab[0] = (v/y[0]) / 2^63,  ktab[0] = r·y[0]/v · 2^63
    //   Layer i: wtab[i] = x[i-1] / 2^63,    ktab[i] = x[i]/x[i-1] · 2^63
    uint64_t ktab[N];
    double   wtab[N];
};

/// Compute ziggurat tables via bisection on the closure condition.
///
/// Algorithm:
///   1. Bisection for r such that N layers of area v(r) close at y = 1
///   2. Upward recurrence y[i] = y[i-1] + v/x[i-1]
///   3. Near y≈1: use log1p for x = √(-2·log1p(y-1)) to avoid cancellation
///   4. Build fast-path tables with Marsaglia's base-layer convention
///
inline ZigguratTables generateZigguratTables() {
    ZigguratTables tab{};
    constexpr int N = ZigguratTables::N;

    auto f  = [](double x) { return std::exp(-0.5 * x * x); };

    auto finv = [](double y) -> double {
        if (y >= 0.9)
            return std::sqrt(-2.0 * std::log1p(y - 1.0));
        return std::sqrt(-2.0 * std::log(y));
    };

    const double sqrt_pi_over_2 = std::sqrt(M_PI / 2.0);

    // v(r) = r·f(r) + √(π/2)·erfc(r/√2) = area of each layer
    auto volume = [&](double r) -> double {
        return r * f(r) + sqrt_pi_over_2 * std::erfc(r * M_SQRT1_2);
    };

    // ── Step 1: Find r via bisection on the closure condition ──
    //
    // With N layers (1 base + N-2 interior + 1 top):
    // - Recurrence runs N-2 steps: y[i] = y[i-1] + v/x[i-1] for i=1..N-2
    // - Closure: x[N-2]·(1 - y[N-2]) = v  ⟺  y[N-2] + v/x[N-2] = 1
    //
    // small r → large v → recurrence overshoots 1 → residual > 0
    // large r → small v → recurrence undershoots  → residual < 0

    auto closureResidual = [&](double r) -> double {
        double v = volume(r);
        double yi = f(r);
        double xi = r;
        for (int i = 0; i < N - 2; ++i) {
            yi += v / xi;
            if (yi >= 1.0) return 1.0;
            xi = finv(yi);
        }
        return yi + v / xi - 1.0;
    };

    double r_lo = 3.0, r_hi = 5.0;
    for (int i = 0; i < 64; ++i) {
        double r_mid = 0.5 * (r_lo + r_hi);
        if (closureResidual(r_mid) > 0)
            r_lo = r_mid;
        else
            r_hi = r_mid;
    }

    double r_ = 0.5 * (r_lo + r_hi);
    double v = volume(r_);

    tab.r = r_;
    tab.A = v;

    // ── Step 2: Build x[] and y[] via upward recurrence ──
    //
    // Layer 0 (base): x[0] = r, y[0] = f(r)
    // Layer i (1..N-2): y[i] = y[i-1] + v/x[i-1], x[i] = f⁻¹(y[i])
    // Layer N-1 (top): x[N-1] = 0 (sentinel), y[N-1] from recurrence

    tab.x[0] = r_;
    tab.y[0] = f(r_);

    for (int i = 1; i <= N - 2; ++i) {
        double yi = tab.y[i - 1] + v / tab.x[i - 1];
        if (yi > 1.0) yi = 1.0;
        tab.y[i] = yi;
        tab.x[i] = finv(yi);
    }

    // Top layer sentinel
    tab.x[N - 1] = 0.0;
    // y[N-1]: compute from closure (should be ≈ 1 - v/x[N-2])
    tab.y[N - 1] = tab.y[N - 2] + v / tab.x[N - 2];
    if (tab.y[N - 1] > 1.0) tab.y[N - 1] = 1.0;

    // ── Step 3: Compute fast-path tables (Marsaglia convention) ──
    //
    // Layer 0 (base):
    //   width = v / y[0]  (extends beyond r to cover base rectangle area)
    //   ktab[0] = floor(r / (v/y[0]) · 2^63) = floor(r·y[0]/v · 2^63)
    //   wtab[0] = (v/y[0]) / 2^63
    //
    // Layer i (1..N-2):
    //   width = x[i-1]  (shifted: layer i's rectangle has width x[i-1])
    //   ktab[i] = floor(x[i] / x[i-1] · 2^63)
    //   wtab[i] = x[i-1] / 2^63
    //
    // Layer N-1 (top):
    //   width = x[N-2]
    //   ktab[N-1] = 0  (always slow path since x[N-1]=0)
    //   wtab[N-1] = x[N-2] / 2^63

    const double pow2_63 = static_cast<double>(1ULL << 63);
    const double scale = 1.0 / pow2_63;

    // Base layer
    double base_width = v / tab.y[0];
    tab.wtab[0] = base_width * scale;
    tab.ktab[0] = static_cast<uint64_t>((r_ / base_width) * pow2_63);

    // Interior layers (shifted by 1)
    for (int i = 1; i < N - 1; ++i) {
        tab.wtab[i] = tab.x[i - 1] * scale;
        tab.ktab[i] = static_cast<uint64_t>((tab.x[i] / tab.x[i - 1]) * pow2_63);
    }

    // Top layer
    tab.wtab[N - 1] = tab.x[N - 2] * scale;
    tab.ktab[N - 1] = 0;

    return tab;
}

// ═══════════════════════════════════════════════════════════════════════════
// VERIFICATION
// ═══════════════════════════════════════════════════════════════════════════

struct ZigguratVerification {
    double max_area_error;        // max |area[i] - A| / A
    double closure_error;         // |top_area - A| / A
    double max_f_error;           // max |y[i] - exp(-x[i]²/2)|
    double max_finv_error;        // max |x[i] - sqrt(-2*log(y[i]))|
    bool   monotone_x;           // x[0] > x[1] > ... > x[N-1]
    bool   monotone_y;           // y[0] < y[1] < ... < y[N-1]
};

inline ZigguratVerification verifyZigguratTables(const ZigguratTables& tab) {
    constexpr int N = ZigguratTables::N;
    ZigguratVerification vr{};
    vr.monotone_x = true;
    vr.monotone_y = true;

    const double sqrt_pi_over_2 = std::sqrt(M_PI / 2.0);
    const double A = tab.A;

    // Check base layer area: x[0]*y[0] + tail = v
    double base_area = tab.x[0] * tab.y[0]
                     + sqrt_pi_over_2 * std::erfc(tab.x[0] * M_SQRT1_2);
    vr.max_area_error = std::abs(base_area - A) / A;

    // Check interior layers: layer i has rectangle [0,x[i-1]] × [y[i-1],y[i]]
    // area = x[i-1] * (y[i] - y[i-1]) = v
    for (int i = 1; i <= N - 2; ++i) {
        double layer_area = tab.x[i - 1] * (tab.y[i] - tab.y[i - 1]);
        double rel_err = std::abs(layer_area - A) / A;
        vr.max_area_error = std::max(vr.max_area_error, rel_err);
    }

    // Check top layer: [0, x[N-2]] × [y[N-2], 1], area = x[N-2]*(1 - y[N-2])
    double top_area = tab.x[N - 2] * (1.0 - tab.y[N - 2]);
    vr.closure_error = std::abs(top_area - A) / A;
    vr.max_area_error = std::max(vr.max_area_error, vr.closure_error);

    // f consistency and inverse consistency
    vr.max_f_error = 0;
    vr.max_finv_error = 0;
    for (int i = 0; i < N - 1; ++i) {  // skip sentinel x[N-1]=0
        double f_xi = std::exp(-0.5 * tab.x[i] * tab.x[i]);
        vr.max_f_error = std::max(vr.max_f_error, std::abs(tab.y[i] - f_xi));

        if (tab.y[i] > 0 && tab.y[i] < 1) {
            double x_from_y = std::sqrt(-2.0 * std::log(tab.y[i]));
            vr.max_finv_error = std::max(vr.max_finv_error,
                                         std::abs(tab.x[i] - x_from_y));
        }
    }

    // Monotonicity (only for x[0]..x[N-2], since x[N-1]=0 is sentinel)
    for (int i = 0; i < N - 2; ++i) {
        if (tab.x[i] <= tab.x[i + 1]) vr.monotone_x = false;
    }
    for (int i = 0; i < N - 2; ++i) {
        if (tab.y[i] >= tab.y[i + 1]) vr.monotone_y = false;
    }

    return vr;
}

// ═══════════════════════════════════════════════════════════════════════════
// FAST PRNG: xoshiro256** (Blackman & Vigna, 2018)
// ═══════════════════════════════════════════════════════════════════════════

struct Xoshiro256ss {
    uint64_t s[4]{};

    explicit Xoshiro256ss(uint64_t seed = 42) { this->seed(seed); }

    void seed(uint64_t seed_val) {
        auto splitmix = [](uint64_t& z) -> uint64_t {
            z += 0x9e3779b97f4a7c15ULL;
            uint64_t r = z;
            r = (r ^ (r >> 30)) * 0xbf58476d1ce4e5b9ULL;
            r = (r ^ (r >> 27)) * 0x94d049bb133111ebULL;
            return r ^ (r >> 31);
        };
        uint64_t z = seed_val;
        s[0] = splitmix(z);
        s[1] = splitmix(z);
        s[2] = splitmix(z);
        s[3] = splitmix(z);
    }

    uint64_t operator()() {
        auto rotl = [](uint64_t x, int k) -> uint64_t {
            return (x << k) | (x >> (64 - k));
        };
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    double uniform01() {
        return static_cast<double>((*this)() >> 11) * 0x1.0p-53;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// ZIGGURAT SAMPLER
// ═══════════════════════════════════════════════════════════════════════════

class ZigguratNormal {
public:
    explicit ZigguratNormal(uint64_t seed = 42)
        : tab_(generateZigguratTables()), rng_(seed) {}

    /// Generate one standard normal variate
    double operator()() {
        while (true) {
            uint64_t u = rng_();
            int i = static_cast<int>(u & 0xFF);
            auto s = static_cast<int64_t>(u);

            double x = s * tab_.wtab[i];

            // Fast path: |s| < ktab[i]
            auto abs_s = static_cast<uint64_t>(s < 0 ? -s : s);
            if (abs_s < tab_.ktab[i])
                return x;

            // ── Slow path ──
            if (i == 0) {
                // Base layer: proposal width = v/y[0] > r
                // Fast accept already checked |x| < r.
                // If |x| >= r: sample from exponential tail beyond r.
                double abs_x = std::abs(x);
                if (abs_x < tab_.r)
                    return x;  // under the curve in base rectangle
                double tail_x = sampleTail();
                return (s < 0) ? -tail_x : tail_x;
            }

            // Interior / top layer rejection
            // Layer i has strip from y[i-1] to y[i] (or y[N-2] to 1.0 for top)
            double abs_x = std::abs(x);
            double y_lo = tab_.y[i - 1];
            double y_hi;
            if (i < ZigguratTables::N - 1)
                y_hi = tab_.y[i];
            else
                y_hi = 1.0;  // top layer
            double y_test = y_lo + rng_.uniform01() * (y_hi - y_lo);
            if (y_test < std::exp(-0.5 * abs_x * abs_x))
                return x;
        }
    }

    const ZigguratTables& tables() const { return tab_; }

private:
    ZigguratTables tab_;
    Xoshiro256ss rng_;

    /// Sample from the tail: P(X > r) using Marsaglia's method
    double sampleTail() {
        double r = tab_.r;
        while (true) {
            double u1 = rng_.uniform01();
            double u2 = rng_.uniform01();
            double tail_x = -std::log(u1) / r + r;
            if (-2.0 * std::log(u2) >= (tail_x - r) * (tail_x - r))
                return tail_x;
        }
    }
};

} // namespace mc
