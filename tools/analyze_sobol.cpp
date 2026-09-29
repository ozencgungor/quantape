/**
 * @file analyze_sobol.cpp
 * @brief Broad statistical analysis of an extended Joe-Kuo table.
 *
 * Methods (all randomized where applicable via digital shifts):
 *   1. Digital-net exactness in 2D, 3D and 4D: for sampled projections the
 *      brute-force t-value t_s at level m is computed and every cell of a
 *      grid with b_1+...+b_s = m - t_s must contain exactly 2^{t_s} points.
 *   2. L2 star discrepancy (Warnock formula) of sampled 2D projections at
 *      N = 2^m points, compared with the average over random point sets.
 *   3. Pearson correlations of sampled dimension pairs (max |r|).
 *   4. QMC vs MC integration RMSE on smooth/oscillatory/kinked product
 *      integrands of dimension 1..8, with analytic references, using digital
 *      shifts of the Sobol points and independent pseudo-random samples.
 *   5. Usability in a finance pattern: arithmetic-average GBM Asian call
 *      priced from the extended dimensions in time order, QMC vs MC error
 *      against a large MC reference.
 *
 * Standalone (standard library + the Sobol headers).
 *
 * Usage:
 *   analyze_sobol --input=TABLE [--base=BASE] [--sample=8] [--m=12]
 *                 [--dims-log2=12] [--shifts=16] [--threads=0] [--jobs=all]
 */
#include "quantape/log/Log.h"
#include "quantape/math/Random/Sobol/DirectionNumbers.h"
#include "quantape/math/Random/Sobol/GF2.h"
#include "quantape/math/Random/Sobol/SobolQuality.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace quantape::math::mc;
using namespace quantape::math::mc::sobol;

namespace {

constexpr double kPi = 3.14159265358979323846;

/// Acklam's inverse normal CDF (|error| < 1.15e-9).
double inverseNormal(double p) {
    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                -2.759285104469687e+02, 1.383577518672690e+02,
                                -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                -1.556989798598866e+02, 6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                                3.754408661907416e+00};
    const double pl = 0.02425;
    if (p <= 0.0)
        return -1e10;
    if (p >= 1.0)
        return 1e10;
    if (p < pl) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > 1.0 - pl) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

/// Uniform (0,1) from a 64-bit engine.
inline double uniform01(std::mt19937_64& g) {
    return ((g() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

// ── 32-bit gray-code generation with digital shifts ──
std::vector<std::uint32_t> directionWords(const Entry& e, int L) {
    const auto m = integerDirectionNumbers(e, L);
    std::vector<std::uint32_t> V(static_cast<std::size_t>(L) + 1, 0);
    for (int i = 1; i <= L; ++i) {
        V[static_cast<std::size_t>(i)] =
            static_cast<std::uint32_t>(m[static_cast<std::size_t>(i)] << (32 - i));
    }
    return V;
}

std::vector<std::uint32_t> dimensionOneWords(int L) {
    std::vector<std::uint32_t> V(static_cast<std::size_t>(L) + 1, 0);
    for (int i = 1; i <= L; ++i) {
        V[static_cast<std::size_t>(i)] = 1u << (32 - i);
    }
    return V;
}

/// Gray-code points (32-bit X values) for a dimension, optionally XOR-shifted.
std::vector<std::uint32_t> generate(const std::vector<std::uint32_t>& V, int L,
                                    std::uint32_t shift) {
    const std::uint32_t N = 1u << L;
    std::vector<std::uint32_t> C(N, 1);
    for (std::uint32_t i = 1; i < N; ++i) {
        std::uint32_t v = i, c = 1;
        while (v & 1u) {
            v >>= 1;
            ++c;
        }
        C[i] = c;
    }
    std::vector<std::uint32_t> X(N, 0);
    for (std::uint32_t i = 1; i < N; ++i) {
        X[i] = X[i - 1] ^ V[C[i - 1]]; // flip at first zero bit of i-1
    }
    if (shift != 0) {
        for (auto& x : X) {
            x ^= shift;
        }
    }
    return X;
}

// ── generic rank / s-dimensional t-value by brute force ──
bool independentRows(const std::vector<std::uint32_t>& rows) {
    std::uint32_t pivots[32] = {};
    int rank = 0;
    for (std::uint32_t v : rows) {
        while (v) {
            const int b = 31 - __builtin_clz(v);
            if (!pivots[b]) {
                pivots[b] = v;
                ++rank;
                break;
            }
            v ^= pivots[b];
        }
    }
    return rank == static_cast<int>(rows.size());
}

int tValueS(const std::vector<DirectionMatrix>& mats, int m) {
    const int s = static_cast<int>(mats.size());
    const std::uint32_t mask = (m >= 32) ? 0xFFFFFFFFu : ((1u << m) - 1u);
    for (int t = 0; t <= m; ++t) {
        bool allOk = true;
        std::vector<int> r(s, 0);
        const auto rec = [&](auto&& self, int dim, int remaining) -> bool {
            if (dim == s - 1) {
                r[dim] = remaining;
                std::vector<std::uint32_t> rows;
                for (int j = 0; j < s; ++j) {
                    for (int q = 0; q < r[j]; ++q) {
                        rows.push_back(mats[j].rows[static_cast<std::size_t>(q)] & mask);
                    }
                }
                return independentRows(rows);
            }
            for (int v = 0; v <= remaining; ++v) {
                r[dim] = v;
                if (!self(self, dim + 1, remaining - v)) {
                    return false;
                }
            }
            return true;
        };
        if (!rec(rec, 0, m - t)) {
            allOk = false;
        }
        if (allOk) {
            return t;
        }
    }
    return m;
}

/// Exact cell counts for an s-dimensional grid with b_j bits per axis.
bool exactGridS(const std::vector<std::vector<std::uint32_t>>& X, const std::vector<int>& b,
                std::uint32_t expected, std::uint32_t* worstSeen) {
    std::size_t cells = 1;
    for (int v : b) {
        cells <<= v;
    }
    std::vector<std::uint32_t> counts(cells, 0);
    const std::size_t N = X.empty() ? 0 : X[0].size();
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t idx = 0;
        for (std::size_t j = 0; j < X.size(); ++j) {
            const std::uint32_t digit = (b[j] == 0) ? 0u : (X[j][i] >> (32 - b[j]));
            idx = (idx << b[j]) | digit;
        }
        counts[idx]++;
    }
    bool ok = true;
    std::uint32_t worst = 0;
    for (std::uint32_t c : counts) {
        ok = ok && (c == expected);
        worst = std::max(worst, c);
    }
    if (worstSeen) {
        *worstSeen = worst;
    }
    return ok;
}

// ── L2 star discrepancy (Warnock) for s = 2, points as doubles in [0,1)^2 ──
double l2StarDiscrepancy2D(const std::vector<double>& x, const std::vector<double>& y) {
    const std::size_t N = x.size();
    double sum1 = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        sum1 += (1.0 - x[i] * x[i]) * (1.0 - y[i] * y[i]);
    }
    double sum2 = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t k = 0; k < N; ++k) {
            sum2 += (1.0 - std::max(x[i], x[k])) * (1.0 - std::max(y[i], y[k]));
        }
    }
    return 1.0 / 9.0 - (0.5 / N) * sum1 + sum2 / (static_cast<double>(N) * N);
}

double pearson(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
    const double N = static_cast<double>(a.size());
    double ma = 0, mb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        ma += a[i] / 4294967296.0;
        mb += b[i] / 4294967296.0;
    }
    ma /= N;
    mb /= N;
    double cov = 0, va = 0, vb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double ua = a[i] / 4294967296.0 - ma;
        const double ub = b[i] / 4294967296.0 - mb;
        cov += ua * ub;
        va += ua * ua;
        vb += ub * ub;
    }
    return cov / std::sqrt(va * vb);
}

// ── test integrands with analytic references (u = 0.5, a_j = j/(s+1)) ──
struct Integrand {
    const char* name;
    double (*value)(const std::vector<double>&, const std::vector<double>&);
    double (*reference)(const std::vector<double>&);
};

double gaussianValue(const std::vector<double>& x, const std::vector<double>& a) {
    double sum = 0;
    for (std::size_t j = 0; j < x.size(); ++j) {
        sum += a[j] * a[j] * (x[j] - 0.5) * (x[j] - 0.5);
    }
    return std::exp(-sum);
}
double gaussianRef(const std::vector<double>& a) {
    // prod sqrt(pi)/(2a) (erf(a*0.5) * 2)
    double p = 1.0;
    for (double aj : a) {
        p *= std::sqrt(kPi) / (2.0 * aj) * (2.0 * std::erf(aj * 0.5));
    }
    return p;
}
double productPeakValue(const std::vector<double>& x, const std::vector<double>& a) {
    double p = 1.0;
    for (std::size_t j = 0; j < x.size(); ++j) {
        const double d = x[j] - 0.5;
        p *= 1.0 / (1.0 / (a[j] * a[j]) + d * d);
    }
    return p;
}
double productPeakRef(const std::vector<double>& a) {
    double p = 1.0;
    for (double aj : a) {
        p *= aj * (std::atan(0.5 * aj) + std::atan(0.5 * aj));
    }
    return p;
}
double oscillatoryValue(const std::vector<double>& x, const std::vector<double>& a) {
    double phase = 2.0 * kPi * 0.5;
    for (std::size_t j = 0; j < x.size(); ++j) {
        phase += a[j] * x[j];
    }
    return std::cos(phase);
}
double oscillatoryRef(const std::vector<double>& a) {
    // Re[ e^{i pi} prod (e^{i a_j} - 1)/(i a_j) ]
    std::complex<double> product(1.0, 0.0);
    for (double aj : a) {
        product *= (std::exp(std::complex<double>(0.0, aj)) - 1.0) / std::complex<double>(0.0, aj);
    }
    return std::real(-product); // cos(pi) = -1
}
double exponentialValue(const std::vector<double>& x, const std::vector<double>& a) {
    double sum = 0;
    for (std::size_t j = 0; j < x.size(); ++j) {
        sum += a[j] * x[j];
    }
    return std::exp(sum);
}
double exponentialRef(const std::vector<double>& a) {
    double p = 1.0;
    for (double aj : a) {
        p *= (std::exp(aj) - 1.0) / aj;
    }
    return p;
}
double kinkValue(const std::vector<double>& x, const std::vector<double>&) {
    double p = 1.0;
    for (double xj : x) {
        p *= std::fabs(xj - 0.5);
    }
    return p;
}
double kinkRef(const std::vector<double>& a) {
    return std::pow(0.25, static_cast<double>(a.size()));
}

double qmcError(const Integrand& f, const std::vector<std::vector<std::uint32_t>>& base,
                const std::vector<std::vector<std::uint32_t>>& shifts, const std::vector<double>& a,
                std::mt19937_64& rng) {
    double sumSq = 0.0;
    unsigned count = 0;
    for (const auto& sh : shifts) {
        double s = 0.0;
        const std::size_t N = base[0].size();
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<double> x(base.size());
            for (std::size_t j = 0; j < base.size(); ++j) {
                x[j] = static_cast<double>(base[j][i] ^ sh[j]) / 4294967296.0;
            }
            s += f.value(x, a);
        }
        const double err = s / N - f.reference(a);
        sumSq += err * err;
        ++count;
    }
    (void)rng;
    return std::sqrt(sumSq / count);
}

double mcError(const Integrand& f, int s, int m, int replications, const std::vector<double>& a,
               std::mt19937_64& rng) {
    const std::size_t N = std::size_t(1) << m;
    double sumSq = 0.0;
    for (int r = 0; r < replications; ++r) {
        double ssum = 0.0;
        std::vector<double> x(static_cast<std::size_t>(s));
        for (std::size_t i = 0; i < N; ++i) {
            for (int j = 0; j < s; ++j) {
                x[static_cast<std::size_t>(j)] = uniform01(rng);
            }
            ssum += f.value(x, a);
        }
        const double err = ssum / N - f.reference(a);
        sumSq += err * err;
    }
    return std::sqrt(sumSq / replications);
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string input, base;
    int sample = 8;
    int mLevel = 12;
    int shifts = 16;
    std::uint64_t seed = 0xABCDEF123456789ULL;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            const auto pos = arg.find('=');
            return pos == std::string::npos ? std::string() : arg.substr(pos + 1);
        };
        if (arg.rfind("--input=", 0) == 0)
            input = value();
        else if (arg.rfind("--base=", 0) == 0)
            base = value();
        else if (arg.rfind("--sample=", 0) == 0)
            sample = std::stoi(value());
        else if (arg.rfind("--m=", 0) == 0)
            mLevel = std::stoi(value());
        else if (arg.rfind("--shifts=", 0) == 0)
            shifts = std::stoi(value());
        else if (arg.rfind("--seed=", 0) == 0)
            seed = std::stoull(value());
        else if (arg == "--help" || arg == "-h") {
            std::printf("usage: analyze_sobol --input=FILE [--base=FILE] [--sample=K] [--m=N] "
                        "[--shifts=R] [--seed=S]\n");
            return 0;
        } else {
            QTA_LOG_ERROR("quantape.tools", "unknown option: {}", arg);
            return 1;
        }
    }
    if (input.empty()) {
        QTA_LOG_ERROR("quantape.tools", "--input required");
        return 1;
    }
    const auto entries = load_joe_kuo(input);
    if (entries.empty()) {
        QTA_LOG_ERROR("quantape.tools", "cannot load {}", input);
        return 1;
    }
    const auto prefix = base.empty() ? std::vector<Entry>{} : load_joe_kuo(base);
    const std::uint32_t firstNew =
        prefix.empty() ? 2 : static_cast<std::uint32_t>(prefix.size() + 2);
    const std::uint32_t lastDim = entries.back().dim;
    const std::uint32_t nNew = lastDim >= firstNew ? lastDim - firstNew + 1 : 0;
    std::printf("analyze: %zu entries (dims 2..%u), extension dims %u, %u new%s\n", entries.size(),
                lastDim, firstNew, nNew, base.empty() ? "" : " (base given)");

    // Direction matrices.
    std::vector<DirectionMatrix> mats(entries.size() + 2);
    mats[0] = identityMatrix();
    for (std::size_t i = 0; i < entries.size(); ++i) {
        mats[i + 1] = directionMatrix(entries[i]);
    }
    const int L = mLevel;
    const std::uint32_t N = 1u << L;

    // Sampled dimension indices (matrix indices) in the extension range.
    std::vector<std::size_t> dims;
    for (int s = 0; s < sample; ++s) {
        const std::size_t idx =
            firstNew - 1 +
            static_cast<std::size_t>((static_cast<std::uint64_t>(s) * nNew) / sample);
        dims.push_back(std::min<std::size_t>(idx, mats.size() - 2));
    }
    if (!dims.empty()) {
        dims.front() = firstNew - 1;
    }

    std::mt19937_64 rng(seed);

    // ── 1. Digital-net exactness 2D/3D/4D ──
    {
        int tested = 0, failed = 0;
        const int sizes[] = {2, 3, 4};
        for (std::size_t di = 0; di < dims.size() && di < 12; ++di) {
            const std::size_t centre = dims[di];
            for (int ti = 0; ti < 3; ++ti) {
                const int s = sizes[ti];
                std::vector<std::size_t> idxs;
                idxs.push_back(centre);
                idxs.push_back(centre >= 1 ? centre - 1 : 0);
                if (s >= 3) {
                    idxs.push_back(centre >= 2 ? centre - 2 : 0);
                }
                if (s >= 4) {
                    idxs.push_back(centre >= 8 ? centre - 8 : 0);
                }
                std::vector<DirectionMatrix> ms;
                for (std::size_t i : idxs) {
                    ms.push_back(mats[i]);
                }
                const int t = tValueS(ms, mLevel);
                int budget = mLevel - t;
                if (budget < s) {
                    continue;
                }
                std::vector<int> b(static_cast<std::size_t>(s), 0);
                for (int q = 0; q < budget; ++q) {
                    b[static_cast<std::size_t>(q % s)]++;
                }
                std::vector<std::vector<std::uint32_t>> X;
                for (std::size_t i : idxs) {
                    X.push_back(generate(
                        i == 0 ? dimensionOneWords(L) : directionWords(entries[i - 1], L), L, 0));
                }
                std::uint32_t worst = 0;
                ++tested;
                if (!exactGridS(X, b, 1u << t, &worst)) {
                    ++failed;
                    std::string dimsText;
                    for (std::size_t i : idxs) {
                        dimsText += " " + std::to_string(i + 1);
                    }
                    QTA_LOG_WARN("quantape.tools",
                                 "exactness FAIL dims{} t={} worst={} expected={}", dimsText, t,
                                 worst, 1u << t);
                }
            }
        }
        std::printf("net exactness 2D/3D/4D: %d projections tested, %d failed -> %s\n", tested,
                    failed, failed ? "FAIL" : "ok");

        // Self-tests: tValueS(s=2) must equal tValue2D; dims 1..3 must be exact.
        {
            const std::size_t d1 = dims.front(), d2 = mats.size() - 2;
            const int t2 = tValue2D(mats[d1], mats[d2], mLevel);
            const int ts = tValueS({mats[d1], mats[d2]}, mLevel);
            std::printf("self-test tValueS(2D)==tValue2D: %s (t=%d)\n", t2 == ts ? "ok" : "FAIL",
                        t2);
            std::vector<DirectionMatrix> m3 = {mats[0], mats[1], mats[2]};
            const int t3 = tValueS(m3, 10);
            std::vector<int> b3(3, (10 - t3) / 3);
            for (int q = 0; q < (10 - t3) - 3 * b3[0]; ++q) {
                b3[static_cast<std::size_t>(q)]++;
            }
            std::vector<std::vector<std::uint32_t>> X;
            X.push_back(generate(dimensionOneWords(10), 10, 0));
            X.push_back(generate(directionWords(entries[0], 10), 10, 0));
            X.push_back(generate(directionWords(entries[1], 10), 10, 0));
            std::uint32_t worst = 0;
            std::printf("self-test dims 1..3 exact(2^10): %s (t=%d worst=%u expected=%u)\n",
                        exactGridS(X, b3, 1u << t3, &worst) ? "ok" : "FAIL", t3, worst, 1u << t3);
        }
    }

    // ── 2. L2 star discrepancy (2D) QMC vs random ──
    {
        const int mD = std::min(mLevel, 12);
        const std::uint32_t ND = 1u << mD;
        double qmcMean = 0.0, rndMean = 0.0;
        int count = 0;
        for (std::size_t di = 0; di + 1 < dims.size() && di < 6; ++di) {
            const std::size_t d1 = dims[di], d2 = dims[di + 1];
            auto x1 = generate(
                d1 == 0 ? dimensionOneWords(mD) : directionWords(entries[d1 - 1], mD), mD, 0);
            auto x2 = generate(
                d2 == 0 ? dimensionOneWords(mD) : directionWords(entries[d2 - 1], mD), mD, 0);
            std::vector<double> dx(ND), dy(ND);
            for (std::uint32_t i = 0; i < ND; ++i) {
                dx[i] = x1[i] / 4294967296.0;
                dy[i] = x2[i] / 4294967296.0;
            }
            qmcMean += std::sqrt(l2StarDiscrepancy2D(dx, dy));
            std::vector<double> rx(ND), ry(ND);
            for (std::uint32_t i = 0; i < ND; ++i) {
                rx[i] = uniform01(rng);
                ry[i] = uniform01(rng);
            }
            rndMean += std::sqrt(l2StarDiscrepancy2D(rx, ry));
            ++count;
        }
        if (count > 0) {
            std::printf(
                "L2 star discrepancy (2D, N=2^%d): QMC mean %.4g vs random %.4g (ratio %.3g)\n", mD,
                qmcMean / count, rndMean / count, qmcMean / rndMean);
        }
    }

    // ── 3. Correlations of sampled dimension pairs ──
    {
        double maxAbs = 0.0, sumAbs = 0.0;
        int count = 0;
        for (std::size_t di = 0; di + 1 < dims.size(); ++di) {
            const std::size_t d1 = dims[di], d2 = dims[di + 1];
            auto x1 =
                generate(d1 == 0 ? dimensionOneWords(L) : directionWords(entries[d1 - 1], L), L, 0);
            auto x2 =
                generate(d2 == 0 ? dimensionOneWords(L) : directionWords(entries[d2 - 1], L), L, 0);
            const double r = pearson(x1, x2);
            maxAbs = std::max(maxAbs, std::fabs(r));
            sumAbs += std::fabs(r);
            ++count;
        }
        std::printf("pair correlations: %d pairs, max|r|=%.4g mean|r|=%.4g\n", count, maxAbs,
                    count ? sumAbs / count : 0.0);
    }

    // ── 4. QMC vs MC integration, dimensions 1..8, prefix dims ──
    {
        const Integrand funcs[] = {
            {"gaussian", gaussianValue, gaussianRef},
            {"product-peak", productPeakValue, productPeakRef},
            {"oscillatory", oscillatoryValue, oscillatoryRef},
            {"exponential", exponentialValue, exponentialRef},
            {"kink", kinkValue, kinkRef},
        };
        std::printf("QMC/MC RMSE ratios (N=2^%d, %d shifts/replications, prefix dims 1..s)\n",
                    mLevel, shifts);
        for (int s = 1; s <= 8; s += 3) {
            std::vector<double> a(static_cast<std::size_t>(s));
            for (int j = 0; j < s; ++j) {
                a[static_cast<std::size_t>(j)] = (j + 1.0) / (s + 1.0);
            }
            std::vector<std::vector<std::uint32_t>> baseDims;
            for (int j = 0; j < s; ++j) {
                baseDims.push_back(j == 0 ? generate(dimensionOneWords(L), L, 0)
                                          : generate(directionWords(entries[j - 1], L), L, 0));
            }
            std::vector<std::vector<std::uint32_t>> shiftVecs(static_cast<std::size_t>(shifts));
            for (auto& sv : shiftVecs) {
                sv.resize(static_cast<std::size_t>(s));
                for (auto& v : sv) {
                    v = static_cast<std::uint32_t>(rng());
                }
            }
            std::printf("  s=%d:", s);
            for (const auto& f : funcs) {
                const double eq = qmcError(f, baseDims, shiftVecs, a, rng);
                const double em = mcError(f, s, mLevel, shifts, a, rng);
                std::printf("  %s=%.2f", f.name, em / eq);
            }
            std::printf("\n");
        }
    }

    // ── 5. GBM Asian call from extended dimensions (time-ordered) ──
    if (nNew > 0) {
        const int steps = 64;
        const double S0 = 100.0, r = 0.05, sigma = 0.20, T = 1.0, K = 100.0;
        const double dt = T / steps;
        const double mu = (r - 0.5 * sigma * sigma) * dt;
        const double sd = sigma * std::sqrt(dt);
        const auto payoff = [&](const std::vector<double>& z) {
            double s = S0, sum = 0.0;
            for (int i = 0; i < steps; ++i) {
                s *= std::exp(mu + sd * z[static_cast<std::size_t>(i)]);
                sum += s;
            }
            const double avg = sum / steps;
            return std::max(avg - K, 0.0);
        };
        // Reference: plain MC with many paths (using Box-Muller-free normal via sum of 12
        // uniforms).
        const auto normalFrom = [&](std::mt19937_64& g) { return inverseNormal(uniform01(g)); };
        const std::size_t nRef = std::size_t(1) << 20;
        double refSum = 0;
        for (std::size_t i = 0; i < nRef; ++i) {
            std::vector<double> z(steps);
            for (auto& v : z) {
                v = normalFrom(rng);
            }
            refSum += payoff(z);
        }
        const double ref = std::exp(-r * T) * refSum / nRef;
        // QMC: inverse-normal from Sobol points on the first `steps` extension dims.
        std::vector<std::vector<std::uint32_t>> baseDims;
        for (int j = 0; j < steps; ++j) {
            baseDims.push_back(generate(
                directionWords(entries[static_cast<std::size_t>(firstNew - 1 + j) - 1], L), L, 0));
        }
        double qSum = 0, qSumSq = 0;
        std::vector<double> z(steps);
        for (int sh = 0; sh < shifts; ++sh) {
            std::uint32_t sv[64];
            for (auto& v : sv) {
                v = static_cast<std::uint32_t>(rng());
            }
            double ssum = 0;
            for (std::uint32_t i = 0; i < N; ++i) {
                for (int j = 0; j < steps; ++j) {
                    const double u =
                        (static_cast<double>(baseDims[static_cast<std::size_t>(j)][i] ^ sv[j]) +
                         0.5) /
                        4294967296.0;
                    z[static_cast<std::size_t>(j)] = inverseNormal(u);
                }
                ssum += payoff(z);
            }
            const double est = std::exp(-r * T) * ssum / N;
            qSum += est;
            qSumSq += est * est;
        }
        const double qMean = qSum / shifts;
        const double qSe = std::sqrt(std::max(qSumSq / shifts - qMean * qMean, 0.0) / (shifts - 1));
        {
            std::printf("  per-shift QMC estimates:");
            std::vector<double> base2;
            for (int sh = 0; sh < shifts; ++sh) {
                std::uint32_t sv[64];
                for (auto& v : sv) {
                    v = static_cast<std::uint32_t>(rng());
                }
                double ssum = 0;
                for (std::uint32_t i = 0; i < N; ++i) {
                    for (int j = 0; j < steps; ++j) {
                        const double u =
                            (static_cast<double>(baseDims[static_cast<std::size_t>(j)][i] ^ sv[j]) +
                             0.5) /
                            4294967296.0;
                        z[static_cast<std::size_t>(j)] = inverseNormal(u);
                    }
                    ssum += payoff(z);
                }
                std::printf(" %.3f", std::exp(-r * T) * ssum / N);
            }
            std::printf("\n");
        }
        // Control: same Asian payoff using prefix dims 1..64 (untouched Joe-Kuo).
        {
            std::vector<std::vector<std::uint32_t>> pre;
            for (int j = 0; j < steps; ++j) {
                pre.push_back(j == 0 ? generate(dimensionOneWords(L), L, 0)
                                     : generate(directionWords(entries[j - 1], L), L, 0));
            }
            double ssum = 0, ssq = 0;
            for (int sh = 0; sh < shifts; ++sh) {
                std::uint32_t sv[64];
                for (auto& v : sv) {
                    v = static_cast<std::uint32_t>(rng());
                }
                double est = 0;
                for (std::uint32_t i = 0; i < N; ++i) {
                    for (int j = 0; j < steps; ++j) {
                        const double u =
                            (static_cast<double>(pre[static_cast<std::size_t>(j)][i] ^ sv[j]) +
                             0.5) /
                            4294967296.0;
                        z[static_cast<std::size_t>(j)] = inverseNormal(u);
                    }
                    est += payoff(z);
                }
                est = std::exp(-r * T) * est / N;
                ssum += est;
                ssq += est * est;
            }
            const double pm = ssum / shifts;
            const double ps = std::sqrt(std::max(ssq / shifts - pm * pm, 0.0) / (shifts - 1));
            std::printf("  control: prefix dims 1..64 Asian QMC %.4f (+-%.2g)\n", pm, ps);
        }

        // MC comparison with the same total number of function evaluations.
        double mSum = 0, mSumSq = 0;
        for (int rep = 0; rep < shifts; ++rep) {
            double ssum = 0;
            for (std::uint32_t i = 0; i < N; ++i) {
                for (auto& v : z) {
                    v = normalFrom(rng);
                }
                ssum += payoff(z);
            }
            const double est = std::exp(-r * T) * ssum / N;
            mSum += est;
            mSumSq += est * est;
        }
        const double mMean = mSum / shifts;
        const double mSe = std::sqrt(std::max(mSumSq / shifts - mMean * mMean, 0.0) / (shifts - 1));
        std::printf(
            "GBM Asian call (64 steps from extended dims %u..%u): reference %.4f | QMC %.4f "
            "(+-%.2g) | MC %.4f (+-%.2g) | SE ratio %.1fx\n",
            firstNew, firstNew + steps - 1, ref, qMean, qSe, mMean, mSe, mSe / qSe);

        // Brownian-bridge ordering: endpoint first, then dyadic midpoints.
        {
            std::vector<int> order; // times in the order their normals are used
            order.push_back(steps); // endpoint
            std::vector<std::pair<int, int>> intervals{{0, steps}};
            while ((int)intervals.size() < steps) {
                std::vector<std::pair<int, int>> next;
                for (auto [l, r] : intervals) {
                    const int mid = (l + r) / 2;
                    if (mid != l && mid != r && (int)order.size() < steps) {
                        order.push_back(mid);
                        next.push_back({l, mid});
                        next.push_back({mid, r});
                    }
                }
                intervals = next;
                if (intervals.empty()) {
                    break;
                }
            }
            const auto bridgePayoff = [&](const std::vector<double>& zz) {
                std::vector<double> W(steps + 1, 0.0);
                W[steps] = std::sqrt(T) * zz[0];
                std::size_t zi = 1;
                std::vector<std::pair<int, int>> iv{{0, steps}};
                while (zi < static_cast<std::size_t>(steps)) {
                    std::vector<std::pair<int, int>> nx;
                    for (auto [l, r] : iv) {
                        const int mid = (l + r) / 2;
                        if (mid != l && mid != r && zi < static_cast<std::size_t>(steps)) {
                            W[mid] = 0.5 * (W[l] + W[r]) +
                                     std::sqrt(T * (r - l) / (4.0 * steps)) * zz[zi++];
                            nx.push_back({l, mid});
                            nx.push_back({mid, r});
                        }
                    }
                    iv = nx;
                    if (iv.empty()) {
                        break;
                    }
                }
                double sv = S0, acc = 0;
                for (int i = 1; i <= steps; ++i) {
                    sv *= std::exp((r - 0.5 * sigma * sigma) * dt + sigma * (W[i] - W[i - 1]));
                    acc += sv;
                }
                return std::max(acc / steps - K, 0.0);
            };
            double sum = 0, sq = 0;
            for (int sh = 0; sh < shifts; ++sh) {
                std::uint32_t sv2[64];
                for (auto& v : sv2) {
                    v = static_cast<std::uint32_t>(rng());
                }
                double ssum = 0;
                for (std::uint32_t i = 0; i < N; ++i) {
                    std::vector<double> zz(steps);
                    for (int j = 0; j < steps; ++j) {
                        const double u = (static_cast<double>(
                                              baseDims[static_cast<std::size_t>(j)][i] ^ sv2[j]) +
                                          0.5) /
                                         4294967296.0;
                        zz[static_cast<std::size_t>(j)] = inverseNormal(u);
                    }
                    ssum += bridgePayoff(zz);
                }
                const double est = std::exp(-r * T) * ssum / N;
                sum += est;
                sq += est * est;
            }
            const double bm = sum / shifts;
            const double bs = std::sqrt(std::max(sq / shifts - bm * bm, 0.0) / (shifts - 1));
            double msum = 0, msq = 0;
            for (int rep = 0; rep < shifts; ++rep) {
                double ssum = 0;
                for (std::uint32_t i = 0; i < N; ++i) {
                    std::vector<double> zz(steps);
                    for (auto& v : zz) {
                        v = normalFrom(rng);
                    }
                    ssum += bridgePayoff(zz);
                }
                const double est = std::exp(-r * T) * ssum / N;
                msum += est;
                msq += est * est;
            }
            const double mm = msum / shifts;
            const double ms = std::sqrt(std::max(msq / shifts - mm * mm, 0.0) / (shifts - 1));
            std::printf(
                "GBM Asian call, Brownian bridge (extended dims): QMC %.4f (+-%.2g) | MC %.4f "
                "(+-%.2g) | SE ratio %.1fx\n",
                bm, bs, mm, ms, ms / bs);
        }

        // Terminal GBM (1 step) sanity: E[exp(mu + sigma*Z)] = exp(sigma^2/2).
        for (std::size_t dim : {dims.front(), static_cast<std::size_t>(mats.size() - 2)}) {
            const auto base = generate(
                dim == 0 ? dimensionOneWords(L) : directionWords(entries[dim - 1], L), L, 0);
            double ssum = 0;
            for (std::uint32_t i = 0; i < N; ++i) {
                const double u = (static_cast<double>(base[i]) + 0.5) / 4294967296.0;
                ssum += std::exp(sigma * std::sqrt(T) * inverseNormal(u));
            }
            std::printf("terminal-GBM sanity dim %zu: QMC mean %.6f vs analytic %.6f\n", dim + 1,
                        ssum / N, std::exp(0.5 * sigma * sigma * T));
        }
    }

    std::printf("ANALYSIS COMPLETE\n");
    return 0;
}
