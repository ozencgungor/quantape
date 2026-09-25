#ifndef PROBE_DUAL_H
#define PROBE_DUAL_H

#include <cmath>
#include <vector>

namespace quantape::math {
namespace detail {

/**
 * @brief Forward-mode dual for branch-pinned local Jacobians.
 *
 * Carries a primal value v and a gradient vector d. ALL branch decisions
 * (comparisons, abs sign) use the PRIMAL only, so evaluating the
 * coefficient computation with ProbeDual y-values linearizes the ACTIVE
 * branch of a piecewise-linear interpolation method — the exact adjoint
 * the tape would produce (a subgradient at kinks), without putting any
 * arithmetic on the tape.
 */
struct ProbeDual {
    double v = 0.0;
    std::vector<double> d;

    explicit ProbeDual(double value = 0.0, size_t gradSize = 0) : v(value), d(gradSize, 0.0) {}

    ProbeDual operator-() const {
        // Negate BOTH value and derivatives. (An earlier version forgot the
        // value, which made abs() return negatives and silently corrupted
        // every dual-path coefficient.)
        ProbeDual r(-v, d.size());
        for (size_t k = 0; k < d.size(); ++k)
            r.d[k] = -d[k];
        return r;
    }
    // Binary ops are SIZE-TOLERANT: mismatched gradient sizes are treated
    // as zero-padded (a scalar literal like ProbeDual(0.0) has size 0 and
    // acts as a constant with zero gradient). This makes expressions like
    // T(2.0) / (T(1.0)/S[i-1] + T(1.0)/S[i]) well-defined for T=ProbeDual
    // without forcing every literal to carry the full grid size.
    static size_t maxSize(size_t a, size_t b) { return a > b ? a : b; }

    ProbeDual operator+(const ProbeDual& o) const {
        const size_t m = maxSize(d.size(), o.d.size());
        ProbeDual r(v + o.v, m);
        for (size_t k = 0; k < m; ++k)
            r.d[k] = (k < d.size() ? d[k] : 0.0) + (k < o.d.size() ? o.d[k] : 0.0);
        return r;
    }
    ProbeDual operator-(const ProbeDual& o) const {
        const size_t m = maxSize(d.size(), o.d.size());
        ProbeDual r(v - o.v, m);
        for (size_t k = 0; k < m; ++k)
            r.d[k] = (k < d.size() ? d[k] : 0.0) - (k < o.d.size() ? o.d[k] : 0.0);
        return r;
    }
    ProbeDual operator*(const ProbeDual& o) const {
        const size_t m = maxSize(d.size(), o.d.size());
        ProbeDual r(v * o.v, m);
        for (size_t k = 0; k < m; ++k)
            r.d[k] = (k < d.size() ? d[k] : 0.0) * o.v + v * (k < o.d.size() ? o.d[k] : 0.0);
        return r;
    }
    ProbeDual operator/(const ProbeDual& o) const {
        const size_t m = maxSize(d.size(), o.d.size());
        ProbeDual r(v / o.v, m);
        for (size_t k = 0; k < m; ++k)
            r.d[k] = ((k < d.size() ? d[k] : 0.0) * o.v - v * (k < o.d.size() ? o.d[k] : 0.0)) /
                     (o.v * o.v);
        return r;
    }

    // mixed double arithmetic
    ProbeDual operator+(double x) const { return *this + ProbeDual(x, d.size()); }
    ProbeDual operator-(double x) const { return *this - ProbeDual(x, d.size()); }
    ProbeDual operator*(double x) const {
        ProbeDual r(v * x, d.size());
        for (size_t k = 0; k < d.size(); ++k)
            r.d[k] = d[k] * x;
        return r;
    }
    ProbeDual operator/(double x) const { return *this / ProbeDual(x, d.size()); }
};

inline ProbeDual operator+(double x, const ProbeDual& o) {
    return o + x;
}
inline ProbeDual operator-(double x, const ProbeDual& o) {
    ProbeDual r(x - o.v, o.d.size());
    for (size_t k = 0; k < o.d.size(); ++k)
        r.d[k] = -o.d[k];
    return r;
}
inline ProbeDual operator*(double x, const ProbeDual& o) {
    return o * x;
}
inline ProbeDual operator/(double x, const ProbeDual& o) {
    ProbeDual r(x / o.v, o.d.size());
    for (size_t k = 0; k < o.d.size(); ++k)
        r.d[k] = -x * o.d[k] / (o.v * o.v);
    return r;
}

// abs with the PRIMAL sign (subgradient 0 at the kink) — the branch-pinning
// is exactly this: comparisons use v, never d.
inline ProbeDual abs(const ProbeDual& x) {
    if (x.v > 0.0)
        return x;
    if (x.v < 0.0)
        return -x;
    ProbeDual r(0.0, x.d.size());
    return r;
}

inline double primal(const ProbeDual& x) {
    return x.v;
}
inline double primal(double x) {
    return x;
}

/// Smooth approximation of |u|: sqrt(u^2 + eps^2) — C^1, subgradient-free.
inline ProbeDual smoothAbs(const ProbeDual& x, double eps = 1e-8) {
    const double mag = std::sqrt(x.v * x.v + eps * eps);
    ProbeDual r(mag, x.d.size());
    for (size_t k = 0; k < x.d.size(); ++k)
        r.d[k] = x.v * x.d[k] / mag;
    return r;
}
inline double smoothAbs(double x, double eps = 1e-8) {
    return std::sqrt(x * x + eps * eps);
}

/// Sigmoid blend for sign-based branch conditions: sigma(kappa * p).
inline double sigmoid(double p, double kappa) {
    const double z = kappa * p;
    return 1.0 / (1.0 + std::exp(-std::min(std::max(z, -50.0), 50.0)));
}

} // namespace detail
} // namespace quantape::math

#endif // PROBE_DUAL_H
