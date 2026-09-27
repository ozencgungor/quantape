#ifndef QUANTAPE_MC_FORWARD_SCALAR_H
#define QUANTAPE_MC_FORWARD_SCALAR_H

#include <array>
#include <cmath>
#include <cstddef>

namespace quantape::mc {
/**
 * @file ForwardScalar.h
 * @brief Fixed-size forward-mode dual number (value + N tangents)
 *
 * `Tangent<Scalar, N>` carries the value and `N` directional derivatives
 * of every intermediate quantity. The SDE engine is already scalar-generic
 * and uses ADL-friendly math (`using std::exp; exp(x)`), so plugging this
 * type in as the scheme Scalar gives pathwise values and all sensitivities
 * in **one forward pass, with no tape and no allocations**:
 *
 *     d/dθ E[π(X)] = E[ d/dθ π(X) ],      d = (d + p) directions
 *
 * Cost is ~(N + 1)x a plain `double` path (no AD nodes, no reverse sweep),
 * which beats per-path reverse tapes for small parameter counts (N <= ~8)
 * — measured several-fold on the GBM/Heston-scale cases; reverse mode
 * remains the right tool for large parameter vectors.
 *
 * Every loop over `N` is branchless and contiguous, so the value/tangent
 * updates vectorize; control-flow comparisons are on the primal value
 * (primal-pinned branches, the same a.e. contract as reverse mode).
 *
 * ADL note: all overloads live in `quantape::mc`, the namespace of the
 * argument, so unqualified calls in the engine functors find them.
 */

template <typename Scalar, std::size_t N>
struct Tangent {
    Scalar value{};
    std::array<Scalar, N> d{};

    Tangent() = default;
    /// Implicit conversion from a plain scalar (zero tangents).
    Tangent(const Scalar& v) : value(v), d{} {}
    Tangent(const Scalar& v, const std::array<Scalar, N>& deriv) : value(v), d(deriv) {}

    Scalar& tangent(std::size_t i) { return d[i]; }
    const Scalar& tangent(std::size_t i) const { return d[i]; }
};

// ── compound assignment ──

template <typename S, std::size_t N>
inline Tangent<S, N>& operator+=(Tangent<S, N>& a, const Tangent<S, N>& b) {
    a.value += b.value;
    for (std::size_t i = 0; i < N; ++i) {
        a.d[i] += b.d[i];
    }
    return a;
}

template <typename S, std::size_t N>
inline Tangent<S, N>& operator-=(Tangent<S, N>& a, const Tangent<S, N>& b) {
    a.value -= b.value;
    for (std::size_t i = 0; i < N; ++i) {
        a.d[i] -= b.d[i];
    }
    return a;
}

template <typename S, std::size_t N>
inline Tangent<S, N>& operator*=(Tangent<S, N>& a, const Tangent<S, N>& b) {
    for (std::size_t i = 0; i < N; ++i) {
        a.d[i] = a.d[i] * b.value + a.value * b.d[i];
    }
    a.value *= b.value;
    return a;
}

template <typename S, std::size_t N>
inline Tangent<S, N>& operator/=(Tangent<S, N>& a, const Tangent<S, N>& b) {
    a.value /= b.value;
    for (std::size_t i = 0; i < N; ++i) {
        a.d[i] /= b.value;
    }
    return a;
}

// ── binary arithmetic ──

template <typename S, std::size_t N>
inline Tangent<S, N> operator+(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    Tangent<S, N> out;
    out.value = a.value + b.value;
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = a.d[i] + b.d[i];
    }
    return out;
}

template <typename S, std::size_t N>
inline Tangent<S, N> operator-(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    Tangent<S, N> out;
    out.value = a.value - b.value;
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = a.d[i] - b.d[i];
    }
    return out;
}

template <typename S, std::size_t N>
inline Tangent<S, N> operator*(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    Tangent<S, N> out;
    out.value = a.value * b.value;
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = a.d[i] * b.value + a.value * b.d[i];
    }
    return out;
}

template <typename S, std::size_t N>
inline Tangent<S, N> operator/(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    Tangent<S, N> out;
    const S inv = S(1.0) / b.value;
    out.value = a.value * inv;
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = (a.d[i] - out.value * b.d[i]) * inv;
    }
    return out;
}

template <typename S, std::size_t N>
inline Tangent<S, N> operator-(const Tangent<S, N>& a) {
    Tangent<S, N> out;
    out.value = -a.value;
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = -a.d[i];
    }
    return out;
}

// mixed scalar arithmetic (Scalar = double)
template <typename S, std::size_t N>
inline Tangent<S, N> operator+(const Tangent<S, N>& a, const S& b) {
    return a + Tangent<S, N>(b);
}
template <typename S, std::size_t N>
inline Tangent<S, N> operator+(const S& a, const Tangent<S, N>& b) {
    return Tangent<S, N>(a) + b;
}
template <typename S, std::size_t N>
inline Tangent<S, N> operator-(const Tangent<S, N>& a, const S& b) {
    return a - Tangent<S, N>(b);
}
template <typename S, std::size_t N>
inline Tangent<S, N> operator-(const S& a, const Tangent<S, N>& b) {
    return Tangent<S, N>(a) - b;
}
template <typename S, std::size_t N>
inline Tangent<S, N> operator*(const Tangent<S, N>& a, const S& b) {
    return a * Tangent<S, N>(b);
}
template <typename S, std::size_t N>
inline Tangent<S, N> operator*(const S& a, const Tangent<S, N>& b) {
    return Tangent<S, N>(a) * b;
}
template <typename S, std::size_t N>
inline Tangent<S, N> operator/(const Tangent<S, N>& a, const S& b) {
    return a / Tangent<S, N>(b);
}
template <typename S, std::size_t N>
inline Tangent<S, N> operator/(const S& a, const Tangent<S, N>& b) {
    return Tangent<S, N>(a) / b;
}

// ── comparisons: primal value decides the branch (pinned) ──

template <typename S, std::size_t N>
inline bool operator<(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value < b.value;
}
template <typename S, std::size_t N>
inline bool operator>(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value > b.value;
}
template <typename S, std::size_t N>
inline bool operator<=(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value <= b.value;
}
template <typename S, std::size_t N>
inline bool operator>=(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value >= b.value;
}
template <typename S, std::size_t N>
inline bool operator==(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value == b.value;
}
template <typename S, std::size_t N>
inline bool operator!=(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value != b.value;
}

// ── elementary functions ──

template <typename S, std::size_t N>
inline Tangent<S, N> sqrt(const Tangent<S, N>& x) {
    using std::sqrt;
    Tangent<S, N> out;
    out.value = sqrt(x.value);
    const S scale = S(0.5) / out.value;
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
inline Tangent<S, N> exp(const Tangent<S, N>& x) {
    using std::exp;
    Tangent<S, N> out;
    out.value = exp(x.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * out.value;
    }
    return out;
}

template <typename S, std::size_t N>
inline Tangent<S, N> log(const Tangent<S, N>& x) {
    using std::log;
    Tangent<S, N> out;
    out.value = log(x.value);
    const S scale = S(1.0) / x.value;
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
inline Tangent<S, N> fabs(const Tangent<S, N>& x) {
    using std::fabs;
    Tangent<S, N> out;
    out.value = fabs(x.value);
    const S sign = x.value >= S(0.0) ? S(1.0) : S(-1.0);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * sign;
    }
    return out;
}

template <typename S, std::size_t N>
inline Tangent<S, N> fmin(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value <= b.value ? a : b;
}

template <typename S, std::size_t N>
inline Tangent<S, N> fmax(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value >= b.value ? a : b;
}

// std::abs compatibility
template <typename S, std::size_t N>
inline Tangent<S, N> abs(const Tangent<S, N>& x) {
    return fabs(x);
}

} // namespace quantape::mc

// Eigen scalar traits so `Matrix<Tangent>` gets the generic code paths.
namespace Eigen {
template <typename S, std::size_t N>
struct NumTraits<quantape::mc::Tangent<S, N>> : NumTraits<double> {
    using Real = quantape::mc::Tangent<S, N>;
    using NonInteger = quantape::mc::Tangent<S, N>;
    using Nested = quantape::mc::Tangent<S, N>;
    using Literal = quantape::mc::Tangent<S, N>;
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 4,
        MulCost = 8,
    };
};
} // namespace Eigen

#endif // QUANTAPE_MC_FORWARD_SCALAR_H
