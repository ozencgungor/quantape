#ifndef LIGHTAPE_FWD_FORWARD_SCALAR_H
#define LIGHTAPE_FWD_FORWARD_SCALAR_H

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "lightape/base/Digamma.h"
#include "lightape/base/ScalarConstant.h"

namespace lightape {
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
 * ADL note: all overloads live in `lightape`, the namespace of the
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
Tangent<S, N>& operator+=(Tangent<S, N>& a, const Tangent<S, N>& b) {
    a.value += b.value;
    for (std::size_t i = 0; i < N; ++i) {
        a.d[i] += b.d[i];
    }
    return a;
}

template <typename S, std::size_t N>
Tangent<S, N>& operator-=(Tangent<S, N>& a, const Tangent<S, N>& b) {
    a.value -= b.value;
    for (std::size_t i = 0; i < N; ++i) {
        a.d[i] -= b.d[i];
    }
    return a;
}

template <typename S, std::size_t N>
Tangent<S, N>& operator*=(Tangent<S, N>& a, const Tangent<S, N>& b) {
    for (std::size_t i = 0; i < N; ++i) {
        a.d[i] = a.d[i] * b.value + a.value * b.d[i];
    }
    a.value *= b.value;
    return a;
}

template <typename S, std::size_t N>
Tangent<S, N>& operator/=(Tangent<S, N>& a, const Tangent<S, N>& b) {
    const S inv = S(1.0) / b.value;
    a.value *= inv;
    for (std::size_t i = 0; i < N; ++i) {
        a.d[i] = (a.d[i] - a.value * b.d[i]) * inv;
    }
    return a;
}

// ── binary arithmetic ──

template <typename S, std::size_t N>
Tangent<S, N> operator+(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    Tangent<S, N> out;
    out.value = a.value + b.value;
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = a.d[i] + b.d[i];
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> operator-(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    Tangent<S, N> out;
    out.value = a.value - b.value;
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = a.d[i] - b.d[i];
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> operator*(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    Tangent<S, N> out;
    out.value = a.value * b.value;
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = a.d[i] * b.value + a.value * b.d[i];
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> operator/(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    Tangent<S, N> out;
    const S inv = S(1.0) / b.value;
    out.value = a.value * inv;
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = (a.d[i] - out.value * b.d[i]) * inv;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> operator-(const Tangent<S, N>& a) {
    Tangent<S, N> out;
    out.value = -a.value;
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = -a.d[i];
    }
    return out;
}

// mixed scalar arithmetic (Scalar = double)
template <typename S, std::size_t N>
Tangent<S, N> operator+(const Tangent<S, N>& a, const S& b) {
    return a + Tangent<S, N>(b);
}
template <typename S, std::size_t N>
Tangent<S, N> operator+(const S& a, const Tangent<S, N>& b) {
    return Tangent<S, N>(a) + b;
}
template <typename S, std::size_t N>
Tangent<S, N> operator-(const Tangent<S, N>& a, const S& b) {
    return a - Tangent<S, N>(b);
}
template <typename S, std::size_t N>
Tangent<S, N> operator-(const S& a, const Tangent<S, N>& b) {
    return Tangent<S, N>(a) - b;
}
template <typename S, std::size_t N>
Tangent<S, N> operator*(const Tangent<S, N>& a, const S& b) {
    return a * Tangent<S, N>(b);
}
template <typename S, std::size_t N>
Tangent<S, N> operator*(const S& a, const Tangent<S, N>& b) {
    return Tangent<S, N>(a) * b;
}
template <typename S, std::size_t N>
Tangent<S, N> operator/(const Tangent<S, N>& a, const S& b) {
    return a / Tangent<S, N>(b);
}
template <typename S, std::size_t N>
Tangent<S, N> operator/(const S& a, const Tangent<S, N>& b) {
    return Tangent<S, N>(a) / b;
}

// mixed arithmetic with a plain double: `3 * sin(x)`, `x + 0.5` in generic
// code where `S` is an AD scalar. The `const S&` overloads above cannot deduce
// `S` from a bare double against `const Tangent<S, N>&` (deduction would have
// to yield both `double` and `RevScalar`), so these take the literal as a
// `double` and build the scalar constant one nesting level at a time through
// `detail::constantOf` (RevScalar constants stay node-less). The binary forms
// are disabled for `S = double`, where the `const S&` overloads already apply
// (enabling both would be ambiguous); the compound-assignment forms have no
// `const S&` counterpart and stay enabled for every `S`.
template <typename S, std::size_t N>
Tangent<S, N>& operator+=(Tangent<S, N>& a, double b) {
    a += Tangent<S, N>(detail::constantOf<S>(b));
    return a;
}
template <typename S, std::size_t N>
Tangent<S, N>& operator-=(Tangent<S, N>& a, double b) {
    a -= Tangent<S, N>(detail::constantOf<S>(b));
    return a;
}
template <typename S, std::size_t N>
Tangent<S, N>& operator*=(Tangent<S, N>& a, double b) {
    a *= Tangent<S, N>(detail::constantOf<S>(b));
    return a;
}
template <typename S, std::size_t N>
Tangent<S, N>& operator/=(Tangent<S, N>& a, double b) {
    a /= Tangent<S, N>(detail::constantOf<S>(b));
    return a;
}

template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> operator+(const Tangent<S, N>& a, double b) {
    return a + Tangent<S, N>(detail::constantOf<S>(b));
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> operator+(double a, const Tangent<S, N>& b) {
    return Tangent<S, N>(detail::constantOf<S>(a)) + b;
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> operator-(const Tangent<S, N>& a, double b) {
    return a - Tangent<S, N>(detail::constantOf<S>(b));
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> operator-(double a, const Tangent<S, N>& b) {
    return Tangent<S, N>(detail::constantOf<S>(a)) - b;
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> operator*(const Tangent<S, N>& a, double b) {
    return a * Tangent<S, N>(detail::constantOf<S>(b));
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> operator*(double a, const Tangent<S, N>& b) {
    return Tangent<S, N>(detail::constantOf<S>(a)) * b;
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> operator/(const Tangent<S, N>& a, double b) {
    return a / Tangent<S, N>(detail::constantOf<S>(b));
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> operator/(double a, const Tangent<S, N>& b) {
    return Tangent<S, N>(detail::constantOf<S>(a)) / b;
}

// ── comparisons: primal value decides the branch (pinned) ──

template <typename S, std::size_t N>
bool operator<(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value < b.value;
}
template <typename S, std::size_t N>
bool operator>(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value > b.value;
}
template <typename S, std::size_t N>
bool operator<=(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value <= b.value;
}
template <typename S, std::size_t N>
bool operator>=(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value >= b.value;
}
template <typename S, std::size_t N>
bool operator==(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value == b.value;
}
template <typename S, std::size_t N>
bool operator!=(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value != b.value;
}

// mixed comparisons against a plain scalar: `x < 0.5`, `0.0 <= x` in generic
// code where the right-hand side is an `S` literal (deduction fails for a bare
// double against the homogeneous overloads because no conversion is allowed).
template <typename S, std::size_t N>
bool operator<(const Tangent<S, N>& a, const S& b) {
    return a.value < b;
}
template <typename S, std::size_t N>
bool operator>(const Tangent<S, N>& a, const S& b) {
    return a.value > b;
}
template <typename S, std::size_t N>
bool operator<=(const Tangent<S, N>& a, const S& b) {
    return a.value <= b;
}
template <typename S, std::size_t N>
bool operator>=(const Tangent<S, N>& a, const S& b) {
    return a.value >= b;
}
template <typename S, std::size_t N>
bool operator==(const Tangent<S, N>& a, const S& b) {
    return a.value == b;
}
template <typename S, std::size_t N>
bool operator!=(const Tangent<S, N>& a, const S& b) {
    return a.value != b;
}
template <typename S, std::size_t N>
bool operator<(const S& a, const Tangent<S, N>& b) {
    return a < b.value;
}
template <typename S, std::size_t N>
bool operator>(const S& a, const Tangent<S, N>& b) {
    return a > b.value;
}
template <typename S, std::size_t N>
bool operator<=(const S& a, const Tangent<S, N>& b) {
    return a <= b.value;
}
template <typename S, std::size_t N>
bool operator>=(const S& a, const Tangent<S, N>& b) {
    return a >= b.value;
}
template <typename S, std::size_t N>
bool operator==(const S& a, const Tangent<S, N>& b) {
    return a == b.value;
}
template <typename S, std::size_t N>
bool operator!=(const S& a, const Tangent<S, N>& b) {
    return a != b.value;
}

// mixed comparisons against a plain double (`x < 0.5`): same deduction gap as
// the arithmetic above, with the same `S = double` disable where the
// `const S&` overloads already bind. The constant is built through
// `detail::constantOf` so nested `Tangent` levels compare on their primal.
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
bool operator<(const Tangent<S, N>& a, double b) {
    return a.value < detail::constantOf<S>(b);
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
bool operator>(const Tangent<S, N>& a, double b) {
    return a.value > detail::constantOf<S>(b);
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
bool operator<=(const Tangent<S, N>& a, double b) {
    return a.value <= detail::constantOf<S>(b);
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
bool operator>=(const Tangent<S, N>& a, double b) {
    return a.value >= detail::constantOf<S>(b);
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
bool operator==(const Tangent<S, N>& a, double b) {
    return a.value == detail::constantOf<S>(b);
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
bool operator!=(const Tangent<S, N>& a, double b) {
    return a.value != detail::constantOf<S>(b);
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
bool operator<(double a, const Tangent<S, N>& b) {
    return detail::constantOf<S>(a) < b.value;
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
bool operator>(double a, const Tangent<S, N>& b) {
    return detail::constantOf<S>(a) > b.value;
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
bool operator<=(double a, const Tangent<S, N>& b) {
    return detail::constantOf<S>(a) <= b.value;
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
bool operator>=(double a, const Tangent<S, N>& b) {
    return detail::constantOf<S>(a) >= b.value;
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
bool operator==(double a, const Tangent<S, N>& b) {
    return detail::constantOf<S>(a) == b.value;
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
bool operator!=(double a, const Tangent<S, N>& b) {
    return detail::constantOf<S>(a) != b.value;
}

// ── elementary functions ──

template <typename S, std::size_t N>
Tangent<S, N> sqrt(const Tangent<S, N>& x) {
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
Tangent<S, N> exp(const Tangent<S, N>& x) {
    using std::exp;
    Tangent<S, N> out;
    out.value = exp(x.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * out.value;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> log(const Tangent<S, N>& x) {
    using std::log;
    Tangent<S, N> out;
    out.value = log(x.value);
    const S scale = S(1.0) / x.value;
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

// ── expm1 / log1p ──

template <typename S, std::size_t N>
Tangent<S, N> expm1(const Tangent<S, N>& x) {
    using std::expm1;
    Tangent<S, N> out;
    out.value = expm1(x.value);
    const S scale = out.value + S(1.0); // d/dx expm1(x) = exp(x) = expm1(x) + 1
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> log1p(const Tangent<S, N>& x) {
    using std::log1p;
    Tangent<S, N> out;
    out.value = log1p(x.value);
    const S scale = S(1.0) / (S(1.0) + x.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

// ── cbrt / exp2 / log2 / log10 ──

template <typename S, std::size_t N>
Tangent<S, N> cbrt(const Tangent<S, N>& x) {
    using std::cbrt;
    Tangent<S, N> out;
    out.value = cbrt(x.value);
    const S inv = S(1.0) / out.value;
    const S scale = inv * inv / 3.0; // d/dx x^(1/3) = 1/(3 cbrt(x)^2)
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> exp2(const Tangent<S, N>& x) {
    using std::exp2;
    Tangent<S, N> out;
    out.value = exp2(x.value);
    const S scale = out.value * 0.69314718055994530942; // d/dx 2^x = ln2 2^x
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> log2(const Tangent<S, N>& x) {
    using std::log2;
    Tangent<S, N> out;
    out.value = log2(x.value);
    const S scale = 1.0 / (x.value * 0.69314718055994530942); // 1/(x ln2)
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> log10(const Tangent<S, N>& x) {
    using std::log10;
    Tangent<S, N> out;
    out.value = log10(x.value);
    const S scale = 1.0 / (x.value * 2.30258509299404568402); // 1/(x ln10)
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

// ── powers ──

template <typename S, std::size_t N>
Tangent<S, N> pow(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    using std::log;
    using std::pow;
    Tangent<S, N> out;
    out.value = pow(a.value, b.value);
    const S log_a = log(a.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = out.value * (b.d[i] * log_a + b.value * a.d[i] / a.value);
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> pow(const Tangent<S, N>& a, const S& b) {
    using std::pow;
    Tangent<S, N> out;
    out.value = pow(a.value, b);
    // b * a^(b-1) stays real for negative a and integral b.
    const S scale = b * pow(a.value, b - S(1.0));
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = a.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> pow(const S& a, const Tangent<S, N>& b) {
    using std::log;
    using std::pow;
    Tangent<S, N> out;
    out.value = pow(a, b.value);
    const S scale = out.value * log(a);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = b.d[i] * scale;
    }
    return out;
}

// mixed pow with a plain-double base/exponent; see the mixed-double note above.
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> pow(const Tangent<S, N>& a, double b) {
    return pow(a, Tangent<S, N>(detail::constantOf<S>(b)));
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> pow(double a, const Tangent<S, N>& b) {
    return pow(Tangent<S, N>(detail::constantOf<S>(a)), b);
}

// ── fused multiply-add: composite a*b + c (no dedicated op) ──

template <typename S, std::size_t N>
Tangent<S, N> fma(const Tangent<S, N>& a, const Tangent<S, N>& b, const Tangent<S, N>& c) {
    return a * b + c;
}

// ── trigonometric ──

template <typename S, std::size_t N>
Tangent<S, N> sin(const Tangent<S, N>& x) {
    using std::cos;
    using std::sin;
    Tangent<S, N> out;
    out.value = sin(x.value);
    const S scale = cos(x.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> cos(const Tangent<S, N>& x) {
    using std::cos;
    using std::sin;
    Tangent<S, N> out;
    out.value = cos(x.value);
    const S scale = -sin(x.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

/// sin and cos of `x` in one call: both values and their tangent factors
/// (d sin = cos, d cos = -sin) come from the same primal evaluation. The
/// output references must be distinct (and may alias `x`). `s`/`c` get zero
/// tangents when `x` carries none.
template <typename S, std::size_t N>
void sincos(const Tangent<S, N>& x, Tangent<S, N>& s, Tangent<S, N>& c) {
    using std::cos;
    using std::sin;
    const S value = x.value;
    s.value = sin(value);
    c.value = cos(value);
    const S ds = c.value;
    const S dc = -s.value;
    for (std::size_t i = 0; i < N; ++i) {
        const S d = x.d[i];
        s.d[i] = d * ds;
        c.d[i] = d * dc;
    }
}

template <typename S, std::size_t N>
Tangent<S, N> tan(const Tangent<S, N>& x) {
    using std::tan;
    Tangent<S, N> out;
    out.value = tan(x.value);
    const S scale = S(1.0) + out.value * out.value; // 1 + tan^2 = sec^2
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

// ── hyperbolic ──

template <typename S, std::size_t N>
Tangent<S, N> sinh(const Tangent<S, N>& x) {
    using std::cosh;
    using std::sinh;
    Tangent<S, N> out;
    out.value = sinh(x.value);
    const S scale = cosh(x.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> cosh(const Tangent<S, N>& x) {
    using std::cosh;
    using std::sinh;
    Tangent<S, N> out;
    out.value = cosh(x.value);
    const S scale = sinh(x.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> tanh(const Tangent<S, N>& x) {
    using std::tanh;
    Tangent<S, N> out;
    out.value = tanh(x.value);
    const S scale = S(1.0) - out.value * out.value; // sech^2 = 1 - tanh^2
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

// ── inverse hyperbolic (factor from the primal) ──

template <typename S, std::size_t N>
Tangent<S, N> asinh(const Tangent<S, N>& x) {
    using std::asinh;
    using std::sqrt;
    Tangent<S, N> out;
    out.value = asinh(x.value);
    const S scale = S(1.0) / sqrt(S(1.0) + x.value * x.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> acosh(const Tangent<S, N>& x) {
    using std::acosh;
    using std::sqrt;
    Tangent<S, N> out;
    out.value = acosh(x.value);
    const S scale = S(1.0) / sqrt(x.value * x.value - S(1.0));
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> atanh(const Tangent<S, N>& x) {
    using std::atanh;
    Tangent<S, N> out;
    out.value = atanh(x.value);
    const S scale = S(1.0) / (S(1.0) - x.value * x.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

// ── inverse trigonometric ──

template <typename S, std::size_t N>
Tangent<S, N> asin(const Tangent<S, N>& x) {
    using std::asin;
    using std::sqrt;
    Tangent<S, N> out;
    out.value = asin(x.value);
    const S scale = S(1.0) / sqrt(S(1.0) - x.value * x.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> acos(const Tangent<S, N>& x) {
    using std::acos;
    using std::sqrt;
    Tangent<S, N> out;
    out.value = acos(x.value);
    const S scale = S(-1.0) / sqrt(S(1.0) - x.value * x.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> atan(const Tangent<S, N>& x) {
    using std::atan;
    Tangent<S, N> out;
    out.value = atan(x.value);
    const S scale = S(1.0) / (S(1.0) + x.value * x.value);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

// ── planar norm / angle ──

template <typename S, std::size_t N>
Tangent<S, N> hypot(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    using std::hypot;
    Tangent<S, N> out;
    out.value = hypot(a.value, b.value);
    const S h = out.value;
    if (h != S(0.0)) {
        const S inv = S(1.0) / h;
        for (std::size_t i = 0; i < N; ++i) {
            out.d[i] = (a.d[i] * a.value + b.d[i] * b.value) * inv;
        }
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> hypot(const Tangent<S, N>& a, const S& b) {
    return hypot(a, Tangent<S, N>(b));
}

template <typename S, std::size_t N>
Tangent<S, N> hypot(const S& a, const Tangent<S, N>& b) {
    return hypot(Tangent<S, N>(a), b);
}

// mixed hypot with a plain double; see the mixed-double note above.
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> hypot(const Tangent<S, N>& a, double b) {
    return hypot(a, Tangent<S, N>(detail::constantOf<S>(b)));
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> hypot(double a, const Tangent<S, N>& b) {
    return hypot(Tangent<S, N>(detail::constantOf<S>(a)), b);
}

template <typename S, std::size_t N>
Tangent<S, N> atan2(const Tangent<S, N>& y, const Tangent<S, N>& x) {
    using std::atan2;
    Tangent<S, N> out;
    out.value = atan2(y.value, x.value);
    const S denom = y.value * y.value + x.value * x.value;
    if (denom > S(0.0)) {
        const S inv = S(1.0) / denom;
        for (std::size_t i = 0; i < N; ++i) {
            out.d[i] = (y.d[i] * x.value - x.d[i] * y.value) * inv;
        }
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> atan2(const Tangent<S, N>& y, const S& x) {
    return atan2(y, Tangent<S, N>(x));
}

template <typename S, std::size_t N>
Tangent<S, N> atan2(const S& y, const Tangent<S, N>& x) {
    return atan2(Tangent<S, N>(y), x);
}

// mixed atan2 with a plain double; see the mixed-double note above.
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> atan2(const Tangent<S, N>& y, double x) {
    return atan2(y, Tangent<S, N>(detail::constantOf<S>(x)));
}
template <typename S, std::size_t N>
    requires(!std::is_same_v<S, double>)
Tangent<S, N> atan2(double y, const Tangent<S, N>& x) {
    return atan2(Tangent<S, N>(detail::constantOf<S>(y)), x);
}

// ── error functions / normal CDF ──

template <typename S, std::size_t N>
Tangent<S, N> erf(const Tangent<S, N>& x) {
    using std::erf;
    using std::exp;
    Tangent<S, N> out;
    out.value = erf(x.value);
    const S scale = S(1.12837916709551257390) * exp(-x.value * x.value); // 2 / sqrt(pi)
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> erfc(const Tangent<S, N>& x) {
    using std::erfc;
    using std::exp;
    Tangent<S, N> out;
    out.value = erfc(x.value);
    const S scale = S(-1.12837916709551257390) * exp(-x.value * x.value); // -2 / sqrt(pi)
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> fabs(const Tangent<S, N>& x) {
    using std::fabs;
    Tangent<S, N> out;
    out.value = fabs(x.value);
    const S sign = x.value >= S(0.0) ? S(1.0) : S(-1.0);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * sign;
    }
    return out;
}

/// copysign(a, b): magnitude of `a` with the sign of `b`. The derivative of
/// `a` is scaled by sign(b.value) (primal-pinned; `b.value >= 0` counts as +).
template <typename S, std::size_t N>
Tangent<S, N> copysign(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    using std::copysign;
    Tangent<S, N> out;
    out.value = copysign(a.value, b.value);
    const S signB = b.value >= S(0.0) ? S(1.0) : S(-1.0);
    const S signA = a.value >= S(0.0) ? S(1.0) : S(-1.0);
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = a.d[i] * signB * signA;
    }
    return out;
}

// mixed copysign with a plain double (no `const S&` counterpart, so enabled
// for every `S`; see the mixed-double note above).
template <typename S, std::size_t N>
Tangent<S, N> copysign(const Tangent<S, N>& a, double b) {
    return copysign(a, Tangent<S, N>(detail::constantOf<S>(b)));
}
template <typename S, std::size_t N>
Tangent<S, N> copysign(double a, const Tangent<S, N>& b) {
    return copysign(Tangent<S, N>(detail::constantOf<S>(a)), b);
}

template <typename S, std::size_t N>
Tangent<S, N> fmin(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value <= b.value ? a : b;
}

template <typename S, std::size_t N>
Tangent<S, N> fmax(const Tangent<S, N>& a, const Tangent<S, N>& b) {
    return a.value >= b.value ? a : b;
}

// mixed fmin/fmax with a plain double (no `const S&` counterpart, so enabled
// for every `S`; see the mixed-double note above).
template <typename S, std::size_t N>
Tangent<S, N> fmin(const Tangent<S, N>& a, double b) {
    return fmin(a, Tangent<S, N>(detail::constantOf<S>(b)));
}
template <typename S, std::size_t N>
Tangent<S, N> fmin(double a, const Tangent<S, N>& b) {
    return fmin(Tangent<S, N>(detail::constantOf<S>(a)), b);
}
template <typename S, std::size_t N>
Tangent<S, N> fmax(const Tangent<S, N>& a, double b) {
    return fmax(a, Tangent<S, N>(detail::constantOf<S>(b)));
}
template <typename S, std::size_t N>
Tangent<S, N> fmax(double a, const Tangent<S, N>& b) {
    return fmax(Tangent<S, N>(detail::constantOf<S>(a)), b);
}

// std::abs compatibility
template <typename S, std::size_t N>
Tangent<S, N> abs(const Tangent<S, N>& x) {
    return fabs(x);
}

// ── primal-based predicates ──

template <typename S, std::size_t N>
bool isnan(const Tangent<S, N>& x) {
    using std::isnan;
    return isnan(x.value);
}

template <typename S, std::size_t N>
bool isinf(const Tangent<S, N>& x) {
    using std::isinf;
    return isinf(x.value);
}

template <typename S, std::size_t N>
bool isfinite(const Tangent<S, N>& x) {
    using std::isfinite;
    return isfinite(x.value);
}

/// Predicate only (no tangents): the sign bit of the primal, so `-0.0` is
/// negative.
template <typename S, std::size_t N>
bool signbit(const Tangent<S, N>& x) {
    using std::signbit;
    return signbit(x.value);
}

// ── gamma family: digamma -> trigamma -> tetragamma coefficient chain ──
//
// lgamma/tgamma are generic over S; the first-order coefficient is
// `digamma(x.value)`. Each nested Tangent level differentiates the
// coefficient through the next polygamma order (`digamma(Tangent)` scales by
// `trigamma(x.value)`, `trigamma(Tangent)` by `tetragamma(x.value)`,
// `tetragamma(Tangent)` by `pentagamma(x.value)`), so fwd-over-fwd higher
// derivatives compile while the chain reaches a base scalar with that order
// available. Base scalars: `double` (orders 0..4, Digamma.h) and `RevScalar`
// (tape ops for orders 0..2, LeanReverse.h), so the gamma family now also
// instantiates for `Tangent<RevScalar, N>` (fwd-over-rev mode mixing); a
// Tangent `tetragamma` level over `RevScalar` would need a pentagamma op and
// is not provided. Nesting depth of `trigamma` is one level beyond what its
// base supports (`tetragamma(Tangent)` only needs `pentagamma(double)`).

template <typename S, std::size_t N>
Tangent<S, N> tetragamma(const Tangent<S, N>& x) {
    Tangent<S, N> out;
    out.value = tetragamma(x.value);
    const S scale = pentagamma(x.value); // d/dx tetragamma = pentagamma
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> trigamma(const Tangent<S, N>& x) {
    Tangent<S, N> out;
    out.value = trigamma(x.value);
    const S scale = tetragamma(x.value); // d/dx trigamma = tetragamma
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> digamma(const Tangent<S, N>& x) {
    Tangent<S, N> out;
    out.value = digamma(x.value);
    const S scale = trigamma(x.value); // d/dx digamma = trigamma
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> lgamma(const Tangent<S, N>& x) {
    using std::lgamma;
    Tangent<S, N> out;
    out.value = lgamma(x.value);
    const S scale = digamma(x.value); // d/dx lgamma = digamma
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

template <typename S, std::size_t N>
Tangent<S, N> tgamma(const Tangent<S, N>& x) {
    using std::tgamma;
    Tangent<S, N> out;
    out.value = tgamma(x.value);
    const S scale = out.value * digamma(x.value); // d/dx tgamma = tgamma * digamma
    for (std::size_t i = 0; i < N; ++i) {
        out.d[i] = x.d[i] * scale;
    }
    return out;
}

} // namespace lightape

#endif // LIGHTAPE_FWD_FORWARD_SCALAR_H
