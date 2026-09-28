#ifndef QUANTAPE_MATH_PRECISION_DOUBLE_DOUBLE_H
#define QUANTAPE_MATH_PRECISION_DOUBLE_DOUBLE_H

//
// DoubleDouble.h -- in-house double-double (compensated) arithmetic
//
// Two-double representation `hi + lo` giving ~106 bits (~31-32 decimal
// digits) of precision. The implementation follows the classic
// error-free-transformation approach (Dekker/Knuth, as in Bailey's DDFUN and
// the QD library): `two_sum`/`quick_two_sum` for additions, FMA-based
// `two_prod` for products, and a three-term Newton division. Elementary
// functions are reduced to the first quadrant / [1, 2) and evaluated with
// series whose small argument is the second component.
//
// Used for: in-house high-precision certification of the special functions
// the Heston pricer depends on (Si/Ci anchors), high-precision reference
// integrals in tests, and the planned offline EFGL table generator.
//
// IMPORTANT: compensated arithmetic requires strict IEEE semantics. The
// translation unit using this header must NOT be compiled with
// `-ffast-math`/`-funsafe-math-optimizations` (reassociation folds the
// error terms away); add `-fno-fast-math` where the project's release flags
// enable them.
//

#include <cmath>
#include <cstdint>

namespace quantape::math {

class DoubleDouble {
public:
    double hi = 0.0;
    double lo = 0.0;

    constexpr DoubleDouble() = default;
    constexpr DoubleDouble(double value)
        : hi(value), lo(0.0) {} // NOLINT(google-explicit-constructor)
    constexpr DoubleDouble(double high, double low) : hi(high), lo(low) {}

    double value() const { return hi + lo; }

    friend DoubleDouble operator+(const DoubleDouble& a, const DoubleDouble& b) {
        const double s = a.hi + b.hi;
        const double v = s - a.hi;
        double e = (a.hi - (s - v)) + (b.hi - v);
        e += a.lo + b.lo;
        const double h = s + e;
        return DoubleDouble(h, e - (h - s));
    }

    friend DoubleDouble operator-(const DoubleDouble& a, const DoubleDouble& b) {
        return a + DoubleDouble(-b.hi, -b.lo);
    }

    friend DoubleDouble operator*(const DoubleDouble& a, const DoubleDouble& b) {
        const double p = a.hi * b.hi;
        const double e = std::fma(a.hi, b.hi, -p);
        double lo = e + a.hi * b.lo + a.lo * b.hi;
        lo += a.lo * b.lo;
        const double h = p + lo;
        return DoubleDouble(h, lo - (h - p));
    }

    friend DoubleDouble operator/(const DoubleDouble& a, const DoubleDouble& b) {
        const double q1 = a.hi / b.hi;
        DoubleDouble r = a - DoubleDouble(q1) * b;
        const double q2 = r.hi / b.hi;
        r = r - DoubleDouble(q2) * b;
        const double q3 = r.hi / b.hi;
        const double h1 = q1 + q2;
        return DoubleDouble(h1, (q1 - h1) + q2 + q3);
    }

    DoubleDouble operator-() const { return DoubleDouble(-hi, -lo); }

    DoubleDouble& operator+=(const DoubleDouble& o) { return *this = *this + o; }
    DoubleDouble& operator-=(const DoubleDouble& o) { return *this = *this - o; }
    DoubleDouble& operator*=(const DoubleDouble& o) { return *this = *this * o; }
    DoubleDouble& operator/=(const DoubleDouble& o) { return *this = *this / o; }

    friend bool operator<(const DoubleDouble& a, const DoubleDouble& b) {
        return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
    }
    friend bool operator>(const DoubleDouble& a, const DoubleDouble& b) { return b < a; }
    friend bool operator<=(const DoubleDouble& a, const DoubleDouble& b) { return !(b < a); }
    friend bool operator>=(const DoubleDouble& a, const DoubleDouble& b) { return !(a < b); }
    friend bool operator==(const DoubleDouble& a, const DoubleDouble& b) {
        return a.hi == b.hi && a.lo == b.lo;
    }
};

inline DoubleDouble sqrt(const DoubleDouble& a) {
    if (a.hi <= 0.0) {
        return DoubleDouble(std::sqrt(a.hi));
    }
    DoubleDouble r(std::sqrt(a.hi));
    // Two Newton steps in double-double: the first refines ~1e-16 -> ~1e-32.
    r = (r + a / r) * DoubleDouble(0.5);
    r = (r + a / r) * DoubleDouble(0.5);
    return r;
}

namespace detail {
/// pi/2 in double-double (Dekker constants).
inline const DoubleDouble& piOverTwoDD() {
    static const DoubleDouble v(1.5707963267948966, 6.123233995736766e-17);
    return v;
}
inline const DoubleDouble& piDD() {
    static const DoubleDouble v(3.141592653589793, 1.2246467991473532e-16);
    return v;
}
inline const DoubleDouble& log2DD() {
    static const DoubleDouble v(0.6931471805599453, 2.3190468138462996e-17);
    return v;
}
} // namespace detail

inline DoubleDouble exp(const DoubleDouble& a) {
    // a = k ln2 + r, |r| <= ln2/2; exp(r) by Taylor in double-double,
    // then exact scaling by 2^k (both components ldexp'ed).
    const DoubleDouble kf(std::nearbyint(a.hi / detail::log2DD().hi));
    const int k = static_cast<int>(kf.hi);
    const DoubleDouble r = a - detail::log2DD() * kf;
    DoubleDouble term(1.0);
    DoubleDouble sum(1.0);
    for (int n = 1; n < 40; ++n) {
        term = term * r / DoubleDouble(static_cast<double>(n));
        sum = sum + term;
        if (term.hi == 0.0 && term.lo == 0.0) {
            break;
        }
    }
    return DoubleDouble(std::ldexp(sum.hi, k), std::ldexp(sum.lo, k));
}

inline DoubleDouble log(const DoubleDouble& a) {
    if (!(a.hi > 0.0)) {
        return DoubleDouble(std::log(a.hi));
    }
    int exponent = 0;
    const double m = std::frexp(a.hi, &exponent); // a.hi = m * 2^exponent, m in [0.5, 1)
    const DoubleDouble scaledLo = DoubleDouble(a.lo) / DoubleDouble(std::ldexp(1.0, exponent));
    const DoubleDouble u = DoubleDouble(m) + scaledLo; // in [0.5, 1)
    const DoubleDouble t = (u - DoubleDouble(1.0)) / (u + DoubleDouble(1.0));
    // log(u) = 2 (t + t^3/3 + t^5/5 + ...), |t| <= 1/3
    const DoubleDouble t2 = t * t;
    DoubleDouble term = t;
    DoubleDouble sum = term;
    for (int kk = 3; kk < 80; kk += 2) {
        term = term * t2;
        const DoubleDouble add = term / DoubleDouble(static_cast<double>(kk));
        sum = sum + add;
        if (add.hi == 0.0 && add.lo == 0.0) {
            break;
        }
    }
    return detail::log2DD() * DoubleDouble(static_cast<double>(exponent)) + DoubleDouble(2.0) * sum;
}

namespace detail {

/// sin/cos of |r| <= pi/4 with double-double Taylor series.
inline void sincosSmall(const DoubleDouble& r, DoubleDouble& s, DoubleDouble& c) {
    const DoubleDouble r2 = r * r;
    DoubleDouble term = r;
    DoubleDouble sinSum = r;
    for (int n = 1; n < 30; ++n) {
        term =
            -term * r2 / DoubleDouble(static_cast<double>(2 * n) * static_cast<double>(2 * n + 1));
        sinSum = sinSum + term;
        if (term.hi == 0.0 && term.lo == 0.0) {
            break;
        }
    }
    term = DoubleDouble(1.0);
    DoubleDouble cosSum(1.0);
    for (int n = 1; n < 30; ++n) {
        term =
            -term * r2 / DoubleDouble(static_cast<double>(2 * n - 1) * static_cast<double>(2 * n));
        cosSum = cosSum + term;
        if (term.hi == 0.0 && term.lo == 0.0) {
            break;
        }
    }
    s = sinSum;
    c = cosSum;
}

} // namespace detail

inline DoubleDouble sin(const DoubleDouble& a) {
    // Reduce by pi/2 in double-double, Taylor on |r| <= pi/4, quadrant map.
    const DoubleDouble halfPi = detail::piOverTwoDD();
    const DoubleDouble q(std::nearbyint(a.hi / halfPi.hi));
    const DoubleDouble r = a - halfPi * q;
    DoubleDouble s, c;
    detail::sincosSmall(r, s, c);
    switch (static_cast<long>(q.hi) & 3) {
        case 0:
            return s;
        case 1:
            return c;
        case 2:
            return -s;
        default:
            return -c;
    }
}

inline DoubleDouble cos(const DoubleDouble& a) {
    const DoubleDouble halfPi = detail::piOverTwoDD();
    const DoubleDouble q(std::nearbyint(a.hi / halfPi.hi));
    const DoubleDouble r = a - halfPi * q;
    DoubleDouble s, c;
    detail::sincosSmall(r, s, c);
    switch (static_cast<long>(q.hi) & 3) {
        case 0:
            return c;
        case 1:
            return -s;
        case 2:
            return -c;
        default:
            return s;
    }
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_PRECISION_DOUBLE_DOUBLE_H
