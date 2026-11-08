#ifndef QUANTAPE_MATH_SPECIAL_FUNCTIONS_TRIG_INTEGRALS_H
#define QUANTAPE_MATH_SPECIAL_FUNCTIONS_TRIG_INTEGRALS_H

#include "quantape/math/Precision/DoubleDouble.h"
#include "quantape/util/Constants.h"

#include <cmath>
#include <complex>
#include <stdexcept>

namespace quantape::math {
using ::quantape::util::kHalfPi;
using ::quantape::util::kPi;
/**
 * @file TrigIntegrals.h
 * @brief Trigonometric integrals Si, Ci (real and complex), in-house
 *
 *   Si(z) = int_0^z sin(t)/t dt,      Ci(z) = gamma + ln z + int_0^z (cos t - 1)/t dt
 *
 * Needed by the Heston asymptotic control variate
 * (`HestonControlVariate::Asymptotic`): the closed-form integral of the
 * asymptotic characteristic-function expansion is
 *
 *   int_0^inf e^{phi u + psi}/(u^2 + 1/4) du
 *     = e^psi ( -2 Ci(-phi/2) sin(phi/2) + cos(phi/2)(pi + 2 Si(phi/2)) )
 *
 * with complex `phi`. The Heston pricer evaluates it at `phi/2 + i mu/2`
 * with `mu = ln(S/K) + (r-q)T`, so `|z|` is *not* bounded by the CV
 * parameter size: a deep out-of-the-money strike (K = 1e-8, mu ~ 18.4)
 * already needs |z| ~ 9.2, and any K > 0 is valid input.
 *
 * Real branch accuracy: the published rational approximations from the
 * GALSIM toolkit (Rowe et al., 2014, arXiv:1407.7676; the same coefficient
 * sets used by QuantLib's `exponentialintegrals.cpp`, BSD-licensed) give
 * ~1e-16 relative accuracy for all x >= 0, with the power series used below
 * x = 4. Complex branch (all |arg z| < pi, principal branch):
 *   - |z| <= 8: plain-double power series (terms peak at e^8, leaving
 *     ~1e-13 relative from the cancellation);
 *   - 8 < |z| <= 40: the same series in compensated (double-double)
 *     arithmetic, which keeps the full double-double resolution against the
 *     cancellation of O(e^{|z|}) terms down to an O(1) result;
 *   - |z| > 40: the asymptotic expansion in the sector |arg z| <= pi/2
 *     (arguments outside it are reflected onto Re z >= 0),
 *
 *       Si(z) = pi/2 - f(z) cos z - g(z) sin z,
 *       Ci(z) = f(z) sin z - g(z) cos z,
 *       f(z) ~ sum_{k>=0} (-1)^k (2k)!/z^{2k+1},
 *       g(z) ~ sum_{k>=0} (-1)^k (2k+1)!/z^{2k+2},
 *
 *     summed to its smallest term (optimal truncation; the series diverges
 *     beyond it). The truncation error there is bounded by the first omitted
 *     term, and against the defining series the complex relative error is
 *     below 1e-15 for |z| > 40. The pure-imaginary axis z = i b uses the
 *     non-alternating hyperbolic series
 *     Si(i b) = i Shi(b), Ci(i b) = Chi(|b|) + i (pi/2) sign(b) instead,
 *     because the asymptotic remainder on that Stokes line is O(1) absolute
 *     (bounded by e^{-|b|} e^{|b|}) and the hyperbolic series keeps both
 *     components to double precision. The regimes are continuous at the
 *     switches (checked by the unit tests).
 *
 * Derivatives are structural (`Si' = sin(z)/z`, `Ci' = cos(z)/z`), so a
 * future AD instantiation only needs those two overloads.
 */

namespace detail {

constexpr double kEulerGamma = 0.577215664901532860606512090082402431;

/// GALSIM auxiliary functions (real, x > 0); g already carries the 1/x^2.
inline double galsimF(double x) {
    const double x2 = 1.0 / (x * x);
    return (1.0 +
            x2 *
                (7.44437068161936700618e2 +
                 x2 *
                     (1.96396372895146869801e5 +
                      x2 *
                          (2.37750310125431834034e7 +
                           x2 *
                               (1.43073403821274636888e9 +
                                x2 *
                                    (4.33736238870432522765e10 +
                                     x2 *
                                         (6.40533830574022022911e11 +
                                          x2 *
                                              (4.20968180571076940208e12 +
                                               x2 *
                                                   (1.00795182980368574617e13 +
                                                    x2 * (4.94816688199951963482e12 -
                                                          x2 * 4.94701168645415959931e11)))))))))) /
           (x *
            (1.0 +
             x2 *
                 (7.46437068161927678031e2 +
                  x2 *
                      (1.97865247031583951450e5 +
                       x2 * (2.41535670165126845144e7 +
                             x2 * (1.47478952192985464958e9 +
                                   x2 * (4.58595115847765779830e10 +
                                         x2 * (7.08501308149515401563e11 +
                                               x2 * (5.06084464593475076774e12 +
                                                     x2 * (1.43468549171581016479e13 +
                                                           x2 * 1.11535493509914254097e13))))))))));
}

inline double galsimG(double x) {
    const double x2 = 1.0 / (x * x);
    return x2 *
           (1.0 +
            x2 *
                (8.1359520115168615e2 +
                 x2 *
                     (2.35239181626478200e5 +
                      x2 *
                          (3.12557570795778731e7 +
                           x2 *
                               (2.06297595146763354e9 +
                                x2 *
                                    (6.83052205423625007e10 +
                                     x2 * (1.09049528450362786e12 +
                                           x2 * (7.57664583257834349e12 +
                                                 x2 * (1.81004487464664575e13 +
                                                       x2 * (6.43291613143049485e12 -
                                                             x2 * 1.36517137670871689e12)))))))))) /
           (1.0 +
            x2 * (8.19595201151451564e2 +
                  x2 * (2.40036752835578777e5 +
                        x2 * (3.26026661647090822e7 +
                              x2 * (2.23355543278099360e9 +
                                    x2 * (7.87465017341829930e10 +
                                          x2 * (1.39866710696414565e12 +
                                                x2 * (1.17164723371736605e13 +
                                                      x2 * (4.01839087307656620e13 +
                                                            x2 * 3.99653257887490811e13)))))))));
}

/// Power series (used for real x <= 4 and complex |z| <= 8).
template <typename T>
T siSeries(const T& z) {
    T sum = z;
    T term = z;
    T z2 = z * z;
    for (int k = 1; k < 200; ++k) {
        const double kk = static_cast<double>(k);
        term *= -z2 / ((2.0 * kk) * (2.0 * kk + 1.0));
        const T add = term / (2.0 * kk + 1.0);
        sum += add;
        if (std::abs(add) <= 1e-18 * std::abs(sum)) {
            break;
        }
    }
    return sum;
}

template <typename T>
T ciSeries(const T& z) {
    T sum = T(0.0);
    T term = T(1.0);
    T z2 = z * z;
    for (int k = 1; k < 200; ++k) {
        const double kk = static_cast<double>(k);
        term *= -z2 / ((2.0 * kk - 1.0) * (2.0 * kk));
        const T add = term / (2.0 * kk);
        sum += add;
        if (std::abs(add) <= 1e-18 * std::abs(sum)) {
            break;
        }
    }
    using std::log;
    return kEulerGamma + log(z) + sum;
}

/// Complex number with double-double components. Needed only by the power
/// series on 8 < |z| <= kComplexSeriesLimit, where the plain-double version
/// cancels O(e^{|z|}) terms down to an O(1) result.
struct ComplexDD {
    DoubleDouble re;
    DoubleDouble im;
};

inline ComplexDD toComplexDD(const std::complex<double>& z) {
    return ComplexDD{DoubleDouble(z.real()), DoubleDouble(z.imag())};
}

inline std::complex<double> toStdComplex(const ComplexDD& z) {
    return std::complex<double>(z.re.value(), z.im.value());
}

inline ComplexDD operator+(const ComplexDD& a, const ComplexDD& b) {
    return ComplexDD{a.re + b.re, a.im + b.im};
}

inline ComplexDD operator-(const ComplexDD& a, const ComplexDD& b) {
    return ComplexDD{a.re - b.re, a.im - b.im};
}

inline ComplexDD operator-(const ComplexDD& a) {
    return ComplexDD{-a.re, -a.im};
}

inline ComplexDD operator*(const ComplexDD& a, const ComplexDD& b) {
    return ComplexDD{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

inline ComplexDD operator/(const ComplexDD& a, const ComplexDD& b) {
    const DoubleDouble den = b.re * b.re + b.im * b.im;
    return ComplexDD{(a.re * b.re + a.im * b.im) / den, (a.im * b.re - a.re * b.im) / den};
}

inline DoubleDouble abs2(const ComplexDD& a) {
    return a.re * a.re + a.im * a.im;
}

inline ComplexDD realDD(double v) {
    return ComplexDD{DoubleDouble(v), DoubleDouble(0.0)};
}

/// Terms below 1e-32 of the partial sum are past the double-double resolution.
constexpr double kSeriesTermTol2 = 1e-64;

/// Si power series in compensated arithmetic (see siSeries).
inline ComplexDD siSeriesDD(const ComplexDD& z) {
    const ComplexDD z2 = z * z;
    ComplexDD term = z;
    ComplexDD sum = z;
    for (int k = 1; k < 400; ++k) {
        const double kk = static_cast<double>(k);
        term = -(term * z2) / realDD((2.0 * kk) * (2.0 * kk + 1.0));
        const ComplexDD add = term / realDD(2.0 * kk + 1.0);
        sum = sum + add;
        if (abs2(add) < abs2(sum) * DoubleDouble(kSeriesTermTol2)) {
            break;
        }
    }
    return sum;
}

/// Ci power series in compensated arithmetic (see ciSeries). The principal
/// log is added as log|z| = log(|z|^2)/2 in double-double plus the double
/// atan2 phase, so the real part keeps full compensation.
inline ComplexDD ciSeriesDD(const ComplexDD& z) {
    const ComplexDD z2 = z * z;
    ComplexDD b = realDD(1.0);
    ComplexDD sum = realDD(0.0);
    for (int k = 1; k < 400; ++k) {
        const double kk = static_cast<double>(k);
        b = -(b * z2) / realDD((2.0 * kk - 1.0) * (2.0 * kk));
        const ComplexDD add = b / realDD(2.0 * kk);
        sum = sum + add;
        if (abs2(add) < abs2(sum) * DoubleDouble(kSeriesTermTol2)) {
            break;
        }
    }
    const DoubleDouble logAbs = DoubleDouble(0.5) * log(abs2(z));
    const double arg = std::atan2(z.im.value(), z.re.value());
    return ComplexDD{sum.re + DoubleDouble(kEulerGamma) + logAbs, sum.im + DoubleDouble(arg)};
}

/// Power series is used in double-double up to this radius; beyond it the
/// asymptotic expansion is both faster and (once optimally truncated) more
/// accurate than the compensated series' cancellation floor.
constexpr double kComplexSeriesLimit = 40.0;

/// Si(i b) = i Shi(b) with Shi(b) = sum b^(2k+1) / ((2k+1)! (2k+1)).
/// Every term has the sign of b, so there is no cancellation even for the
/// large b where the asymptotic expansion below has an O(1) Stokes remainder
/// on the imaginary axis (the truncation error is bounded by the smallest
/// term e^{-|b|}, but it is multiplied by |cos b| ~ e^{|b|}).
inline std::complex<double> siImaginaryAxis(double b) {
    double term = b;
    double sum = b;
    for (int k = 1; k < 4000; ++k) {
        const double kk = static_cast<double>(k);
        term *= (b * b) / ((2.0 * kk) * (2.0 * kk + 1.0));
        const double add = term / (2.0 * kk + 1.0);
        sum += add;
        if (std::fabs(add) <= 1e-18 * std::fabs(sum)) {
            break;
        }
    }
    return std::complex<double>(0.0, sum);
}

/// Ci(i b) = Chi(|b|) + i (pi/2) sign(b) with
/// Chi(b) = gamma + ln b + sum b^(2k) / ((2k)! (2k)); same non-alternating
/// series as siImaginaryAxis (all terms positive for b > 0).
inline std::complex<double> ciImaginaryAxis(double b) {
    const double absB = std::fabs(b);
    double term = 1.0;
    double sum = kEulerGamma + std::log(absB);
    for (int k = 1; k < 4000; ++k) {
        const double kk = static_cast<double>(k);
        term *= (absB * absB) / ((2.0 * kk - 1.0) * (2.0 * kk));
        const double add = term / (2.0 * kk);
        sum += add;
        if (std::fabs(add) <= 1e-18 * std::fabs(sum)) {
            break;
        }
    }
    return std::complex<double>(sum, std::copysign(kHalfPi.hi, b));
}

/// f, g from the asymptotic expansion (see the file comment), each summed to
/// its smallest term: the terms' magnitudes decrease while 2k < |z|, then the
/// (divergent) series is cut at the first term that no longer decreases.
inline void asymptoticFG(const std::complex<double>& z, std::complex<double>& f,
                         std::complex<double>& g) {
    using std::abs;
    const std::complex<double> invZ2 = 1.0 / (z * z);
    std::complex<double> fTerm = 1.0 / z;
    std::complex<double> gTerm = invZ2; // 1 / z^2
    std::complex<double> fSum = fTerm;
    std::complex<double> gSum = gTerm;
    double fLast = abs(fTerm);
    double gLast = abs(gTerm);
    bool fDone = false;
    bool gDone = false;
    for (int k = 1; k < 4000 && !(fDone && gDone); ++k) {
        const double kk = static_cast<double>(k);
        fTerm *= -((2.0 * kk - 1.0) * (2.0 * kk)) * invZ2;
        gTerm *= -((2.0 * kk) * (2.0 * kk + 1.0)) * invZ2;
        const double fAbs = abs(fTerm);
        const double gAbs = abs(gTerm);
        if (fAbs < fLast) {
            fSum += fTerm;
            fLast = fAbs;
        } else {
            fDone = true;
        }
        if (gAbs < gLast) {
            gSum += gTerm;
            gLast = gAbs;
        } else {
            gDone = true;
        }
    }
    f = fSum;
    g = gSum;
}

/// Si(z) = pi/2 - f cos z - g sin z for |arg z| <= pi/2; the caller reflects
/// the remaining arguments onto Re z >= 0.
inline std::complex<double> siAsymptotic(const std::complex<double>& z) {
    using std::cos;
    using std::sin;
    std::complex<double> f;
    std::complex<double> g;
    asymptoticFG(z, f, g);
    return std::complex<double>(kHalfPi.hi, 0.0) - f * cos(z) - g * sin(z);
}

/// Ci(z) = f sin z - g cos z for |arg z| <= pi/2.
inline std::complex<double> ciAsymptotic(const std::complex<double>& z) {
    using std::cos;
    using std::sin;
    std::complex<double> f;
    std::complex<double> g;
    asymptoticFG(z, f, g);
    return f * sin(z) - g * cos(z);
}

} // namespace detail

/// Sine integral, real argument (~1e-16 for all x >= 0).
inline double sinIntegral(double x) {
    using std::cos;
    using std::sin;
    if (x < 0.0) {
        return -sinIntegral(-x);
    }
    if (x <= 4.0) {
        const double x2 = x * x;
        return x *
               (1.0 + x2 * (-4.54393409816329991e-2 +
                            x2 * (1.15457225751016682e-3 +
                                  x2 * (-1.41018536821330254e-5 +
                                        x2 * (9.43280809438713025e-8 +
                                              x2 * (-3.53201978997168357e-10 +
                                                    x2 * (7.08240282274875911e-13 -
                                                          x2 * 6.05338212010422477e-16))))))) /
               (1.0 + x2 * (1.01162145739225565e-2 +
                            x2 * (4.99175116169755106e-5 +
                                  x2 * (1.55654986308745614e-7 +
                                        x2 * (3.28067571055789734e-10 +
                                              x2 * (4.5049097575386581e-13 +
                                                    x2 * 3.21107051193712168e-16))))));
    }
    return kHalfPi - detail::galsimF(x) * cos(x) - detail::galsimG(x) * sin(x);
}

/// Cosine integral, real argument (principal branch, x > 0).
inline double cosIntegral(double x) {
    using std::cos;
    using std::log;
    using std::sin;
    if (x <= 0.0) {
        throw std::invalid_argument("cosIntegral: principal branch requires x > 0");
    }
    if (x <= 4.0) {
        const double x2 = x * x;
        return detail::kEulerGamma + log(x) +
               x2 *
                   (-0.25 + x2 * (7.51851524438898291e-3 +
                                  x2 * (-1.27528342240267686e-4 +
                                        x2 * (1.05297363846239184e-6 +
                                              x2 * (-4.68889508144848019e-9 +
                                                    x2 * (1.06480802891189243e-11 -
                                                          x2 * 9.93728488857585407e-15)))))) /
                   (1.0 + x2 * (1.1592605689110735e-2 +
                                x2 * (6.72126800814254432e-5 +
                                      x2 * (2.55533277086129636e-7 +
                                            x2 * (6.97071295760958946e-10 +
                                                  x2 * (1.38536352772778619e-12 +
                                                        x2 * (1.89106054713059759e-15 +
                                                              x2 * 1.39759616731376855e-18)))))));
    }
    return detail::galsimF(x) * sin(x) - detail::galsimG(x) * cos(x);
}

/// Sine integral, complex argument (principal branch; any |z| >= 0).
inline std::complex<double> sinIntegral(const std::complex<double>& z) {
    const double az = std::abs(z);
    if (az <= 8.0) {
        return detail::siSeries(z);
    }
    if (z.real() < 0.0) {
        return -sinIntegral(-z); // Si is odd (entire), so reflect to Re z >= 0
    }
    if (az <= detail::kComplexSeriesLimit) {
        return detail::toStdComplex(detail::siSeriesDD(detail::toComplexDD(z)));
    }
    if (z.real() == 0.0) {
        return detail::siImaginaryAxis(z.imag());
    }
    return detail::siAsymptotic(z);
}

/// Cosine integral, complex argument (principal branch: log z has its cut on
/// the negative real axis, approached from above for Im z = 0).
inline std::complex<double> cosIntegral(const std::complex<double>& z) {
    const double az = std::abs(z);
    if (az <= 8.0) {
        return detail::ciSeries(z);
    }
    if (z.real() < 0.0) {
        // Ci(-z) = Ci(z) -+ i pi for Im z > 0 / < 0 (the z^2 series is
        // entire), so the principal branch on Re z < 0 is Ci(-z) + i pi on
        // the upper side of the cut and Ci(-z) - i pi on the lower side.
        const double side = std::signbit(z.imag()) ? -1.0 : 1.0;
        return cosIntegral(-z) + std::complex<double>(0.0, side * kPi.hi);
    }
    if (az <= detail::kComplexSeriesLimit) {
        return detail::toStdComplex(detail::ciSeriesDD(detail::toComplexDD(z)));
    }
    if (z.real() == 0.0) {
        return detail::ciImaginaryAxis(z.imag());
    }
    return detail::ciAsymptotic(z);
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_SPECIAL_FUNCTIONS_TRIG_INTEGRALS_H
