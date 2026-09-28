#ifndef QUANTAPE_MATH_SPECIAL_FUNCTIONS_TRIG_INTEGRALS_H
#define QUANTAPE_MATH_SPECIAL_FUNCTIONS_TRIG_INTEGRALS_H

#include <cmath>
#include <complex>
#include <stdexcept>

namespace quantape::math {
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
 * with complex `phi`.
 *
 * Real branch accuracy: the published rational approximations from the
 * GALSIM toolkit (Rowe et al., 2014, arXiv:1407.7676; the same coefficient
 * sets used by QuantLib's `exponentialintegrals.cpp`, BSD-licensed) give
 * ~1e-16 relative accuracy for all x >= 0, with the power series used below
 * x = 4. Complex branch: the power series for |z| <= 8 — the CV arguments
 * are small (`|phi| ~ (v0 + T kappa theta) sqrt(1-rho^2)/sigma`), and this
 * is the only internal use; larger complex arguments throw (documented).
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
            x2 * (7.44437068161936700618e2 +
                  x2 * (1.96396372895146869801e5 +
                        x2 * (2.37750310125431834034e7 +
                              x2 * (1.43073403821274636888e9 +
                                    x2 * (4.33736238870432522765e10 +
                                          x2 * (6.40533830574022022911e11 +
                                                x2 * (4.20968180571076940208e12 +
                                                      x2 * (1.00795182980368574617e13 +
                                                            x2 * (4.94816688199951963482e12 -
                                                                  x2 * 4.94701168645415959931e11)))))))))) /
           (x * (1.0 +
                 x2 * (7.46437068161927678031e2 +
                       x2 * (1.97865247031583951450e5 +
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
            x2 * (8.1359520115168615e2 +
                  x2 * (2.35239181626478200e5 +
                        x2 * (3.12557570795778731e7 +
                              x2 * (2.06297595146763354e9 +
                                    x2 * (6.83052205423625007e10 +
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
               (1.0 +
                x2 * (-4.54393409816329991e-2 +
                      x2 * (1.15457225751016682e-3 +
                            x2 * (-1.41018536821330254e-5 +
                                  x2 * (9.43280809438713025e-8 +
                                        x2 * (-3.53201978997168357e-10 +
                                              x2 * (7.08240282274875911e-13 -
                                                    x2 * 6.05338212010422477e-16))))))) /
               (1.0 +
                x2 * (1.01162145739225565e-2 +
                      x2 * (4.99175116169755106e-5 +
                            x2 * (1.55654986308745614e-7 +
                                  x2 * (3.28067571055789734e-10 +
                                        x2 * (4.5049097575386581e-13 +
                                              x2 * 3.21107051193712168e-16))))));
    }
    return M_PI_2 - detail::galsimF(x) * cos(x) - detail::galsimG(x) * sin(x);
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
               x2 * (-0.25 +
                     x2 * (7.51851524438898291e-3 +
                           x2 * (-1.27528342240267686e-4 +
                                 x2 * (1.05297363846239184e-6 +
                                       x2 * (-4.68889508144848019e-9 +
                                             x2 * (1.06480802891189243e-11 -
                                                   x2 * 9.93728488857585407e-15)))))) /
                   (1.0 +
                    x2 * (1.1592605689110735e-2 +
                          x2 * (6.72126800814254432e-5 +
                                x2 * (2.55533277086129636e-7 +
                                      x2 * (6.97071295760958946e-10 +
                                            x2 * (1.38536352772778619e-12 +
                                                  x2 * (1.89106054713059759e-15 +
                                                        x2 * 1.39759616731376855e-18)))))));
    }
    return detail::galsimF(x) * sin(x) - detail::galsimG(x) * cos(x);
}

/// Sine integral, complex argument (series branch; |z| <= 8).
inline std::complex<double> sinIntegral(const std::complex<double>& z) {
    if (std::abs(z) > 8.0) {
        throw std::invalid_argument("sinIntegral: complex series limited to |z| <= 8");
    }
    return detail::siSeries(z);
}

/// Cosine integral, complex argument (principal log; |z| <= 8).
inline std::complex<double> cosIntegral(const std::complex<double>& z) {
    if (std::abs(z) > 8.0) {
        throw std::invalid_argument("cosIntegral: complex series limited to |z| <= 8");
    }
    return detail::ciSeries(z);
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_SPECIAL_FUNCTIONS_TRIG_INTEGRALS_H
