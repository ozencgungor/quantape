#ifndef BRENT_SOLVER_H
#define BRENT_SOLVER_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "Solver1DBase.h"

namespace Math {
/**
 * @brief Brent's method solver
 *
 * Combines bisection, secant method, and inverse quadratic interpolation
 * for robust and fast root finding.
 *
 * Brent's method is generally considered the best general-purpose root finder:
 * - Guaranteed convergence (like bisection)
 * - Super-linear convergence (like secant/inverse quadratic)
 * - No derivative required
 *
 * Formulation follows Forsythe-Malcolm-Moler's zeroin (also used by scipy's
 * brentq): (a,b,c) always bracket the root, b is the best estimate, and d/e
 * carry the last two steps.
 *
 * @tparam DoubleT Numeric type (double or stan::math::var)
 */
template <typename DoubleT>
class BrentSolver : public Solver1D<DoubleT, BrentSolver<DoubleT>> {
public:
    using Base = Solver1D<DoubleT, BrentSolver<DoubleT>>;

    BrentSolver() = default;

    template <typename F>
    DoubleT solveImpl(const F& f, double accuracy, SolverState<DoubleT>& s) const {
        DoubleT a = s.xMin;
        DoubleT b = s.xMax;
        DoubleT c = s.xMax;
        DoubleT fa = s.fxMin;
        DoubleT fb = s.fxMax;
        DoubleT fc = s.fxMax;
        DoubleT d = b - a;
        DoubleT e = d;

        while (s.evaluations < s.maxEvaluations) {
            // f(b) and f(c) must straddle the root; otherwise rebracket
            if (!Base::oppositeSigns(fb, fc)) {
                c = a;
                fc = fa;
                d = b - a;
                e = d;
            }

            // Keep b the best estimate. The net effect is
            // (a,b,c) <- (b_old,c_old,b_old), so a == c marks the linear
            // (secant) interpolation branch below; it is not a typo.
            if (std::fabs(Base::value(fc)) < std::fabs(Base::value(fb))) {
                const DoubleT oldB = b;
                b = c;
                a = oldB;
                c = oldB;
                const DoubleT oldFb = fb;
                fb = fc;
                fa = oldFb;
                fc = oldFb;
            }

            const DoubleT tol1 =
                DoubleT(2.0 * std::numeric_limits<double>::epsilon() * std::fabs(Base::value(b)) +
                        0.5 * accuracy);
            const DoubleT xm = (c - b) / DoubleT(2.0);

            if (std::fabs(Base::value(xm)) <= Base::value(tol1) || Base::value(fb) == 0.0 ||
                Base::isZero(fb, s.fScale)) {
                return b;
            }

            if (std::fabs(Base::value(e)) >= Base::value(tol1) &&
                std::fabs(Base::value(fa)) > std::fabs(Base::value(fb))) {
                DoubleT p, q;
                const DoubleT r0 = fb / fa;

                if (Base::value(a) == Base::value(c)) {
                    // Linear interpolation (secant) through a and b
                    p = DoubleT(2.0) * xm * r0;
                    q = DoubleT(1.0) - r0;
                } else {
                    // Inverse quadratic interpolation through a, b, c
                    q = fa / fc;
                    const DoubleT r = fb / fc;
                    p = r0 * (DoubleT(2.0) * xm * q * (q - r) - (b - a) * (r - DoubleT(1.0)));
                    q = (q - DoubleT(1.0)) * (r - DoubleT(1.0)) * (r0 - DoubleT(1.0));
                }

                if (Base::value(p) > 0.0) {
                    q = -q;
                }
                if (Base::value(p) < 0.0) {
                    p = -p;
                }

                // Accept the interpolated step only if it falls well inside
                // the bracket and shrinks it by at least a factor of two
                const double min1 = 3.0 * Base::value(xm) * Base::value(q) -
                                    std::fabs(Base::value(tol1) * Base::value(q));
                const double min2 = std::fabs(Base::value(e) * Base::value(q));
                if (2.0 * Base::value(p) < std::min(min1, min2)) {
                    e = d;
                    d = p / q;
                } else {
                    d = xm;
                    e = d;
                }
            } else {
                d = xm;
                e = d;
            }

            a = b;
            fa = fb;
            if (std::fabs(Base::value(d)) > Base::value(tol1)) {
                b = b + d;
            } else {
                b = b + DoubleT(Base::value(xm) > 0.0 ? Base::value(tol1) : -Base::value(tol1));
            }
            fb = f(b);
            ++s.evaluations;
        }

        throw std::runtime_error("BrentSolver: max number of iterations reached");
    }
};
} // namespace Math

#endif // BRENT_SOLVER_H
