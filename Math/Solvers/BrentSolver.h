#ifndef BRENT_SOLVER_H
#define BRENT_SOLVER_H

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
 * @tparam DoubleT Numeric type (double or stan::math::var)
 */
template <typename DoubleT>
class BrentSolver : public Solver1D<DoubleT, BrentSolver<DoubleT>> {
public:
    using Base = Solver1D<DoubleT, BrentSolver<DoubleT>>;
    using FunctionType = typename Base::FunctionType;

    BrentSolver() = default;

    DoubleT solveImpl(const FunctionType& f, double accuracy) const {
        DoubleT a = this->m_xMin;
        DoubleT b = this->m_xMax;
        DoubleT fa = this->m_fxMin;
        DoubleT fb = this->m_fxMax;

        DoubleT c = a;
        DoubleT fc = fa;
        DoubleT d = b - a;
        DoubleT e = d;

        while (this->m_evaluationNumber < this->maxEvaluations()) {
            if (std::fabs(value(fc)) < std::fabs(value(fb))) {
                a = b;
                b = c;
                c = a;
                fa = fb;
                fb = fc;
                fc = fa;
            }

            DoubleT tol =
                DoubleT(2.0 * std::numeric_limits<double>::epsilon() * std::fabs(value(b)) +
                        0.5 * accuracy);
            DoubleT m = (c - b) / DoubleT(2.0);

            if (std::fabs(value(m)) <= value(tol) || std::fabs(value(fb)) < accuracy) {
                return b;
            }

            if (std::fabs(value(e)) < value(tol) || std::fabs(value(fa)) <= std::fabs(value(fb))) {
                // Bisection
                d = m;
                e = m;
            } else {
                DoubleT s;
                if (value(a) == value(c)) {
                    // Linear interpolation (secant)
                    s = fb * (b - a) / (fa - fb);
                } else {
                    // Inverse quadratic interpolation
                    DoubleT p, q, r;
                    q = fa / fc;
                    r = fb / fc;
                    p = (b - a) * r * (q - r) - (b - c) * (DoubleT(1.0) - r);
                    q = (q - DoubleT(1.0)) * (r - DoubleT(1.0)) * ((DoubleT(1.0) - r));
                    s = b - p / q;
                }

                if ((value(s) - value(b)) * (value(b) - value(c)) < 0.0 ||
                    std::fabs(value(s) - value(b)) > std::fabs(value(e) / 2.0)) {
                    d = m;
                    e = m;
                } else {
                    e = d;
                    d = s - b;
                }
            }

            a = b;
            fa = fb;

            if (std::fabs(value(d)) > value(tol)) {
                b = b + d;
            } else {
                b = b + (value(m) > 0.0 ? tol : -tol);
            }

            fb = f(b);
            this->m_evaluationNumber++;

            if (value(fb) * value(fc) > 0.0) {
                c = a;
                fc = fa;
                d = b - a;
                e = d;
            }
        }

        throw std::runtime_error("BrentSolver: max number of iterations reached");
    }

private:
    static double value(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return x;
        } else {
            return x.val();
        }
    }
};
} // namespace Math

#endif // BRENT_SOLVER_H
