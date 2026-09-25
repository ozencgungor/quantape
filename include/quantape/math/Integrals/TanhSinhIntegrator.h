#ifndef TANH_SINH_INTEGRATOR_H
#define TANH_SINH_INTEGRATOR_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "Integrator.h"

namespace quantape::math {
/**
 * @brief Tanh-sinh (double-exponential) quadrature, AD-compatible
 *
 * Implements the Takahasi–Mori double-exponential transform:
 *
 *   x = phi(t) = tanh((pi/2) sinh(t))
 *   int_{-1}^{1} f(x) dx = int_{-inf}^{inf} f(phi(t)) phi'(t) dt
 *   phi'(t) = (pi/2) cosh(t) / cosh^2((pi/2) sinh(t))
 *
 * and applies the trapezoidal rule with adaptive step halving. The
 * transformed integrand decays like exp(-c exp(|t|)) — double exponential —
 * so the trapezoidal rule converges exponentially in 1/h and nodes cluster
 * toward the endpoints, resolving endpoint singularities automatically.
 *
 * Refinement exploits the DOUBLING PROPERTY: at each level h -> h/2, only
 * the new odd-indexed nodes are evaluated and
 *
 *   I_k = (1/2) I_{k-1} + h_k * sum_{odd j} g(j h_k)
 *
 * so all previous evaluations are reused.
 *
 * AD design (matching the interpolation library's discipline):
 *   - abscissas and weights are computed in pure double — OFF the tape;
 *   - only the f-evaluations and the weighted sums carry AD values, so
 *     d/d(theta) int f(x; theta) dx flows through generically for double,
 *     var and fvar<var> (Hessians included, via nested AD).
 *
 * Numerical details from the design doc (docs/tanh_sinh.md):
 *   - sech^2 is computed through exp(-2|u|) to avoid cosh overflow;
 *   - nodes whose weight underflows below W_MIN are skipped entirely;
 *   - a boundary guard drops f-values that come back non-finite;
 *   - termination: err_k = |I_k - I_{k-1}| vs max(abs_tol, rel_tol |I_k|),
 *     with min/max level counts and thrashing detection;
 *   - the COMPLEMENT path (integrateWithComplement) implements Boost's
 *     two-argument functor: f receives the abscissa x and its signed
 *     distance d to the nearer endpoint (d = x -/+ 1), computed stably
 *     from the transform — for endpoint-singular integrands write
 *     f(x, d) in terms of d ((1 - x)^alpha -> (-d)^alpha near +1) to keep
 *     full precision where the plain x argument has already collapsed.
 *
 * @tparam DoubleT Numeric type (double, stan::math::var, stan::math::fvar<var>)
 */
template <typename DoubleT>
class TanhSinhIntegrator : public Integrator<DoubleT> {
public:
    using FunctionType = typename Integrator<DoubleT>::FunctionType;
    /// Complement-aware functor: f(x, d) where d = signed distance to the
    /// nearer endpoint (negative near +1, positive near -1).
    using ComplementFunctionType = std::function<DoubleT(DoubleT, double)>;

    /**
     * @param absAccuracy Absolute accuracy target for the level-to-level error
     * @param maxEvaluations Maximum number of function evaluations
     * @param relAccuracy Relative accuracy target (0 = not used)
     * @param minLevels Minimum refinement levels before termination is allowed
     * @param maxLevels Maximum refinement levels
     */
    TanhSinhIntegrator(double absAccuracy, size_t maxEvaluations, double relAccuracy = 0.0,
                       size_t minLevels = 4, size_t maxLevels = 15)
        : Integrator<DoubleT>(absAccuracy, maxEvaluations), m_relAccuracy(relAccuracy),
          m_minLevels(minLevels), m_maxLevels(maxLevels) {}

    /**
     * @brief Integrate f from a to b.
     *
     * double: the adaptive level halving implemented below. var and
     * fvar<var>: the converged quadrature rule is extracted in one
     * double-valued pass and the AD integrand samples are combined with it
     * through a single callback var per output (IntegratorStanPrimitives.h).
     * The complement and infinite-domain entry points keep the generic tape
     * construction.
     */
    DoubleT operator()(const FunctionType& f, DoubleT a, DoubleT b) const {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return Integrator<DoubleT>::operator()(f, a, b);
        } else {
            return integratePrimitives(f, a, b);
        }
    }

    /**
     * @brief Complement-aware integration (recommended for integrands that
     *        are singular or nearly singular at the endpoints).
     */
    DoubleT integrateWithComplement(const ComplementFunctionType& f, DoubleT a, DoubleT b) const {
        this->setNumberOfEvaluations(0);
        this->setAbsoluteError(0.0);
        const auto nodeEval = [&](double t) -> DoubleT {
            const double u = HALF_PI * std::sinh(t);
            const double e2 = std::exp(-2.0 * std::abs(u));
            const double comp = 2.0 * e2 / (1.0 + e2); // = 1 - |x|, no cancellation
            const double w = HALF_PI * std::cosh(t) * 4.0 * e2 / ((1.0 + e2) * (1.0 + e2));
            // NOTE: no COMP_MIN skip here — the complement is stable and
            // strictly positive for every node passing the W_MIN cut, so the
            // functor stays finite through the singularity and full precision
            // is preserved (the W_MIN cut alone controls the tail).
            if (w < W_MIN)
                return DoubleT(0.0);
            const double x = std::tanh(u);
            // Physical abscissa and its signed distance to the NEARER endpoint
            // of [a, b]: positive near a, negative near b. Computed from the
            // stable complement (times the affine jacobian), so it keeps full
            // precision exactly where x itself has collapsed onto ±1.
            const double x_phys = m_mid + m_half * x;
            const double d_phys = (u >= 0.0 ? -comp : comp) * m_half;
            const DoubleT fx = f(DoubleT(x_phys), d_phys);
            this->increaseNumberOfEvaluations(1);
            if (!isFiniteFastMathProof(value(fx)))
                return DoubleT(0.0);
            return fx * DoubleT(w * m_half);
        };
        return integrateImpl(nodeEval, a, b);
    }

    /**
     * @brief Integrate over the whole real line, int_{-inf}^{inf} f(x) dx
     *
     * Pre-transform t -> t/(1 - t^2) mapping (-1, 1) onto the real line,
     * composed with the tanh-sinh transform (jacobian (1 + t^2)/(1 - t^2)^2,
     * pure double — off the tape).
     */
    DoubleT integrateInfinite(const FunctionType& f) const {
        const auto mapped = [f](DoubleT t) -> DoubleT {
            const double tv = value(t);
            const double denom = 1.0 - tv * tv;
            const double x = tv / denom;
            const double jac = (1.0 + tv * tv) / (denom * denom);
            return f(DoubleT(x)) * DoubleT(jac);
        };
        return this->operator()(mapped, DoubleT(-1.0), DoubleT(1.0));
    }

    /**
     * @brief Integrate over [a, inf), pre-transform x = a + 2/(1 + t) - 1
     * mapping (-1, 1) onto [a, inf) (positive jacobian 2/(1 + t)^2).
     */
    DoubleT integrateToInfinity(const FunctionType& f, DoubleT a) const {
        const double av = value(a);
        const auto mapped = [f, av](DoubleT t) -> DoubleT {
            const double tv = value(t);
            const double x = av + 2.0 / (1.0 + tv) - 1.0;
            const double jac = 2.0 / ((1.0 + tv) * (1.0 + tv));
            return f(DoubleT(x)) * DoubleT(jac);
        };
        return this->operator()(mapped, DoubleT(-1.0), DoubleT(1.0));
    }

protected:
    DoubleT integrate(const FunctionType& f, DoubleT a, DoubleT b) const override {
        const auto nodeEval = [&](double t) -> DoubleT {
            const double u = HALF_PI * std::sinh(t);
            const double e2 = std::exp(-2.0 * std::abs(u));
            const double sech2 = 4.0 * e2 / ((1.0 + e2) * (1.0 + e2));
            const double w = HALF_PI * std::cosh(t) * sech2;
            const double comp = 2.0 * e2 / (1.0 + e2); // = 1 - |x|, no cancellation
            if (w < W_MIN || comp < COMP_MIN)
                return DoubleT(0.0); // tail/endpoint node: below double precision

            const double x = std::tanh(u);
            const DoubleT fx = f(DoubleT(m_mid + m_half * x));
            this->increaseNumberOfEvaluations(1);
            if (!isFiniteFastMathProof(value(fx)))
                return DoubleT(0.0); // boundary guard: pull the endpoint in
            return fx * DoubleT(w * m_half);
        };
        return integrateImpl(nodeEval, a, b);
    }

private:
    double m_relAccuracy;
    size_t m_minLevels;
    size_t m_maxLevels;

    // Defined for stan::math::var / stan::math::fvar<var> in
    // IntegratorStanPrimitives.h. Never ODR-used for double (nor for the
    // RuleScalar probe, which enters through the base operator()).
    DoubleT integratePrimitives(const FunctionType& f, DoubleT a, DoubleT b) const;

    // affine map state for the active integration (set by integrateImpl)
    mutable double m_mid = 0.0;
    mutable double m_half = 1.0;

    static constexpr double HALF_PI = 1.57079632679489661923;
    static constexpr double T_MAX = 7.0;   ///< weights underflow beyond this
    static constexpr double W_MIN = 1e-18; ///< skip nodes below this weight
    /// Skip nodes whose complement (distance to the nearer endpoint of
    /// [-1, 1]) is below this: this is what makes the integrator safe for
    /// ENDPOINT-SINGULAR integrands — the functor is never evaluated closer
    /// than ~4 eps to an endpoint, so an integrable singularity returns a
    /// large but FINITE value (no infinity ever enters the sum, which also
    /// makes the boundary guard fast-math-proof). The unresolved tail is
    /// O(sqrt(eps)) ~ 1e-8 for square-root singularities.
    static constexpr double COMP_MIN = 4.0 * std::numeric_limits<double>::epsilon();
    static constexpr double LEVEL0_STEP = 1.0;

    /// Recursive primal extraction: double -> itself, var -> .val(),
    /// fvar<var> -> .val().val()
    static double value(double x) { return x; }
    template <typename T>
    static double value(const T& x) {
        return value(x.val());
    }

    /**
     * @brief Bit-level finite check, immune to floating-point fast-math.
     *
     * This project's Release flags include -ffast-math (-ffinite-math-only),
     * under which std::isfinite is assumed true and gets compiled away —
     * the boundary guard would silently pass infinities into the sum. A
     * memcpy-based exponent check stays in the integer domain and cannot be
     * optimized under FP assumptions.
     */
    static bool isFiniteFastMathProof(double v) {
        std::uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return ((bits >> 52) & 0x7FFULL) != 0x7FFULL; // all-ones exponent = inf/nan
    }

    /**
     * @brief Adaptive level loop shared by the plain and complement paths.
     * @param nodeEval t -> weighted contribution (already includes phi'(t)
     *                 and the affine jacobian; 0 for skipped nodes)
     */
    template <typename NodeEval>
    DoubleT integrateImpl(const NodeEval& nodeEval, DoubleT a, DoubleT b) const {
        double av = value(a);
        double bv = value(b);
        if (av == bv) {
            this->setAbsoluteError(0.0);
            return DoubleT(0.0);
        }
        const bool reversed = av > bv;
        if (reversed)
            std::swap(av, bv);

        // affine map from [-1, 1] onto [a, b] — pure double, off the tape
        m_half = 0.5 * (bv - av);
        m_mid = 0.5 * (bv + av);

        double h = LEVEL0_STEP;
        DoubleT I_prev = DoubleT(0.0);
        DoubleT I = DoubleT(0.0);
        const double BIG = std::numeric_limits<double>::max();
        double err = BIG;
        double prev_err = BIG;
        int worsening = 0;

        for (size_t level = 0; level <= m_maxLevels; ++level) {
            if (level == 0) {
                DoubleT sum = nodeEval(0.0);
                const size_t n_tail = static_cast<size_t>(std::ceil(T_MAX / h));
                for (size_t j = 1; j <= n_tail; ++j) {
                    const double t = j * h;
                    sum += nodeEval(t) + nodeEval(-t);
                }
                I = DoubleT(h) * sum;
                I_prev = I;
            } else {
                // doubling property: only the NEW odd-indexed nodes
                DoubleT odd_sum = DoubleT(0.0);
                for (size_t m = 0;; ++m) {
                    const double t = (2.0 * m + 1.0) * h;
                    if (t > T_MAX)
                        break;
                    odd_sum += nodeEval(t) + nodeEval(-t);
                }
                I = DoubleT(0.5) * I_prev + DoubleT(h) * odd_sum;
                err = std::abs(value(I) - value(I_prev));
            }

            if (this->numberOfEvaluations() > this->maxEvaluations())
                throw std::runtime_error(
                    "TanhSinhIntegrator: max number of function evaluations reached");

            if (level > 0) {
                // convergence first: accept as soon as we are within tolerance
                if (level >= m_minLevels) {
                    const double tol =
                        std::max(this->absoluteAccuracy(), m_relAccuracy * std::abs(value(I)));
                    if (err <= tol)
                        break;
                }
                // thrashing detection: abort only when the error gets
                // SUBSTANTIALLY worse repeatedly — at the rounding-noise
                // floor the level-to-level change wobbles randomly, and
                // aborting on any increase would stop refinement far too
                // early for tight tolerances.
                if (err > 1.5 * prev_err) {
                    if (++worsening >= 3)
                        break;
                } else {
                    worsening = 0;
                }
                prev_err = err;
            }

            I_prev = I;
            h *= 0.5;
        }

        this->setAbsoluteError(err);
        return reversed ? -I : I;
    }
};

} // namespace quantape::math

#endif // TANH_SINH_INTEGRATOR_H
