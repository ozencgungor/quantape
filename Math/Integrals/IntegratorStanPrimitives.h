#ifndef INTEGRATOR_STAN_PRIMITIVES_H
#define INTEGRATOR_STAN_PRIMITIVES_H

//
// IntegratorStanPrimitives.h -- Stan AD specializations for the Integrator classes
//
// Every integrator in Math/Integrals is a FIXED LINEAR functional of the
// integrand samples once the (value-only) adaptive refinement has chosen the
// rule:
//
//     I(theta) = sum_k c_k f(x_k; theta),   x_k, c_k pure double
//
// so every derivative order commutes with the integral:
//
//     d^m I / d theta^m = sum_k c_k * d^m f_k / d theta^m.
//
// The specializations below exploit this in ONE pass:
//
//   1. the integrator is re-run on detail::RuleScalar — a probe scalar that
//      carries the primal value (so adaptivity sees exactly the double
//      behaviour) plus a linear-expression DAG over evaluation slots of
//      O(1) cost per arithmetic operation. The final expression IS the
//      converged discrete quadrature rule;
//   2. the same run evaluates the integrand with the AD scalar (double
//      nodes convert implicitly), so the raw AD f-evaluations f_k are kept;
//   3. the integral is assembled as sum_k c_k f_k with ONE callback var per
//      output component whose adjoint pushes c_k into f_k. Reverse mode
//      therefore sees O(#nodes) tape nodes instead of the ~2 extra
//      accumulate/multiply nodes per quadrature node, and the fvar<var>
//      Hessian is the weighted sum sum_k c_k H_k with no nested accumulate
//      tape.
//
// Following the interpolation pattern (InterpolationStanPrimitives.h) and
// the pricing pattern (Pricing/StanPrimitives.h), user code never calls
// anything extra: include this header alongside the integrator headers (or
// an umbrella that pulls it in) and use the integrators as usual. The
// DoubleT template parameter selects the path automatically:
//
//   Math::TrapezoidIntegratorDefault<double>  integ(a, n);   // plain
//   Math::TrapezoidIntegratorDefault<var>     integ(a, n);   // primitives
//   Math::TrapezoidIntegratorDefault<fvar<var>> integ(a, n); // primitives
//
//   stan::math::var I = integ(f, stan::math::var(0.0), stan::math::var(1.0));
//   I.grad();   // theta adjoints now hold dI/dtheta
//
// Bounds are differentiated as well: the frozen rule extends exactly to AD
// bounds through  I = (b - a) * sum_k c_hat_k f(a + (b - a) u_k)  (nodes are
// affine in the bounds, quadrature weights are homogeneous of degree one in
// the interval length). This matches the generic tape path's derivative of
// the same frozen rule.
//

#include <stan/math.hpp>
#include <stan/math/mix.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "GaussLobattoIntegrator.h"
#include "GaussianQuadrature.h"
#include "SimpsonIntegrator.h"
#include "TanhSinhIntegrator.h"
#include "TrapezoidIntegrator.h"

namespace Math {

/// Tag used to rebind an integrator to another scalar type (tests, tools).
template <typename T>
struct ScalarTag {
    using type = T;
};

namespace detail {

struct RuleNode;
using RuleNodePtr = std::shared_ptr<const RuleNode>;

/**
 * @brief One node of the linear-expression DAG carried by RuleScalar.
 *
 * Scale: scale * a; Add: a + b; Sub: a - b; Slot: the k-th f-evaluation.
 * The evaluation coefficients are constants, so the whole DAG stays linear
 * and each operation costs O(1) (no per-scalar coefficient vectors).
 */
struct RuleNode {
    enum class Kind : unsigned char { Slot, Scale, Add, Sub };

    Kind kind;
    std::size_t slot = 0;
    double scale = 0.0;
    RuleNodePtr a;
    RuleNodePtr b;
};

/**
 * @brief Rule-recording scalar: primal value + linear expression over slots.
 *
 * Passed through the integrator's arithmetic unchanged (the integrator only
 * ever combines f-evaluations linearly: additions and multiplications by
 * pure-double weights/nodes). The integral's expression is the discrete
 * quadrature rule, and accumulate() reads its coefficients off in O(#slots).
 *
 * Constants (constructed from double) have an empty expression and are
 * treated as slot-free values. Products/quotients of two slot-bearing
 * scalars would be nonlinear in the f-evaluations and throw: no integrator
 * in this library performs them.
 */
struct RuleScalar {
    double v = 0.0;
    RuleNodePtr expr;

    RuleScalar() = default;
    RuleScalar(double value) : v(value) {} // constant: empty expr
    RuleScalar(double value, std::size_t slot)
        : v(value), expr(std::make_shared<const RuleNode>(
                        RuleNode{RuleNode::Kind::Slot, slot, 0.0, nullptr, nullptr})) {}
    RuleScalar(double value, RuleNodePtr e) : v(value), expr(std::move(e)) {}

    double val() const { return v; }

    static RuleNodePtr add(const RuleNodePtr& x, const RuleNodePtr& y) {
        if (!x)
            return y;
        if (!y)
            return x;
        return std::make_shared<const RuleNode>(RuleNode{RuleNode::Kind::Add, 0, 0.0, x, y});
    }
    static RuleNodePtr sub(const RuleNodePtr& x, const RuleNodePtr& y) {
        if (!y)
            return x;
        if (!x)
            return scale(y, -1.0);
        return std::make_shared<const RuleNode>(RuleNode{RuleNode::Kind::Sub, 0, 0.0, x, y});
    }
    static RuleNodePtr scale(const RuleNodePtr& x, double c) {
        if (!x || c == 1.0)
            return x;
        return std::make_shared<const RuleNode>(RuleNode{RuleNode::Kind::Scale, 0, c, x, nullptr});
    }

    [[noreturn]] static void nonlinear() {
        throw std::logic_error(
            "Math::detail::RuleScalar: integrator combination is nonlinear in the f-evaluations; "
            "the primitives rule-extraction path requires a fixed linear quadrature rule");
    }

    RuleScalar operator-() const { return RuleScalar(-v, scale(expr, -1.0)); }
    RuleScalar operator+(const RuleScalar& o) const {
        return RuleScalar(v + o.v, add(expr, o.expr));
    }
    RuleScalar operator-(const RuleScalar& o) const {
        return RuleScalar(v - o.v, sub(expr, o.expr));
    }
    RuleScalar operator*(const RuleScalar& o) const {
        if (expr && o.expr)
            nonlinear();
        return RuleScalar(v * o.v, expr ? scale(expr, o.v) : scale(o.expr, v));
    }
    RuleScalar operator/(const RuleScalar& o) const {
        if (o.expr)
            nonlinear();
        return RuleScalar(v / o.v, scale(expr, 1.0 / o.v));
    }

    RuleScalar& operator+=(const RuleScalar& o) { return *this = *this + o; }
    RuleScalar& operator-=(const RuleScalar& o) { return *this = *this - o; }
    RuleScalar& operator*=(const RuleScalar& o) { return *this = *this * o; }
    RuleScalar& operator/=(const RuleScalar& o) { return *this = *this / o; }

    /// Accumulate this expression's coefficients into coeffs[slot].
    void accumulate(std::vector<double>& coeffs) const { accumulate(expr.get(), 1.0, coeffs); }

private:
    static void accumulate(const RuleNode* n, double w, std::vector<double>& coeffs) {
        if (n == nullptr)
            return;
        switch (n->kind) {
            case RuleNode::Kind::Slot:
                coeffs[n->slot] += w;
                break;
            case RuleNode::Kind::Scale:
                accumulate(n->a.get(), w * n->scale, coeffs);
                break;
            case RuleNode::Kind::Add:
                accumulate(n->a.get(), w, coeffs);
                accumulate(n->b.get(), w, coeffs);
                break;
            case RuleNode::Kind::Sub:
                accumulate(n->a.get(), w, coeffs);
                accumulate(n->b.get(), -w, coeffs);
                break;
        }
    }
};

inline RuleScalar operator+(double x, const RuleScalar& o) {
    return RuleScalar(x) + o;
}
inline RuleScalar operator-(double x, const RuleScalar& o) {
    return RuleScalar(x) - o;
}
inline RuleScalar operator*(double x, const RuleScalar& o) {
    return RuleScalar(x) * o;
}
inline RuleScalar operator/(double x, const RuleScalar& o) {
    return RuleScalar(x) / o;
}

/// Recursive primal extraction: double -> itself, var -> .val(),
/// fvar<var> -> .val().val()
inline double primalValue(double x) {
    return x;
}
template <typename T>
double primalValue(const T& x) {
    return primalValue(x.val());
}

/**
 * @brief One-pass primitives integration.
 *
 * Runs @p integ (the same integrator, instantiated on RuleScalar) over the
 * integrand, evaluating the integrand with the AD scalar type S and keeping
 * every evaluation. The RuleScalar pass yields the discrete quadrature
 * coefficients c_k, and the integral is assembled from the weighted
 * f-evaluations with one callback var per output component.
 *
 * The frozen rule is extended to AD bounds exactly: nodes are affine in the
 * bounds (x_k = a + (b - a) u_k with u_k the frozen relative position) and
 * quadrature weights are homogeneous of degree one in the interval length
 * (c_k = (b - a) c_hat_k), so
 *
 *     I = (b - a) * sum_k c_hat_k f(a + (b - a) u_k)
 *
 * carries dI/da, dI/db and the parameter path through the integrand.
 *
 * The RuleScalar pass enters through the BASE operator() explicitly, so the
 * derived class's AD dispatch is bypassed and the plain numerical algorithm
 * runs (for RuleScalar, which is neither double nor an AD scalar).
 */
template <typename S, typename RuleIntegrator, typename Fn>
S integrateWithRule(RuleIntegrator& integ, const Fn& f, S a, S b) {
    const double av = primalValue(a);
    const double bv = primalValue(b);
    const double span = bv - av;
    const double inv_span = (span != 0.0) ? 1.0 / span : 0.0;

    std::vector<S> fks;

    const auto wrapped = [&](RuleScalar x) -> RuleScalar {
        const double u = (x.val() - av) * inv_span; // frozen relative position
        fks.push_back(f(a + (b - a) * u));          // AD node, affine in bounds
        return RuleScalar(primalValue(fks.back()), fks.size() - 1);
    };

    const RuleScalar result =
        static_cast<const Integrator<RuleScalar>&>(integ)(wrapped, RuleScalar(av), RuleScalar(bv));

    // Shape coefficients c_hat_k; the interval scaling stays on the tape as
    // the final (b - a) factor below.
    std::vector<double> cs(fks.size(), 0.0);
    result.accumulate(cs);
    for (double& c : cs)
        c *= inv_span;

    if constexpr (stan::is_var<S>::value) {
        double sum = 0.0;
        for (std::size_t k = 0; k < fks.size(); ++k)
            sum += cs[k] * fks[k].val();

        stan::math::var core = stan::math::make_callback_var(sum, [fks, cs](auto& vi) {
            const double adj = vi.adj();
            for (std::size_t k = 0; k < fks.size(); ++k)
                fks[k].adj() += adj * cs[k];
        });
        return (b - a) * core;
    } else {
        using stan::math::fvar;
        using stan::math::make_callback_var;
        using stan::math::var;

        double sum = 0.0;
        double tangent_val = 0.0;
        for (std::size_t k = 0; k < fks.size(); ++k) {
            sum += cs[k] * fks[k].val_.val();
            tangent_val += cs[k] * fks[k].d_.val();
        }

        var val = make_callback_var(sum, [fks, cs](auto& vi) {
            const double adj = vi.adj();
            for (std::size_t k = 0; k < fks.size(); ++k)
                fks[k].val_.adj() += adj * cs[k];
        });
        var tangent = make_callback_var(tangent_val, [fks, cs](auto& vi) {
            const double adj = vi.adj();
            for (std::size_t k = 0; k < fks.size(); ++k)
                fks[k].d_.adj() += adj * cs[k];
        });
        return (b - a) * fvar<var>(val, tangent);
    }
}

} // namespace detail

// ============================================================================
// TrapezoidIntegrator (both refinement policies)
// ============================================================================

template <>
inline stan::math::var TrapezoidIntegrator<stan::math::var, DefaultPolicy>::integratePrimitives(
    const FunctionType& f, stan::math::var a, stan::math::var b) const {
    TrapezoidIntegrator<detail::RuleScalar, DefaultPolicy> rule(this->absoluteAccuracy(),
                                                                this->maxEvaluations());
    const stan::math::var result = detail::integrateWithRule<stan::math::var>(rule, f, a, b);
    this->setNumberOfEvaluations(rule.numberOfEvaluations());
    this->setAbsoluteError(rule.absoluteError());
    return result;
}

template <>
inline stan::math::fvar<stan::math::var>
TrapezoidIntegrator<stan::math::fvar<stan::math::var>, DefaultPolicy>::integratePrimitives(
    const FunctionType& f, stan::math::fvar<stan::math::var> a,
    stan::math::fvar<stan::math::var> b) const {
    using fvv = stan::math::fvar<stan::math::var>;
    TrapezoidIntegrator<detail::RuleScalar, DefaultPolicy> rule(this->absoluteAccuracy(),
                                                                this->maxEvaluations());
    const fvv result = detail::integrateWithRule<fvv>(rule, f, a, b);
    this->setNumberOfEvaluations(rule.numberOfEvaluations());
    this->setAbsoluteError(rule.absoluteError());
    return result;
}

template <>
inline stan::math::var TrapezoidIntegrator<stan::math::var, MidPointPolicy>::integratePrimitives(
    const FunctionType& f, stan::math::var a, stan::math::var b) const {
    TrapezoidIntegrator<detail::RuleScalar, MidPointPolicy> rule(this->absoluteAccuracy(),
                                                                 this->maxEvaluations());
    const stan::math::var result = detail::integrateWithRule<stan::math::var>(rule, f, a, b);
    this->setNumberOfEvaluations(rule.numberOfEvaluations());
    this->setAbsoluteError(rule.absoluteError());
    return result;
}

template <>
inline stan::math::fvar<stan::math::var>
TrapezoidIntegrator<stan::math::fvar<stan::math::var>, MidPointPolicy>::integratePrimitives(
    const FunctionType& f, stan::math::fvar<stan::math::var> a,
    stan::math::fvar<stan::math::var> b) const {
    using fvv = stan::math::fvar<stan::math::var>;
    TrapezoidIntegrator<detail::RuleScalar, MidPointPolicy> rule(this->absoluteAccuracy(),
                                                                 this->maxEvaluations());
    const fvv result = detail::integrateWithRule<fvv>(rule, f, a, b);
    this->setNumberOfEvaluations(rule.numberOfEvaluations());
    this->setAbsoluteError(rule.absoluteError());
    return result;
}

// ============================================================================
// SimpsonIntegrator (Richardson-extrapolated trapezoid)
// ============================================================================

template <>
inline stan::math::var
SimpsonIntegrator<stan::math::var>::integratePrimitives(const FunctionType& f, stan::math::var a,
                                                        stan::math::var b) const {
    SimpsonIntegrator<detail::RuleScalar> rule(this->absoluteAccuracy(), this->maxEvaluations());
    const stan::math::var result = detail::integrateWithRule<stan::math::var>(rule, f, a, b);
    this->setNumberOfEvaluations(rule.numberOfEvaluations());
    this->setAbsoluteError(rule.absoluteError());
    return result;
}

template <>
inline stan::math::fvar<stan::math::var>
SimpsonIntegrator<stan::math::fvar<stan::math::var>>::integratePrimitives(
    const FunctionType& f, stan::math::fvar<stan::math::var> a,
    stan::math::fvar<stan::math::var> b) const {
    using fvv = stan::math::fvar<stan::math::var>;
    SimpsonIntegrator<detail::RuleScalar> rule(this->absoluteAccuracy(), this->maxEvaluations());
    const fvv result = detail::integrateWithRule<fvv>(rule, f, a, b);
    this->setNumberOfEvaluations(rule.numberOfEvaluations());
    this->setAbsoluteError(rule.absoluteError());
    return result;
}

// ============================================================================
// GaussLobattoIntegrator (adaptive 6-way subdivision)
// ============================================================================

template <>
inline stan::math::var GaussLobattoIntegrator<stan::math::var>::integratePrimitives(
    const FunctionType& f, stan::math::var a, stan::math::var b) const {
    GaussLobattoIntegrator<detail::RuleScalar> rule(
        this->absoluteAccuracy(), this->maxEvaluations(), m_relAccuracy, m_useConvergenceEstimate);
    const stan::math::var result = detail::integrateWithRule<stan::math::var>(rule, f, a, b);
    this->setNumberOfEvaluations(rule.numberOfEvaluations());
    this->setAbsoluteError(rule.absoluteError());
    return result;
}

template <>
inline stan::math::fvar<stan::math::var>
GaussLobattoIntegrator<stan::math::fvar<stan::math::var>>::integratePrimitives(
    const FunctionType& f, stan::math::fvar<stan::math::var> a,
    stan::math::fvar<stan::math::var> b) const {
    using fvv = stan::math::fvar<stan::math::var>;
    GaussLobattoIntegrator<detail::RuleScalar> rule(
        this->absoluteAccuracy(), this->maxEvaluations(), m_relAccuracy, m_useConvergenceEstimate);
    const fvv result = detail::integrateWithRule<fvv>(rule, f, a, b);
    this->setNumberOfEvaluations(rule.numberOfEvaluations());
    this->setAbsoluteError(rule.absoluteError());
    return result;
}

// ============================================================================
// GaussLegendreIntegrator (fixed-order quadrature)
// ============================================================================

template <>
inline stan::math::var GaussLegendreIntegrator<stan::math::var>::integratePrimitives(
    const FunctionType& f, stan::math::var a, stan::math::var b) const {
    GaussLegendreIntegrator<detail::RuleScalar> rule(m_quadrature.order());
    const stan::math::var result = detail::integrateWithRule<stan::math::var>(rule, f, a, b);
    this->setNumberOfEvaluations(rule.numberOfEvaluations());
    this->setAbsoluteError(rule.absoluteError());
    return result;
}

template <>
inline stan::math::fvar<stan::math::var>
GaussLegendreIntegrator<stan::math::fvar<stan::math::var>>::integratePrimitives(
    const FunctionType& f, stan::math::fvar<stan::math::var> a,
    stan::math::fvar<stan::math::var> b) const {
    using fvv = stan::math::fvar<stan::math::var>;
    GaussLegendreIntegrator<detail::RuleScalar> rule(m_quadrature.order());
    const fvv result = detail::integrateWithRule<fvv>(rule, f, a, b);
    this->setNumberOfEvaluations(rule.numberOfEvaluations());
    this->setAbsoluteError(rule.absoluteError());
    return result;
}

// ============================================================================
// TanhSinhIntegrator (double-exponential quadrature)
//
// Only the plain bounded path dispatches here. The complement / infinite
// domain entry points keep the generic tape construction: their transforms
// mix the abscissa into the integrand arguments, so the fixed-rule identity
// does not hold as directly.
// ============================================================================

template <>
inline stan::math::var
TanhSinhIntegrator<stan::math::var>::integratePrimitives(const FunctionType& f, stan::math::var a,
                                                         stan::math::var b) const {
    TanhSinhIntegrator<detail::RuleScalar> rule(this->absoluteAccuracy(), this->maxEvaluations(),
                                                m_relAccuracy, m_minLevels, m_maxLevels);
    const stan::math::var result = detail::integrateWithRule<stan::math::var>(rule, f, a, b);
    this->setNumberOfEvaluations(rule.numberOfEvaluations());
    this->setAbsoluteError(rule.absoluteError());
    return result;
}

template <>
inline stan::math::fvar<stan::math::var>
TanhSinhIntegrator<stan::math::fvar<stan::math::var>>::integratePrimitives(
    const FunctionType& f, stan::math::fvar<stan::math::var> a,
    stan::math::fvar<stan::math::var> b) const {
    using fvv = stan::math::fvar<stan::math::var>;
    TanhSinhIntegrator<detail::RuleScalar> rule(this->absoluteAccuracy(), this->maxEvaluations(),
                                                m_relAccuracy, m_minLevels, m_maxLevels);
    const fvv result = detail::integrateWithRule<fvv>(rule, f, a, b);
    this->setNumberOfEvaluations(rule.numberOfEvaluations());
    this->setAbsoluteError(rule.absoluteError());
    return result;
}

} // namespace Math

#endif // INTEGRATOR_STAN_PRIMITIVES_H
