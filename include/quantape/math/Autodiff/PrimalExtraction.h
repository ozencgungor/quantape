#ifndef PRIMAL_EXTRACTION_H
#define PRIMAL_EXTRACTION_H

namespace quantape::math {
namespace detail {
/// Recursive primal extraction: double -> itself, var -> .val(),
/// `fvar<var>` -> .val().val().
///
/// Shared by the integrator AD specializations (Integrals/IntegratorStanPrimitives.h)
/// and the 1-D solvers (Solvers/SolverPrimitives.h). Deliberately Stan-free so
/// double-only translation units never need the Stan include paths.
inline double primalValue(double x) {
    return x;
}

template <typename T>
double primalValue(const T& x) {
    return primalValue(x.val());
}
} // namespace detail
} // namespace quantape::math

#endif // PRIMAL_EXTRACTION_H
