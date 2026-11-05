#ifndef LIGHTAPE_EIGEN_LEAN_REVERSE_EIGEN_H
#define LIGHTAPE_EIGEN_LEAN_REVERSE_EIGEN_H

#include <Eigen/Core>

#include "lightape/rev/LeanReverse.h"

namespace Eigen {
template <>
struct NumTraits<lightape::RevScalar> : NumTraits<double> {
    using Real = lightape::RevScalar;
    using NonInteger = lightape::RevScalar;
    using Nested = lightape::RevScalar;
    using Literal = lightape::RevScalar;
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

// Mixed RevScalar/double expressions evaluate to RevScalar so that the
// recorded graph stays connected across the boundary. Same-type combinations
// are already handled by Eigen's generic ScalarBinaryOpTraits<T, T, Op>.
template <typename BinaryOp>
struct ScalarBinaryOpTraits<lightape::RevScalar, double, BinaryOp> {
    using ReturnType = lightape::RevScalar;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, lightape::RevScalar, BinaryOp> {
    using ReturnType = lightape::RevScalar;
};

// Integer operands promote to RevScalar as well. `numext::pow(x, 2)` asks for
// `ScalarBinaryOpTraits<RevScalar, T, scalar_pow_op<...>>` while resolving
// `internal::pow_impl::result_type`; without these, that eager lookup is a
// hard error (not SFINAE), even when a better overload exists. The same traits
// keep mixed integer arithmetic and `array().pow(2)` on the recorded path.
#define LIGHTAPE_EIGEN_REV_SCALAR_INT_TRAITS(T)                                                    \
    template <typename BinaryOp>                                                                   \
    struct ScalarBinaryOpTraits<lightape::RevScalar, T, BinaryOp> {                                \
        using ReturnType = lightape::RevScalar;                                                    \
    };                                                                                             \
    template <typename BinaryOp>                                                                   \
    struct ScalarBinaryOpTraits<T, lightape::RevScalar, BinaryOp> {                                \
        using ReturnType = lightape::RevScalar;                                                    \
    };

LIGHTAPE_EIGEN_REV_SCALAR_INT_TRAITS(int)
LIGHTAPE_EIGEN_REV_SCALAR_INT_TRAITS(unsigned int)
LIGHTAPE_EIGEN_REV_SCALAR_INT_TRAITS(long)
LIGHTAPE_EIGEN_REV_SCALAR_INT_TRAITS(unsigned long)
LIGHTAPE_EIGEN_REV_SCALAR_INT_TRAITS(long long)
LIGHTAPE_EIGEN_REV_SCALAR_INT_TRAITS(unsigned long long)

#undef LIGHTAPE_EIGEN_REV_SCALAR_INT_TRAITS
} // namespace Eigen

#endif // LIGHTAPE_EIGEN_LEAN_REVERSE_EIGEN_H
