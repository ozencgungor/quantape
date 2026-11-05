#ifndef LIGHTAPE_EIGEN_FORWARD_SCALAR_EIGEN_H
#define LIGHTAPE_EIGEN_FORWARD_SCALAR_EIGEN_H

#include <Eigen/Core>

#include <type_traits>

#include "lightape/fwd/ForwardScalar.h"

namespace lightape {

// Eigen's `internal::pow_impl::run` dispatches with an unqualified `pow(x, y)`
// (ADL), and its generic `pow(const T&, const S&)` overload cannot deduce `S`
// from an arithmetic exponent: `pow(Tangent<double, N>, int)` sees `double`
// (from the base) and `int` (from the exponent) and fails. These overloads
// keep integer exponents on the AD path (`numext::pow(x, 2)`,
// `array().pow(2)`); the scalar-exponent (`S`) case is already covered by
// `ForwardScalar.h`. Excluded when `T == S` to leave the core overloads
// unambiguous.
template <typename S, std::size_t N, typename T,
          std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, S>, int> = 0>
Tangent<S, N> pow(const Tangent<S, N>& a, const T& b) {
    return pow(a, static_cast<S>(b));
}

template <typename S, std::size_t N, typename T,
          std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, S>, int> = 0>
Tangent<S, N> pow(const T& a, const Tangent<S, N>& b) {
    return pow(static_cast<S>(a), b);
}

} // namespace lightape

namespace Eigen {
template <typename S, std::size_t N>
struct NumTraits<lightape::Tangent<S, N>> : NumTraits<double> {
    using Real = lightape::Tangent<S, N>;
    using NonInteger = lightape::Tangent<S, N>;
    using Nested = lightape::Tangent<S, N>;
    using Literal = lightape::Tangent<S, N>;
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

// Mixed Tangent<S, N>/S expressions keep the dual part: the plain scalar is
// promoted to a zero-tangent Tangent. Same-type combinations are already
// handled by Eigen's generic ScalarBinaryOpTraits<T, T, Op>.
template <typename S, std::size_t N, typename BinaryOp>
struct ScalarBinaryOpTraits<lightape::Tangent<S, N>, S, BinaryOp> {
    using ReturnType = lightape::Tangent<S, N>;
};

template <typename S, std::size_t N, typename BinaryOp>
struct ScalarBinaryOpTraits<S, lightape::Tangent<S, N>, BinaryOp> {
    using ReturnType = lightape::Tangent<S, N>;
};

// Integer operands promote to Tangent as well. `numext::pow(x, 2)` asks for
// `ScalarBinaryOpTraits<Tangent<S, N>, T, scalar_pow_op<...>>` while resolving
// `internal::pow_impl::result_type`; without these, that eager lookup is a
// hard error (not SFINAE), even when a better overload exists. The same traits
// keep mixed integer arithmetic and `array().pow(2)` on the AD path.
#define LIGHTAPE_EIGEN_TANGENT_INT_TRAITS(T)                                                       \
    template <typename S, std::size_t N, typename BinaryOp>                                        \
    struct ScalarBinaryOpTraits<lightape::Tangent<S, N>, T, BinaryOp> {                            \
        using ReturnType = lightape::Tangent<S, N>;                                                \
    };                                                                                             \
    template <typename S, std::size_t N, typename BinaryOp>                                        \
    struct ScalarBinaryOpTraits<T, lightape::Tangent<S, N>, BinaryOp> {                            \
        using ReturnType = lightape::Tangent<S, N>;                                                \
    };

LIGHTAPE_EIGEN_TANGENT_INT_TRAITS(int)
LIGHTAPE_EIGEN_TANGENT_INT_TRAITS(unsigned int)
LIGHTAPE_EIGEN_TANGENT_INT_TRAITS(long)
LIGHTAPE_EIGEN_TANGENT_INT_TRAITS(unsigned long)
LIGHTAPE_EIGEN_TANGENT_INT_TRAITS(long long)
LIGHTAPE_EIGEN_TANGENT_INT_TRAITS(unsigned long long)

#undef LIGHTAPE_EIGEN_TANGENT_INT_TRAITS
} // namespace Eigen

#endif // LIGHTAPE_EIGEN_FORWARD_SCALAR_EIGEN_H
