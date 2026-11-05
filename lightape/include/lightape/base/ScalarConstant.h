#ifndef LIGHTAPE_BASE_SCALAR_CONSTANT_H
#define LIGHTAPE_BASE_SCALAR_CONSTANT_H

#include <type_traits>
#include <utility>

namespace lightape::detail {

/// Constant of scalar type `T` from a plain double. A plain double cannot
/// reach the scalar of a nested `Tangent` in one implicit conversion (chained
/// user-defined conversions do not compose), so constants are built one
/// nesting level at a time through the `value` member. `RevScalar` constants
/// stay node-less; `Tangent` constants carry zero tangents.
template <typename T>
T constantOf(double v) {
    if constexpr (requires { std::declval<T>().value; }) {
        using Inner = std::remove_cvref_t<decltype(std::declval<T>().value)>;
        return T(constantOf<Inner>(v));
    } else {
        return T(v);
    }
}

} // namespace lightape::detail

#endif // LIGHTAPE_BASE_SCALAR_CONSTANT_H
