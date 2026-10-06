#ifndef QUANTAPE_TESTS_SUPPORT_SCALAR_TRAITS_H
#define QUANTAPE_TESTS_SUPPORT_SCALAR_TRAITS_H

// ScalarTraits<T>: uniform value()/adjoint() access for the scalar types used
// by the engine (arithmetic, Stan var/fvar, and the in-house mcfwdrev lean
// scalars). Used by TYPED_TEST suites that run one generic body over several
// scalar types.
//
// ScalarName is the matching identifier-safe type-name generator for
// TYPED_TEST_SUITE; Stan specializations (and StanScalarTypes) exist only when
// QTA_TEST_STAN is defined by quantape_add_gtest for STAN/STAN_DEFS targets.
//
// Include order: StanMath.h must be the first include in any AD translation
// unit; this header re-includes it under QTA_TEST_STAN after that.

#include "quantape/mc/mcfwdrev/ForwardScalar.h"
#include "quantape/mc/mcfwdrev/LeanReverse.h"

#include <string>

#include <gtest/gtest.h>

#ifdef QTA_TEST_STAN
#include "quantape/math/StanMath.h"
#endif

/// value() extracts the primal double; adjoint() extracts the first-order
/// sensitivity (0 for plain arithmetic types).
template <class T>
struct ScalarTraits {
    static double value(const T& x) { return static_cast<double>(x); }
    static double adjoint(const T&) { return 0.0; }
};

#ifdef QTA_TEST_STAN
template <>
struct ScalarTraits<stan::math::var> {
    static double value(const stan::math::var& x) { return x.val(); }
    static double adjoint(const stan::math::var& x) { return x.adj(); }
};

template <class T>
struct ScalarTraits<stan::math::fvar<T>> {
    static double value(const stan::math::fvar<T>& x) { return ScalarTraits<T>::value(x.val_); }
    static double adjoint(const stan::math::fvar<T>& x) { return ScalarTraits<T>::value(x.d_); }
};

/// Stan chain types for `TYPED_TEST_SUITE` in the calibration/AD files.
using StanScalarTypes = ::testing::Types<stan::math::var, stan::math::fvar<stan::math::var>>;
#endif

template <>
struct ScalarTraits<quantape::mc::Tangent<double, 1>> {
    static double value(const quantape::mc::Tangent<double, 1>& x) { return x.value; }
    static double adjoint(const quantape::mc::Tangent<double, 1>& x) { return x.tangent(0); }
};

/// The reverse adjoint is read from the thread-local tape after the case has
/// run `RevTape::active().reverse(root)`; tape-free constants have no node.
template <>
struct ScalarTraits<quantape::mc::RevScalar> {
    static double value(const quantape::mc::RevScalar& x) { return x.value; }
    static double adjoint(const quantape::mc::RevScalar& x) {
        return x.node == quantape::mc::RevTape::kNone
                   ? 0.0
                   : quantape::mc::RevTape::active().adjoint(x.node);
    }
};

namespace quantape::tests::detail {

template <class T>
struct ScalarTypeName;

template <>
struct ScalarTypeName<double> {
    static std::string get() { return "Double"; }
};

#ifdef QTA_TEST_STAN
template <>
struct ScalarTypeName<stan::math::var> {
    static std::string get() { return "Var"; }
};

template <>
struct ScalarTypeName<stan::math::fvar<stan::math::var>> {
    static std::string get() { return "FvarVar"; }
};

template <>
struct ScalarTypeName<stan::math::fvar<double>> {
    static std::string get() { return "FvarDouble"; }
};
#endif

template <>
struct ScalarTypeName<quantape::mc::Tangent<double, 1>> {
    static std::string get() { return "Tangent1"; }
};

template <>
struct ScalarTypeName<quantape::mc::RevScalar> {
    static std::string get() { return "RevScalar"; }
};

} // namespace quantape::tests::detail

/// Identifier-safe, unique typed-test name: double -> "Double", var -> "Var",
/// fvar<var> -> "FvarVar", Tangent<double,1> -> "Tangent1",
/// RevScalar -> "RevScalar".
struct ScalarName {
    template <class T>
    static std::string GetName(int) {
        return quantape::tests::detail::ScalarTypeName<T>::get();
    }
};

#endif // QUANTAPE_TESTS_SUPPORT_SCALAR_TRAITS_H
