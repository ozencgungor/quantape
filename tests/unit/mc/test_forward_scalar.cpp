// test_forward_scalar.cpp — forward-mode Tangent arithmetic consistency.
//
// `Tangent::operator/=` must agree with the binary `/`: both compute the
// quotient tangent (a.d - (a/b) * b.d) / b.value, so the denominator tangent
// contributes. The checks below compare the two forms for plain doubles and
// for Stan `var` scalars.

#include "quantape/math/StanMath.h" // must be first (Eigen plugin ordering)

#include "quantape/mc/mcfwdrev/ForwardScalar.h"

#include <array>
#include <cstddef>
#include <type_traits>

#include "support/GtestSupport.h"
#include "support/StanTapeFixture.h"

using quantape::mc::Tangent;
using stan::math::var;

namespace {

/// a = (6, d={1,1}), b = (3, d={1,1}): the quotient tangent is
/// (1 - 2*1)/3 = -1/3.
template <typename Scalar>
double valueOf(const Scalar& x) {
    if constexpr (std::is_same_v<Scalar, double>) {
        return x;
    } else {
        return x.val();
    }
}

template <typename Scalar>
void checkQuotientAssignment() {
    const Tangent<Scalar, 2> a(Scalar(6.0), std::array<Scalar, 2>{Scalar(1.0), Scalar(1.0)});
    const Tangent<Scalar, 2> b(Scalar(3.0), std::array<Scalar, 2>{Scalar(1.0), Scalar(1.0)});
    const Tangent<Scalar, 2> binary = a / b;
    Tangent<Scalar, 2> compound = a;
    compound /= b;
    CHECK_CLOSE("quotient value", valueOf(compound.value), valueOf(binary.value), 1e-15);
    for (std::size_t i = 0; i < 2; ++i) {
        CHECK_CLOSE("quotient tangent matches binary operator", valueOf(compound.d[i]),
                    valueOf(binary.d[i]), 1e-15);
        CHECK_CLOSE("quotient tangent is -1/3", valueOf(compound.d[i]), -1.0 / 3.0, 1e-15);
    }
}

} // namespace

TEST(ForwardScalarTest, quotientAssignmentMatchesBinaryOperator) {
    checkQuotientAssignment<double>();
}

class ForwardScalarVarTest : public StanTapeTest {};

TEST_F(ForwardScalarVarTest, quotientAssignmentMatchesBinaryOperatorVar) {
    checkQuotientAssignment<var>();
}
