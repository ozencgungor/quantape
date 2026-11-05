// Compile-coverage translation unit: includes every public lightape header so
// the full header set is instantiated in one TU (and picked up by the
// clang-tidy gate). The test case is intentionally trivial.
#include "lightape/base/BesselI.h"
#include "lightape/base/BesselJY.h"
#include "lightape/base/BesselK.h"
#include "lightape/base/BivariateNormal.h"
#include "lightape/base/Digamma.h"
#include "lightape/base/IncompleteBeta.h"
#include "lightape/base/IncompleteGamma.h"
#include "lightape/base/NoncentralChiSquare.h"
#include "lightape/base/NormalQuantile.h"
#include "lightape/base/ScalarConstant.h"
#include "lightape/batch/BatchFunctions.h"
#include "lightape/batch/BatchTape.h"
#include "lightape/batch/VectorMath.h"
#include "lightape/exec/Executor.h"
#include "lightape/fwd/ForwardScalar.h"
#include "lightape/mix/FwdRev.h"
#include "lightape/mix/ThirdOrder.h"
#include "lightape/ops/Bessel.h"
#include "lightape/ops/BivariateNormal.h"
#include "lightape/ops/Callbacks.h"
#include "lightape/ops/DerivativeCache.h"
#include "lightape/ops/ElementaryExtras.h"
#include "lightape/ops/Generic.h"
#include "lightape/ops/NoncentralChiSquare.h"
#include "lightape/ops/P1Functions.h"
#include "lightape/ops/P2BesselJY.h"
#include "lightape/ops/P2Functions.h"
#include "lightape/ops/P2Quantiles.h"
#include "lightape/ops/Probability.h"
#include "lightape/rev/Checkpoint.h"
#include "lightape/rev/LeanReverse.h"
#include "lightape/rev/RevScalarN.h"
#include "lightape/rev/RevTapeT.h"
#include "lightape/rev/Revolve.h"
#include "lightape/rev/SecondOrder.h"
#include "lightape/rev/Sparsity.h"

// Umbrella last: it must remain a valid closure of the headers above.
#include "lightape/lightape.h"

#if defined(LIGHTAPE_TEST_ALL_HEADERS_EIGEN)
#include <Eigen/Core>

#include "lightape/eigen/ForwardScalarEigen.h"
#include "lightape/eigen/LeanReverseEigen.h"
#include "lightape/eigen/MatrixBlocks.h"
#include "lightape/eigen/Select.h"
#endif

#include <gtest/gtest.h>

TEST(AllHeaders, Instantiates) {
    SUCCEED();
}
