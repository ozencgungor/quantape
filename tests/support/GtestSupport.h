#ifndef QUANTAPE_TESTS_SUPPORT_GTEST_SUPPORT_H
#define QUANTAPE_TESTS_SUPPORT_GTEST_SUPPORT_H

// Umbrella for plain test translation units: gtest + assertion macros +
// printers. gtest/gmock may only enter the suite through this header (or
// "support/StanTapeFixture.h", which pulls Stan).
//
// Include order:
//   * plain TUs: GtestSupport.h first, then library headers, then std;
//   * AD TUs: "quantape/math/StanMath.h" first, then library headers, then
//     GtestSupport.h (+ StanTapeFixture.h when the tape fixture is used).
//
// StanTapeFixture.h is deliberately NOT included here: it pulls Stan Math and
// must not leak into every plain TU.

#include "support/Assertions.h"
#include "support/Printers.h"
#include <gtest/gtest.h>

#endif // QUANTAPE_TESTS_SUPPORT_GTEST_SUPPORT_H
