#ifndef QUANTAPE_TESTS_SUPPORT_STAN_TAPE_FIXTURE_H
#define QUANTAPE_TESTS_SUPPORT_STAN_TAPE_FIXTURE_H

// Every Stan AD test case derives from StanTapeTest: SetUp() starts each case
// on a clean tape, TearDown() recovers memory after EXPECT/ASSERT failures and
// after ordinary completion. Stan var is arena-allocated and its destructor is
// trivial, so gtest's stack unwinding is safe.

#include "quantape/math/StanMath.h"

#include <gtest/gtest.h>

class StanTapeTest : public ::testing::Test {
protected:
    void SetUp() override {
        stan::math::recover_memory();
        stan::math::set_zero_all_adjoints();
    }

    void TearDown() override {
        stan::math::set_zero_all_adjoints();
        stan::math::recover_memory();
    }
};

#endif // QUANTAPE_TESTS_SUPPORT_STAN_TAPE_FIXTURE_H
