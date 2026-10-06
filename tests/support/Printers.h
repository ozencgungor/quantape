#ifndef QUANTAPE_TESTS_SUPPORT_PRINTERS_H
#define QUANTAPE_TESTS_SUPPORT_PRINTERS_H

// gtest PrintTo overloads for the scalar types used across the suite.
//
// They live in the global namespace on purpose: gtest's unqualified PrintTo
// lookup walks testing::internal -> testing -> global, so these overloads are
// found without any gtest/gmock include here (the umbrella includes
// <gtest/gtest.h> first).
//
// Stan overloads are gated on QTA_TEST_STAN (defined by quantape_add_gtest for
// STAN/STAN_DEFS targets). AD translation units must include
// "quantape/math/StanMath.h" before anything that pulls Eigen, including this
// header.

#include "quantape/math/Precision/DoubleDouble.h"
#include "quantape/util/Check.h"

#include <Eigen/Dense>

#include <ostream>

#ifdef QTA_TEST_STAN
#include "quantape/math/StanMath.h"
#endif

inline void PrintTo(const quantape::math::DoubleDouble& value, std::ostream* os) {
    *os << quantape::util::num(value.value(), 20);
}

inline void PrintTo(const Eigen::VectorXd& value, std::ostream* os) {
    *os << "[";
    for (Eigen::Index i = 0; i < value.size(); ++i) {
        if (i != 0) {
            *os << ", ";
        }
        *os << quantape::util::num(value(i));
    }
    *os << "]";
}

inline void PrintTo(const Eigen::MatrixXd& value, std::ostream* os) {
    *os << "[";
    for (Eigen::Index r = 0; r < value.rows(); ++r) {
        if (r != 0) {
            *os << "; ";
        }
        for (Eigen::Index c = 0; c < value.cols(); ++c) {
            if (c != 0) {
                *os << ", ";
            }
            *os << quantape::util::num(value(r, c));
        }
    }
    *os << "]";
}

#ifdef QTA_TEST_STAN
inline void PrintTo(const stan::math::var& value, std::ostream* os) {
    *os << "var(value=" << quantape::util::num(value.val())
        << ", adj=" << quantape::util::num(value.adj()) << ")";
}

template <class T>
inline void PrintTo(const stan::math::fvar<T>& value, std::ostream* os) {
    *os << "fvar(val_=" << value.val_ << ", d_=" << value.d_ << ")";
}
#endif

#endif // QUANTAPE_TESTS_SUPPORT_PRINTERS_H
