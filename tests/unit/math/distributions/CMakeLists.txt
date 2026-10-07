# math/distributions: plain double kernels (fast tier) plus a Stan AD
# conformance target exercising the generic DoubleT scalar path. The module is
# header-only; everything is discovered through the library include dir.

quantape_add_gtest(qta_test_distributions
    SOURCES test_normal.cpp
            test_gamma.cpp
            test_chi_square_poisson.cpp
            test_beta.cpp
            test_student_t.cpp
            test_lognormal_exponential.cpp
            test_inverse_gaussian.cpp
            test_bivariate_normal.cpp
            test_noncentral_beta.cpp
            test_noncentral_chi_square.cpp
            test_bessel.cpp
            test_trivariate_normal.cpp
            test_quantile_derivatives.cpp
    LABELS unit numerics fast)

quantape_add_gtest(qta_test_distributions_ad
    SOURCES test_normal_stan.cpp
    STAN
    LABELS unit numerics ad slow
    TIMEOUT 900)

quantape_collect_executables()
