# models/: Heston pricer, calibration, IFT and Monte Carlo gates. The analytic
# file is double-only (TBB); the four AD files keep the legacy STAN_DEFS + TBB
# profile.

quantape_add_gtest(qta_test_models
    SOURCES test_heston_analytic.cpp
    LIBS TBB::tbb
    LABELS unit models mc slow)

quantape_add_gtest(qta_test_models_ad
    SOURCES test_heston_stan.cpp
            test_heston_calibration.cpp
            test_heston_mc.cpp
            test_heston_ift.cpp
    STAN_DEFS
    LIBS TBB::tbb
    LABELS unit models mc ad slow)

quantape_collect_executables()
