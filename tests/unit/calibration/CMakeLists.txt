# calibration/: AD optimizers, IFT and Hessian gates (full-Stan link profile).

quantape_add_gtest(qta_test_optimization_ad
    SOURCES
        test_levenberg_marquardt_ad.cpp
        test_optimization.cpp
        test_ift.cpp
        test_market_risk_ift.cpp
        test_mixed_hessian.cpp
        test_pricer_hessian.cpp
    STAN
    LABELS unit calibration ad slow)

quantape_add_gtest(qta_test_optimizers_stress
    SOURCES test_optimizers_stress.cpp
    STAN
    LABELS unit calibration ad slow stress)

quantape_collect_executables()
