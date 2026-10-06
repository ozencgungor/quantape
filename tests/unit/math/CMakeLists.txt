# math/: plain kernels (googletest). AD kernels live in math/ad/, Hessian
# gates in calibration/. test_mcfarland is split into its own slow target:
# its 1e9/1e10-sample loops take ~9 min in Debug, so they must stay out of the
# fast tier.

quantape_add_gtest(qta_test_math
    SOURCES test_math.cpp
            test_solvers.cpp
            test_interpolation.cpp
            test_precision.cpp
            test_ziggurat.cpp
    LABELS unit numerics fast smoke)

quantape_add_gtest(qta_test_math_slow
    SOURCES test_mcfarland.cpp
    LABELS unit numerics slow
    TIMEOUT 1800)

quantape_collect_executables()
