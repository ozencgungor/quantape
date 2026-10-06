# math/ad: full-Stan AD math kernels (interpolation, integration, solvers,
# cubic/bicubic weight dispatch).
quantape_add_gtest(qta_test_math_ad
    SOURCES test_math_utils.cpp
            test_ad.cpp
            test_autodiff_primitives.cpp
            test_interpolation_xad.cpp
            test_solvers_ad.cpp
            test_tanh_sinh.cpp
            test_integrator_primitives.cpp
            test_cubic_weights.cpp
            test_bicubic_weights.cpp
    STAN
    LABELS unit numerics ad slow)

quantape_collect_executables()
