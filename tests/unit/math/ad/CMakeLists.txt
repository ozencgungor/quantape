# math/ad: full-Stan AD math kernels. Converted files form qta_test_math_ad;
# the legacy stanzas below stay until their file converts.

quantape_add_gtest(qta_test_math_ad
    SOURCES test_math_utils.cpp
            test_ad.cpp
            test_autodiff_primitives.cpp
    STAN
    LABELS unit numerics ad slow)

foreach(_test test_cubic_weights test_bicubic_weights test_tanh_sinh
               test_integrator_primitives test_solvers_ad
               test_interpolation_xad)
    add_executable(${_test} ${CMAKE_SOURCE_DIR}/tests/${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape_log quantape_format)
    quantape_enable_stan(${_test})
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

quantape_collect_executables()
