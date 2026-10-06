# math/ad: full-Stan AD math kernels (legacy link profiles kept until each
# file converts to googletest).

add_executable(test_math_utils ${CMAKE_SOURCE_DIR}/tests/test_math_utils.cpp)
target_link_libraries(test_math_utils PRIVATE quantape)
quantape_enable_stan(test_math_utils)
quantape_set_warnings(test_math_utils)
quantape_register_test(test_math_utils)

add_executable(test_ad ${CMAKE_SOURCE_DIR}/tests/test_ad.cpp)
target_link_libraries(test_ad PRIVATE quantape)
quantape_enable_stan(test_ad)
quantape_set_warnings(test_ad)
quantape_register_test(test_ad)

foreach(_test test_cubic_weights test_bicubic_weights test_tanh_sinh
               test_integrator_primitives test_solvers_ad
               test_interpolation_xad test_autodiff_primitives)
    add_executable(${_test} ${CMAKE_SOURCE_DIR}/tests/${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape_log quantape_format)
    quantape_enable_stan(${_test})
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

quantape_collect_executables()
