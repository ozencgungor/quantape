# math/: plain kernels plus the full-Stan AD files (legacy link profiles are
# kept until each file converts to googletest).

foreach(_test test_math test_solvers test_interpolation)
    add_executable(${_test} ${CMAKE_SOURCE_DIR}/tests/${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape)
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

# test_math_utils instantiates Stan scalars (var / `fvar<var>`) for the
# interpolation and tridiagonal-solver paths.
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

add_executable(test_precision ${CMAKE_SOURCE_DIR}/tests/test_precision.cpp)
target_link_libraries(test_precision PRIVATE quantape_log quantape_format)
quantape_set_warnings(test_precision)
quantape_register_test(test_precision)

foreach(_test test_ziggurat test_mcfarland)
    add_executable(${_test} ${CMAKE_SOURCE_DIR}/tests/${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape)
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

foreach(_test test_mixed_hessian test_pricer_hessian)
    add_executable(${_test} ${CMAKE_SOURCE_DIR}/tests/${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape)
    quantape_enable_stan(${_test})
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

quantape_collect_executables()
