# math/: plain kernels (legacy link profiles kept until each file converts
# to googletest). AD kernels live in math/ad/, Hessian/calibration gates in
# calibration/.

foreach(_test test_math test_solvers test_interpolation)
    add_executable(${_test} ${CMAKE_SOURCE_DIR}/tests/${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape)
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

quantape_collect_executables()
