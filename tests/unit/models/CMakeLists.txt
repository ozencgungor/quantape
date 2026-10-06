# models/: Heston calibration/AD/MC. All four targets keep the legacy
# Stan-defs + TBB profile.

foreach(_test test_heston_stan test_heston_calibration test_heston_mc
               test_heston_ift)
    add_executable(${_test} ${CMAKE_SOURCE_DIR}/tests/${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape TBB::tbb)
    quantape_enable_stan_defs(${_test})
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

quantape_collect_executables()
