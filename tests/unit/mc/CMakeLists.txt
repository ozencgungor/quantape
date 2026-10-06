# mc/: SDE engine, stochastic sources and the Sobol quality generators. Link
# profiles are per-file per legacy: TBB, TBB + Stan defs, full Stan
# (test_sde_schemes) and log/format only (test_sobol_quality).

add_executable(test_sde_simulator ${CMAKE_SOURCE_DIR}/tests/test_sde_simulator.cpp)
target_link_libraries(test_sde_simulator PRIVATE quantape TBB::tbb)
quantape_set_warnings(test_sde_simulator)
quantape_register_test(test_sde_simulator)

add_executable(test_sde_schemes ${CMAKE_SOURCE_DIR}/tests/test_sde_schemes.cpp)
target_link_libraries(test_sde_schemes PRIVATE quantape)
quantape_enable_stan(test_sde_schemes)
quantape_set_warnings(test_sde_schemes)
quantape_register_test(test_sde_schemes)

foreach(_test test_sde_moment_matching test_sde_processes)
    add_executable(${_test} ${CMAKE_SOURCE_DIR}/tests/${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape TBB::tbb)
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

foreach(_test test_sde_gradients test_sde_qmc)
    add_executable(${_test} ${CMAKE_SOURCE_DIR}/tests/${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape TBB::tbb)
    quantape_enable_stan_defs(${_test})
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

add_executable(test_sobol_generator ${CMAKE_SOURCE_DIR}/tests/test_sobol_generator.cpp)
target_link_libraries(test_sobol_generator PRIVATE quantape TBB::tbb)
quantape_register_test(test_sobol_generator)

add_executable(test_sobol_quality ${CMAKE_SOURCE_DIR}/tests/test_sobol_quality.cpp)
target_link_libraries(test_sobol_quality PRIVATE Threads::Threads quantape_log quantape_format)
quantape_register_test(test_sobol_quality)

add_executable(test_heston_analytic ${CMAKE_SOURCE_DIR}/tests/test_heston_analytic.cpp)
target_link_libraries(test_heston_analytic PRIVATE quantape TBB::tbb)
quantape_set_warnings(test_heston_analytic)
quantape_register_test(test_heston_analytic)

quantape_collect_executables()
