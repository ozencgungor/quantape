# mc/: SDE engine, stochastic sources and the Sobol quality generators. Link
# profiles are per-file per legacy: TBB, TBB + Stan defs, full Stan
# (test_sde_schemes) and log/format only (test_sobol_quality).

quantape_add_gtest(qta_test_mc
    SOURCES test_sde_simulator.cpp
            test_sde_processes.cpp
            test_sde_moment_matching.cpp
            test_sobol_generator.cpp
            test_sobol_quality.cpp
    LIBS TBB::tbb Threads::Threads
    LABELS unit mc slow
    TIMEOUT 1800)

add_executable(test_sde_schemes ${CMAKE_SOURCE_DIR}/tests/test_sde_schemes.cpp)
target_link_libraries(test_sde_schemes PRIVATE quantape)
quantape_enable_stan(test_sde_schemes)
quantape_set_warnings(test_sde_schemes)
quantape_register_test(test_sde_schemes)

foreach(_test test_sde_gradients test_sde_qmc)
    add_executable(${_test} ${CMAKE_SOURCE_DIR}/tests/${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape TBB::tbb)
    quantape_enable_stan_defs(${_test})
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

quantape_collect_executables()
