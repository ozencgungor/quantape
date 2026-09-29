# ============================================================================
# Test executables (registered with CTest, runnable directly as well).
# ============================================================================

# --- Sobol -------------------------------------------------------------------
add_executable(test_sobol_quality test_sobol_quality.cpp)
target_link_libraries(test_sobol_quality PRIVATE Threads::Threads quantape_log quantape_format)
quantape_register_test(test_sobol_quality)

add_executable(test_sobol_generator test_sobol_generator.cpp)
target_link_libraries(test_sobol_generator PRIVATE quantape TBB::tbb)
quantape_register_test(test_sobol_generator)

add_executable(test_sobol_tools test_sobol_tools.cpp)
target_link_libraries(test_sobol_tools PRIVATE Threads::Threads quantape_log quantape_format)
target_compile_definitions(test_sobol_tools PRIVATE
    EXTEND_BIN="$<TARGET_FILE:extend_sobol>"
    REFINE_BIN="$<TARGET_FILE:refine_sobol>"
    VERIFY_BIN="$<TARGET_FILE:verify_sobol>")
add_dependencies(test_sobol_tools extend_sobol refine_sobol verify_sobol)
quantape_register_test(test_sobol_tools)

# --- Infrastructure ----------------------------------------------------------
add_executable(test_logging test_logging.cpp)
target_link_libraries(test_logging PRIVATE quantape)
# The facade test must see every level at compile time; -U overrides the
# project compile-time level (Release strips debug/trace by default).
target_compile_options(test_logging PRIVATE
    -UQUILL_COMPILE_ACTIVE_LOG_LEVEL
    -DQUILL_COMPILE_ACTIVE_LOG_LEVEL=QUILL_COMPILE_ACTIVE_LOG_LEVEL_TRACE_L3)
quantape_set_warnings(test_logging)
quantape_register_test(test_logging)

add_executable(test_number_format test_number_format.cpp)
target_link_libraries(test_number_format PRIVATE quantape)
quantape_set_warnings(test_number_format)
quantape_register_test(test_number_format)

add_executable(test_util test_util.cpp)
target_link_libraries(test_util PRIVATE quantape)
quantape_set_warnings(test_util)
quantape_register_test(test_util)

# --- Time --------------------------------------------------------------------
foreach(_test test_date test_daycount test_calendar test_schedule)
    add_executable(${_test} ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape_log quantape_format)
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

# --- Core library ------------------------------------------------------------
foreach(_test test_math test_solvers test_interpolation test_markets)
    add_executable(${_test} ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape)
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

# --- AD and Stan-backed ------------------------------------------------------
add_executable(test_ad test_ad.cpp)
target_link_libraries(test_ad PRIVATE quantape)
quantape_enable_stan(test_ad)
quantape_set_warnings(test_ad)
quantape_register_test(test_ad)

add_executable(test_black76 test_black76.cpp)
target_link_libraries(test_black76 PRIVATE quantape)
quantape_enable_stan(test_black76)
quantape_set_warnings(test_black76)
quantape_register_test(test_black76)

foreach(_test test_cubic_weights test_bicubic_weights test_tanh_sinh
               test_integrator_primitives test_solvers_ad
               test_interpolation_xad test_autodiff_primitives
               test_optimization)
    add_executable(${_test} ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape_log quantape_format)
    quantape_enable_stan(${_test})
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

add_executable(test_ift test_ift.cpp)
target_link_libraries(test_ift PRIVATE quantape_log quantape_format)
quantape_enable_stan(test_ift)
quantape_set_warnings(test_ift)
quantape_register_test(test_ift)

add_executable(test_market_risk_ift test_market_risk_ift.cpp)
target_link_libraries(test_market_risk_ift PRIVATE quantape_log quantape_format)
quantape_enable_stan(test_market_risk_ift)
quantape_set_warnings(test_market_risk_ift)
quantape_register_test(test_market_risk_ift)

add_executable(test_optimizers_stress test_optimizers_stress.cpp)
target_link_libraries(test_optimizers_stress PRIVATE quantape_log quantape_format)
quantape_enable_stan(test_optimizers_stress)
quantape_set_warnings(test_optimizers_stress)
quantape_register_test(test_optimizers_stress)

foreach(_test test_mixed_hessian test_pricer_hessian)
    add_executable(${_test} ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape)
    quantape_enable_stan(${_test})
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

add_executable(test_precision test_precision.cpp)
target_link_libraries(test_precision PRIVATE quantape_log quantape_format)
quantape_set_warnings(test_precision)
quantape_register_test(test_precision)

foreach(_test test_ziggurat test_mcfarland)
    add_executable(${_test} ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape)
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

# --- SDE engine --------------------------------------------------------------
add_executable(test_sde_simulator test_sde_simulator.cpp)
target_link_libraries(test_sde_simulator PRIVATE quantape TBB::tbb)
quantape_set_warnings(test_sde_simulator)
quantape_register_test(test_sde_simulator)

add_executable(test_sde_schemes test_sde_schemes.cpp)
target_link_libraries(test_sde_schemes PRIVATE quantape)
quantape_enable_stan(test_sde_schemes)
quantape_set_warnings(test_sde_schemes)
quantape_register_test(test_sde_schemes)

foreach(_test test_sde_moment_matching test_sde_processes)
    add_executable(${_test} ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape TBB::tbb)
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

foreach(_test test_sde_gradients test_sde_qmc)
    add_executable(${_test} ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape TBB::tbb)
    quantape_enable_stan_defs(${_test})
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

# --- Heston ------------------------------------------------------------------
add_executable(test_heston_analytic test_heston_analytic.cpp)
target_link_libraries(test_heston_analytic PRIVATE quantape TBB::tbb)
quantape_set_warnings(test_heston_analytic)
quantape_register_test(test_heston_analytic)

foreach(_test test_heston_stan test_heston_calibration test_heston_mc
               test_heston_ift)
    add_executable(${_test} ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape TBB::tbb)
    quantape_enable_stan_defs(${_test})
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

quantape_collect_executables()
