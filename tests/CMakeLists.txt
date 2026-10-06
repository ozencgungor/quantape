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

if(TARGET extend_sobol AND TARGET refine_sobol AND TARGET verify_sobol)
    add_executable(test_sobol_tools test_sobol_tools.cpp)
    target_link_libraries(test_sobol_tools PRIVATE Threads::Threads quantape_log quantape_format)
    target_compile_definitions(test_sobol_tools PRIVATE
        EXTEND_BIN="$<TARGET_FILE:extend_sobol>"
        REFINE_BIN="$<TARGET_FILE:refine_sobol>"
        VERIFY_BIN="$<TARGET_FILE:verify_sobol>")
    add_dependencies(test_sobol_tools extend_sobol refine_sobol verify_sobol)
    quantape_register_test(test_sobol_tools)
else()
    message(STATUS "test_sobol_tools skipped: QUANTAPE_BUILD_TOOLS=OFF")
endif()

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

# --- Curves ------------------------------------------------------------------
add_executable(test_curve test_curve.cpp)
target_link_libraries(test_curve PRIVATE quantape)
quantape_set_warnings(test_curve)
quantape_register_test(test_curve)

add_executable(test_curve_config test_curve_config.cpp)
target_link_libraries(test_curve_config PRIVATE quantape)
target_compile_definitions(test_curve_config PRIVATE
    CURVE_CONFIG_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/data/usd_ois_config.json"
    FX_CONFIG_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/data/fx_config.json")
quantape_set_warnings(test_curve_config)
quantape_register_test(test_curve_config)

add_executable(test_curve_risk test_curve_risk.cpp)
target_link_libraries(test_curve_risk PRIVATE quantape)
quantape_set_warnings(test_curve_risk)
quantape_register_test(test_curve_risk)

add_executable(test_xccy test_xccy.cpp)
target_link_libraries(test_xccy PRIVATE quantape)
quantape_set_warnings(test_xccy)
quantape_register_test(test_xccy)

add_executable(test_instrument_options test_instrument_options.cpp)
target_link_libraries(test_instrument_options PRIVATE quantape)
quantape_set_warnings(test_instrument_options)
quantape_register_test(test_instrument_options)

add_executable(test_forecast_pillars test_forecast_pillars.cpp)
target_link_libraries(test_forecast_pillars PRIVATE quantape)
quantape_set_warnings(test_forecast_pillars)
quantape_register_test(test_forecast_pillars)

add_executable(test_synthetic_quotes test_synthetic_quotes.cpp)
target_link_libraries(test_synthetic_quotes PRIVATE quantape)
quantape_set_warnings(test_synthetic_quotes)
quantape_register_test(test_synthetic_quotes)

add_executable(test_futures test_futures.cpp)
target_link_libraries(test_futures PRIVATE quantape)
quantape_set_warnings(test_futures)
quantape_register_test(test_futures)

add_executable(test_bootstrap_validation test_bootstrap_validation.cpp)
target_link_libraries(test_bootstrap_validation PRIVATE quantape)
quantape_set_warnings(test_bootstrap_validation)
quantape_register_test(test_bootstrap_validation)

# Instruments vocabulary: Cashflow layout, the BootstrapInstrument concept
# (double/var/fvar<var>, including a std::variant Ladder) and the bitwise
# old-vs-new discount-bootstrap gate; the AD gates instantiate Stan scalars.
add_executable(test_instruments test_instruments.cpp)
target_link_libraries(test_instruments PRIVATE quantape)
quantape_enable_stan(test_instruments)
quantape_set_warnings(test_instruments)
quantape_register_test(test_instruments)

# Money-market instruments: factory validation, date/target/kindName/priority
# metadata, the overlap ordering and the var/fvar<var> quote gates.
add_executable(test_ir_instruments test_ir_instruments.cpp)
target_link_libraries(test_ir_instruments PRIVATE quantape)
quantape_enable_stan(test_ir_instruments)
quantape_set_warnings(test_ir_instruments)
quantape_register_test(test_ir_instruments)

add_executable(test_curve_risk_report test_curve_risk_report.cpp)
target_link_libraries(test_curve_risk_report PRIVATE quantape)
quantape_set_warnings(test_curve_risk_report)
quantape_register_test(test_curve_risk_report)

add_executable(test_hull_white_convexity test_hull_white_convexity.cpp)
target_link_libraries(test_hull_white_convexity PRIVATE quantape)
quantape_enable_stan(test_hull_white_convexity)
quantape_set_warnings(test_hull_white_convexity)
quantape_register_test(test_hull_white_convexity)

# Double-only curve replication: shares the data/builder/dump fixture header
# with the AD test below and never instantiates Stan scalars.
add_executable(test_paper_eur_curves test_paper_eur_curves.cpp)
target_link_libraries(test_paper_eur_curves PRIVATE quantape)
quantape_set_warnings(test_paper_eur_curves)
quantape_register_test(test_paper_eur_curves)

# Curve evaluation is AD-templated; this target instantiates Stan scalars and
# gates solver-derived bootstrap sensitivities. It is the sole writer of the
# portfolio risk table, so no test ordering or parallelism can race on it.
add_executable(test_curve_ad test_curve_ad.cpp)
target_link_libraries(test_curve_ad PRIVATE quantape)
quantape_enable_stan(test_curve_ad)
quantape_set_warnings(test_curve_ad)
quantape_register_test(test_curve_ad)

# --- Core library ------------------------------------------------------------
foreach(_test test_math test_solvers test_interpolation test_markets test_math_utils)
    add_executable(${_test} ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE quantape)
    quantape_set_warnings(${_test})
    quantape_register_test(${_test})
endforeach()

# test_math_utils instantiates Stan scalars (var / `fvar<var>`) for the
# interpolation and tridiagonal-solver paths; test_markets instantiates
# `fvar<var>` for the survival-probability curve.
quantape_enable_stan(test_math_utils)
quantape_enable_stan(test_markets)

# FX quote records and settlement helpers (double-based, no Stan scalars).
add_executable(test_fx_quote test_fx_quote.cpp)
target_link_libraries(test_fx_quote PRIVATE quantape)
quantape_set_warnings(test_fx_quote)
quantape_register_test(test_fx_quote)

# FX pricing, reset-aware cross-currency risk rows and FX stack risk; the
# double/var/fvar<var> gates instantiate Stan scalars.
add_executable(test_fx_pricing test_fx_pricing.cpp)
target_link_libraries(test_fx_pricing PRIVATE quantape)
quantape_enable_stan(test_fx_pricing)
quantape_set_warnings(test_fx_pricing)
quantape_register_test(test_fx_pricing)

# Dated FX instruments: settlement-lag factories, cashflow materialization,
# date-aware CIP/PV helpers and their var/fvar<var> gates.
add_executable(test_fx_instruments test_fx_instruments.cpp)
target_link_libraries(test_fx_instruments PRIVATE quantape)
quantape_enable_stan(test_fx_instruments)
quantape_set_warnings(test_fx_instruments)
quantape_register_test(test_fx_instruments)

# --- AD and Stan-backed ------------------------------------------------------
add_executable(test_ad test_ad.cpp)
target_link_libraries(test_ad PRIVATE quantape)
quantape_enable_stan(test_ad)
quantape_set_warnings(test_ad)
quantape_register_test(test_ad)

add_executable(test_levenberg_marquardt_ad test_levenberg_marquardt_ad.cpp)
target_link_libraries(test_levenberg_marquardt_ad PRIVATE quantape)
quantape_enable_stan(test_levenberg_marquardt_ad)
quantape_set_warnings(test_levenberg_marquardt_ad)
quantape_register_test(test_levenberg_marquardt_ad)

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
