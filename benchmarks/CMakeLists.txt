# ============================================================================
# Benchmark executables (built by default, not registered with CTest).
# ============================================================================

add_executable(bench_sobol bench_sobol.cpp)
target_link_libraries(bench_sobol PRIVATE quantape TBB::tbb)

add_executable(bench_daycount bench_daycount.cpp)
target_link_libraries(bench_daycount PRIVATE quantape_log quantape_format)
quantape_set_warnings(bench_daycount)

add_executable(bench_sde bench_sde.cpp)
target_link_libraries(bench_sde PRIVATE quantape TBB::tbb)
quantape_enable_stan_defs(bench_sde)
quantape_set_warnings(bench_sde)

# --- Stan/AD micro-benchmarks -------------------------------------------------
foreach(_bench test_bs_ad_bench test_interp_ad_bench test_fast_exp_log
                test_bvn_ad_bench)
    add_executable(${_bench} ${_bench}.cpp)
    target_link_libraries(${_bench} PRIVATE quantape)
    quantape_enable_stan(${_bench})
    quantape_set_warnings(${_bench})
endforeach()

add_executable(test_bs_hessian_bench test_bs_hessian_bench.cpp)
target_link_libraries(test_bs_hessian_bench PRIVATE quantape)
quantape_enable_stan(test_bs_hessian_bench)
quantape_set_warnings(test_bs_hessian_bench)

add_executable(test_optimizers_bench test_optimizers_bench.cpp)
quantape_enable_stan(test_optimizers_bench)
quantape_set_warnings(test_optimizers_bench)

quantape_collect_executables()
