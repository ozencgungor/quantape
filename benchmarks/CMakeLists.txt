# ============================================================================
# Benchmark executables (built by default, not registered with CTest).
# ============================================================================

add_executable(bench_sobol bench_sobol.cpp)
target_link_libraries(bench_sobol PRIVATE quantape TBB::tbb)

add_executable(bench_daycount bench_daycount.cpp)
target_link_libraries(bench_daycount PRIVATE quantape_log quantape_format)
quantape_set_warnings(bench_daycount)

add_executable(bench_curve bench_curve.cpp)
target_link_libraries(bench_curve PRIVATE quantape)
quantape_set_warnings(bench_curve)

add_executable(bench_sde bench_sde.cpp)
target_link_libraries(bench_sde PRIVATE quantape TBB::tbb)
quantape_enable_stan_defs(bench_sde)
quantape_set_warnings(bench_sde)

# Allocation-tracking build: replaces malloc/free to count every allocation
# (Eigen included). Timings from this target are distorted by the allocator
# shim; use it for allocation counts, bench_sde for timings.
add_executable(bench_sde_alloc bench_sde.cpp)
target_compile_definitions(bench_sde_alloc PRIVATE QUANTAPE_SDE_ALLOC_COUNTS=1)
target_link_libraries(bench_sde_alloc PRIVATE quantape TBB::tbb)
quantape_enable_stan_defs(bench_sde_alloc)
quantape_set_warnings(bench_sde_alloc)

add_executable(bench_risk bench_risk.cpp)
target_link_libraries(bench_risk PRIVATE quantape)
quantape_set_warnings(bench_risk)

# Allocation-tracking build of the risk harness: same malloc shim as
# bench_sde_alloc; use it for allocation counts, bench_risk for timings.
add_executable(bench_risk_alloc bench_risk.cpp)
target_compile_definitions(bench_risk_alloc PRIVATE QUANTAPE_RISK_ALLOC_COUNTS=1)
target_link_libraries(bench_risk_alloc PRIVATE quantape)
quantape_set_warnings(bench_risk_alloc)

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
