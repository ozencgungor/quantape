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

quantape_add_gtest(qta_test_mc_ad
    SOURCES test_sde_schemes.cpp
            test_sde_gradients.cpp
            test_sde_qmc.cpp
    STAN
    LABELS unit mc ad slow)

quantape_collect_executables()
