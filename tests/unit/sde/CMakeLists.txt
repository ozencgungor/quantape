# sde/: market snapshot and the analytic state->curve evaluation seam.

quantape_add_gtest(qta_test_sde
    SOURCES test_frozen_market.cpp
            test_state_curve.cpp
    LABELS unit sde fast)

# The AD target links full Stan for the `var` / `fvar<var>` evaluation gates.
quantape_add_gtest(qta_test_sde_ad
    SOURCES test_state_curve_ad.cpp
            test_numeraire.cpp
    STAN
    LABELS unit sde ad slow)

quantape_collect_executables()
