# markets/: curve, quote, futures and survival gates. All plain except
# test_markets, which instantiates `fvar<var>` for the survival curve.

quantape_add_gtest(qta_test_markets
    SOURCES test_curve.cpp
            test_synthetic_quotes.cpp
            test_bootstrap_validation.cpp
            test_forecast_pillars.cpp
            test_futures.cpp
            test_instrument_options.cpp
    LABELS unit markets fast)

# test_markets instantiates `fvar<var>` for the survival-probability curve and
# therefore links full Stan.
quantape_add_gtest(qta_test_markets_ad
    SOURCES test_markets.cpp
    STAN
    LABELS unit markets ad slow)

quantape_collect_executables()
