# instruments/: cashflow vocabulary, money-market and FX instruments. The AD
# gates instantiate Stan scalars; test_fx_quote is double-only but shares the
# instruments/FX grouping.
quantape_add_gtest(qta_test_instruments
    SOURCES test_instruments.cpp test_ir_instruments.cpp test_fx_quote.cpp
            test_fx_pricing.cpp test_fx_instruments.cpp
    STAN
    LABELS unit instruments fx ad slow)

quantape_collect_executables()
