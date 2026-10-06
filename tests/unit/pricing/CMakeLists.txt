# pricing/: Black-76/GBS analytic Greeks vs FD and Stan AD (tape fixture).
quantape_add_gtest(qta_test_pricing
    SOURCES test_black76.cpp
    STAN
    LABELS unit pricing ad slow)
quantape_collect_executables()
