# util/: FloatingPoint, Constants, Numeric gates (plain, smoke tier).
quantape_add_gtest(qta_test_util
    SOURCES test_util.cpp
    LABELS unit infra fast smoke)
quantape_collect_executables()
