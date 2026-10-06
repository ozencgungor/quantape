# util/: FloatingPoint, Constants, Numeric and number-format gates (plain,
# smoke tier).
quantape_add_gtest(qta_test_util
    SOURCES test_util.cpp test_number_format.cpp
    LABELS unit infra fast smoke)

quantape_collect_executables()
