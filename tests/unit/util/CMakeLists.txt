# util/: FloatingPoint, Constants, Numeric gates (plain, smoke tier).
quantape_add_gtest(qta_test_util
    SOURCES test_util.cpp
    LABELS unit infra fast smoke)

# Legacy stub until test_number_format converts: the source stays in tests/
# for now and the registration must stay byte-identical.
add_executable(test_number_format ${CMAKE_SOURCE_DIR}/tests/test_number_format.cpp)
target_link_libraries(test_number_format PRIVATE quantape)
quantape_set_warnings(test_number_format)
quantape_register_test(test_number_format)

quantape_collect_executables()
