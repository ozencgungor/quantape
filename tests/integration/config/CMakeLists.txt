# integration/config: JSON fixture loading with compile-time fixture paths.

add_executable(test_curve_config ${CMAKE_SOURCE_DIR}/tests/test_curve_config.cpp)
target_link_libraries(test_curve_config PRIVATE quantape)
target_compile_definitions(test_curve_config PRIVATE
    CURVE_CONFIG_FIXTURE="${CMAKE_SOURCE_DIR}/tests/data/usd_ois_config.json"
    FX_CONFIG_FIXTURE="${CMAKE_SOURCE_DIR}/tests/data/fx_config.json")
quantape_set_warnings(test_curve_config)
quantape_register_test(test_curve_config)

quantape_collect_executables()
