# integration/config: JSON fixture loading with compile-time fixture paths.
quantape_add_gtest(qta_test_curve_config
    SOURCES test_curve_config.cpp
    DEFS CURVE_CONFIG_FIXTURE="${CMAKE_SOURCE_DIR}/tests/data/usd_ois_config.json"
         FX_CONFIG_FIXTURE="${CMAKE_SOURCE_DIR}/tests/data/fx_config.json"
    LABELS integration markets slow)

quantape_collect_executables()
