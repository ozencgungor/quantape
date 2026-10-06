# ============================================================================
# Test executables. Each area owns one directory with its own CMakeLists.txt
# (one writer per area during the conversion); this file only aggregates the
# areas. Legacy `quantape_register_test` targets move into their area file
# mechanically and are removed there when the file converts to googletest.
# ============================================================================

# tests/ is on every test target's include path so support/... resolves from
# this directory and all area subdirectories (directory properties inherit).
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(unit/util)
add_subdirectory(unit/infra)
add_subdirectory(unit/datetime)
add_subdirectory(unit/math)
add_subdirectory(unit/calibration)
add_subdirectory(unit/markets)
add_subdirectory(unit/markets/ad)
add_subdirectory(unit/instruments)
add_subdirectory(unit/pricing)
add_subdirectory(unit/mc)
add_subdirectory(unit/models)
add_subdirectory(integration/config)
add_subdirectory(integration/tools)
