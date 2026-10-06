# integration/tools: subprocess gates around the built Sobol tools. The test
# binary only exists when QUANTAPE_BUILD_TOOLS is ON and the tool targets
# exist (guard kept; the test uses GTEST_SKIP when a macro is empty).

if(TARGET extend_sobol AND TARGET refine_sobol AND TARGET verify_sobol)
    add_executable(test_sobol_tools ${CMAKE_SOURCE_DIR}/tests/test_sobol_tools.cpp)
    target_link_libraries(test_sobol_tools PRIVATE Threads::Threads quantape_log quantape_format)
    target_compile_definitions(test_sobol_tools PRIVATE
        EXTEND_BIN="$<TARGET_FILE:extend_sobol>"
        REFINE_BIN="$<TARGET_FILE:refine_sobol>"
        VERIFY_BIN="$<TARGET_FILE:verify_sobol>")
    add_dependencies(test_sobol_tools extend_sobol refine_sobol verify_sobol)
    quantape_register_test(test_sobol_tools)
else()
    message(STATUS "test_sobol_tools skipped: QUANTAPE_BUILD_TOOLS=OFF")
endif()

quantape_collect_executables()
