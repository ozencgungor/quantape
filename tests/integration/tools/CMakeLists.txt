# integration/tools: subprocess gates around the built Sobol tools. The test
# binary only exists when QUANTAPE_BUILD_TOOLS is ON and the tool targets
# exist (guard kept; the test uses GTEST_SKIP when a macro is empty).

if(TARGET extend_sobol AND TARGET refine_sobol AND TARGET verify_sobol)
    quantape_add_gtest(qta_test_sobol_tools
        SOURCES test_sobol_tools.cpp
        LIBS Threads::Threads
        DEFS EXTEND_BIN="$<TARGET_FILE:extend_sobol>"
             REFINE_BIN="$<TARGET_FILE:refine_sobol>"
             VERIFY_BIN="$<TARGET_FILE:verify_sobol>"
        LABELS integration mc slow)
    add_dependencies(qta_test_sobol_tools extend_sobol refine_sobol verify_sobol)
else()
    message(STATUS "test_sobol_tools skipped: QUANTAPE_BUILD_TOOLS=OFF")
endif()

quantape_collect_executables()
