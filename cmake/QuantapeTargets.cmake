# ============================================================================
# Shared target helpers for quantape executables.
# ============================================================================

include_guard(GLOBAL)

# gtest_discover_tests() registers one CTest case per gtest test.
include(GoogleTest)

# Standard warning set (used by targets that had warnings before the split).
function(quantape_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        # GCC disables column tracking on huge headers (Stan) and prints a note
        # per TU even when those headers are system includes; turning the check
        # off removes the note. Clang has no misleading-indentation diagnostic.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${target} PRIVATE -Wno-misleading-indentation)
        endif()
    endif()
endfunction()

# Make Stan-backed executables find libtbb at runtime (macOS/Linux).
function(quantape_set_tbb_rpath target)
    if((APPLE OR UNIX) AND TARGET TBB::tbb)
        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH "$<TARGET_FILE_DIR:TBB::tbb>"
            BUILD_WITH_INSTALL_RPATH TRUE)
        if(UNIX AND NOT APPLE)
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "$ORIGIN/../lib:$<TARGET_FILE_DIR:TBB::tbb>")
        endif()
    endif()
endfunction()

# Thread-safe compile definitions required by Stan Math.
function(quantape_enable_stan_defs target)
    target_compile_definitions(${target} PRIVATE
        STAN_THREADS=true
        STAN_NO_RANGE_CHECKS
        _REENTRANT
        TBB_INTERFACE_NEW
    )
endfunction()

# Full Stan Math wiring: TBB + SUNDIALS (CVODES) + thread defs + rpath.
function(quantape_enable_stan target)
    target_link_libraries(${target} PUBLIC
        TBB::tbb
        SUNDIALS::nvecserial
        SUNDIALS::cvodes
    )
    quantape_enable_stan_defs(${target})
    quantape_set_tbb_rpath(${target})
endfunction()

# Register a test executable with CTest (executables remain runnable directly).
#
# Tests are numerical loops and run far too slowly unoptimized, so optimized
# configurations build them with -O2 on top of the configuration flags.
# Debug is never optimized: tests must be steppable like any other target.
function(quantape_register_test target)
    if(QUANTAPE_TEST_OPTIMIZATION)
        if(MSVC)
            target_compile_options(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2>)
        else()
            target_compile_options(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O2>)
        endif()
    endif()
    # Test translation units instantiate the engine templates; without FP
    # contraction the same-expression paths stay bitwise identical across
    # compilers (GCC fuses multiplies by default, AppleClang does not), which
    # gates like Euler == Milstein for constant diffusion rely on. Library
    # targets keep the default (fused) code for speed.
    if(NOT MSVC)
        target_compile_options(${target} PRIVATE -ffp-contract=off)
    endif()
    add_test(NAME ${target} COMMAND ${target})
endfunction()

# Register a GoogleTest executable with per-case CTest discovery.
#
# quantape_add_gtest(<target> SOURCES <files...> [STAN] [STAN_DEFS]
#                    [LIBS <targets...>] [DEFS <defs...>] [FLAGS <flags...>]
#                    [LABELS <labels...>] [TIMEOUT <sec>])
#
# STAN / STAN_DEFS select the Stan link profile and define QTA_TEST_STAN, which
# gates the Stan PrintTo overloads in tests/support/Printers.h. tests/ is on
# the include path so every area can `#include "support/GtestSupport.h"`.
# Each per-area CMakeLists.txt must end with quantape_collect_executables() so
# `build_all` sees its targets (the function collects the current directory
# only).
function(quantape_add_gtest name)
    cmake_parse_arguments(A "STAN;STAN_DEFS" "TIMEOUT"
        "SOURCES;LIBS;DEFS;FLAGS;LABELS" ${ARGN})
    if(NOT A_SOURCES)
        message(FATAL_ERROR "quantape_add_gtest(${name}): SOURCES required")
    endif()

    add_executable(${name} ${A_SOURCES})
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(${name} PRIVATE quantape GTest::gtest_main ${A_LIBS})

    if(A_STAN)
        quantape_enable_stan(${name})
    elseif(A_STAN_DEFS)
        quantape_enable_stan_defs(${name})
        target_link_libraries(${name} PRIVATE TBB::tbb)
    endif()
    if(A_STAN OR A_STAN_DEFS)
        target_compile_definitions(${name} PRIVATE QTA_TEST_STAN=1)
    endif()

    target_compile_definitions(${name} PRIVATE
        QTA_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data" ${A_DEFS})
    if(NOT MSVC)
        target_compile_options(${name} PRIVATE -ffp-contract=off)
    endif()
    target_compile_options(${name} PRIVATE ${A_FLAGS})

    if(QUANTAPE_TEST_OPTIMIZATION)
        if(MSVC)
            target_compile_options(${name} PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2>)
        else()
            target_compile_options(${name} PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O2>)
        endif()
    endif()
    quantape_set_warnings(${name})

    if(NOT A_LABELS)
        set(A_LABELS unit fast)
    endif()
    if(NOT A_TIMEOUT)
        set(A_TIMEOUT 600)
    endif()
    set(QUANTAPE_GTEST_EXTRA_ARGS "" CACHE STRING
        "Extra gtest arguments for discovery (nightly: --gtest_shuffle;--gtest_repeat=3)")

    gtest_discover_tests(${name}
        TEST_PREFIX "${name}."
        PROPERTIES
            LABELS "${A_LABELS}"
            TIMEOUT ${A_TIMEOUT}
        DISCOVERY_MODE POST_BUILD
        DISCOVERY_TIMEOUT 60
        EXTRA_ARGS ${QUANTAPE_GTEST_EXTRA_ARGS})
endfunction()

# Collect all executable targets of the current directory into a global list,
# so the root can define one build-all target for IDE build configurations.
function(quantape_collect_executables)
    get_property(_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target IN LISTS _targets)
        get_target_property(_type ${_target} TYPE)
        if(_type STREQUAL "EXECUTABLE")
            set_property(GLOBAL APPEND PROPERTY QUANTAPE_EXECUTABLE_TARGETS ${_target})
        endif()
    endforeach()
endfunction()
