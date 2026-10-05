cmake_minimum_required(VERSION 3.20)
project(quantape)

# ============================================================================
# Project settings
# ============================================================================

# Allow subdirectories with older CMake minimum versions
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------------
# Build types: Debug (default) and Release are the supported configurations.
# Debug: -g, no optimization (debugger-friendly). Release: -O3 -DNDEBUG plus
# -march=native below. -ffast-math is deliberately absent: it breaks the IEEE
# semantics (NaN/inf, signed zero, reassociation) the AD stack relies on.
# ----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# Debug is the debugger-first configuration: full optimization off, DWARF with
# macro information (-g3), frame pointers and sibling calls kept so backtraces
# and single-stepping are reliable. NDEBUG stays undefined, so asserts fire.
# Release keeps CMake's -O3 -DNDEBUG and adds -march=native. MSVC uses its own
# default /Od /Zi and /O2 /DNDEBUG pairs.
if(NOT MSVC)
    add_compile_options(
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-g3>
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
        $<$<CONFIG:Debug>:-fno-optimize-sibling-calls>
        $<$<CONFIG:Release>:-march=native>
    )
endif()

option(QUANTAPE_BUILD_EXAMPLES "Build example executables" ON)
option(QUANTAPE_BUILD_TOOLS "Build command-line tools" ON)
option(QUANTAPE_BUILD_TESTS "Build test executables" ON)
# Benchmarks measure optimized code, so they default ON only for Release.
if(NOT DEFINED QUANTAPE_BUILD_BENCHMARKS)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(QUANTAPE_BUILD_BENCHMARKS ON CACHE BOOL "Build benchmark executables")
    else()
        set(QUANTAPE_BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmark executables")
    endif()
endif()
option(QUANTAPE_TEST_OPTIMIZATION
       "Build tests with -O2 in optimized configurations (never in Debug)" ON)
option(QUANTAPE_ENABLE_LOGGING "Build with quill-based logging (macros become no-ops when OFF)" ON)
set(QUANTAPE_LOG_COMPILE_LEVEL "" CACHE STRING
    "Compile-time log level: TRACE/DEBUG/INFO/WARN/ERROR/CRITICAL (default: TRACE in Debug, INFO otherwise)")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(Dependencies)
include(QuantapeTargets)

# Public headers are included as quantape/...
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${Boost_INCLUDE_DIRS})

# ============================================================================
# Library
# ============================================================================

file(GLOB_RECURSE QUANTAPE_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/quantape/*.h)
file(GLOB_RECURSE QUANTAPE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
# Logging and formatting are separate small libraries: build-time tools such as
# sobol_to_binary must be able to use them without linking the full library
# (which would create a target cycle through the generated Sobol asset).
list(FILTER QUANTAPE_SOURCES EXCLUDE REGEX "/src/(log|format)/")

add_library(quantape STATIC ${QUANTAPE_SOURCES} ${QUANTAPE_HEADERS})
target_include_directories(quantape PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${Boost_INCLUDE_DIRS}
)
quantape_set_warnings(quantape)

# Infrastructure libraries: logging facade and number formatting. Both are
# part of the public headers, so quantape links them PUBLIC.
add_library(quantape_log STATIC src/log/Log.cpp)
target_include_directories(quantape_log PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
quantape_set_warnings(quantape_log)

add_library(quantape_format STATIC src/format/Number.cpp)
target_include_directories(quantape_format PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
quantape_set_warnings(quantape_format)
target_link_libraries(quantape_format PUBLIC zmij::zmij)

if(QUANTAPE_ENABLE_LOGGING)
    target_link_libraries(quantape_log PUBLIC quill::quill)
    if(QUANTAPE_LOG_COMPILE_LEVEL)
        string(TOUPPER "${QUANTAPE_LOG_COMPILE_LEVEL}" _qta_level)
        if(_qta_level STREQUAL "TRACE")
            set(_qta_quill_level TRACE_L3)
        elseif(_qta_level STREQUAL "DEBUG")
            set(_qta_quill_level DEBUG)
        elseif(_qta_level STREQUAL "INFO")
            set(_qta_quill_level INFO)
        elseif(_qta_level STREQUAL "WARN")
            set(_qta_quill_level WARNING)
        elseif(_qta_level STREQUAL "ERROR")
            set(_qta_quill_level ERROR)
        elseif(_qta_level STREQUAL "CRITICAL")
            set(_qta_quill_level CRITICAL)
        else()
            message(FATAL_ERROR "Unknown QUANTAPE_LOG_COMPILE_LEVEL: ${QUANTAPE_LOG_COMPILE_LEVEL}")
        endif()
        target_compile_definitions(quantape_log PUBLIC
            QUILL_COMPILE_ACTIVE_LOG_LEVEL=QUILL_COMPILE_ACTIVE_LOG_LEVEL_${_qta_quill_level})
    else()
        target_compile_definitions(quantape_log PUBLIC
            $<$<CONFIG:Debug>:QUILL_COMPILE_ACTIVE_LOG_LEVEL=QUILL_COMPILE_ACTIVE_LOG_LEVEL_TRACE_L3>
            $<$<NOT:$<CONFIG:Debug>>:QUILL_COMPILE_ACTIVE_LOG_LEVEL=QUILL_COMPILE_ACTIVE_LOG_LEVEL_INFO>)
    endif()
else()
    target_compile_definitions(quantape_log PUBLIC QUANTAPE_DISABLE_LOGGING=1)
endif()

target_link_libraries(quantape PUBLIC quantape_log quantape_format)
target_link_libraries(quantape PRIVATE nlohmann_json::nlohmann_json)

# ============================================================================
# Generated assets
#
# Sobol direction-number table (QSB1 binary, zero-copy mmap). Default: the
# refined 65,536-dimension table, falling back to the unrefined run and then
# the Joe-Kuo D(7) base table. Override with
#   cmake -DQUANTAPE_SOBOL_TABLE=/path/to/table
# ============================================================================

add_executable(sobol_to_binary tools/sobol_to_binary.cpp)
target_link_libraries(sobol_to_binary PRIVATE quantape_log)
quantape_collect_executables()

set(_sobol_table_candidates
    "${CMAKE_CURRENT_SOURCE_DIR}/joe-kuo-65536-refined-w256.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/joe-kuo-65536-unrefined-w256.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/internal_docs/new-joe-kuo-7.21201")
set(_sobol_default "")
foreach(_candidate IN LISTS _sobol_table_candidates)
    if(EXISTS "${_candidate}")
        set(_sobol_default "${_candidate}")
        break()
    endif()
endforeach()
set(QUANTAPE_SOBOL_TABLE "${_sobol_default}" CACHE FILEPATH
    "Joe-Kuo direction-number table baked into the build (default: refined 65,536-dim table)")
if(NOT QUANTAPE_SOBOL_TABLE STREQUAL "" AND NOT EXISTS "${QUANTAPE_SOBOL_TABLE}")
    message(STATUS "configured QUANTAPE_SOBOL_TABLE does not exist: ${QUANTAPE_SOBOL_TABLE}")
    if(NOT _sobol_default STREQUAL "")
        message(STATUS "falling back to the default: ${_sobol_default}")
        set(QUANTAPE_SOBOL_TABLE "${_sobol_default}" CACHE FILEPATH
            "Joe-Kuo direction-number table baked into the build" FORCE)
    endif()
endif()
if(EXISTS "${QUANTAPE_SOBOL_TABLE}")
    message(STATUS "Sobol table (compile-time): ${QUANTAPE_SOBOL_TABLE}")
    set(QUANTAPE_SOBOL_ASSET "${CMAKE_BINARY_DIR}/sobol_table.qsb")
    add_custom_command(
        OUTPUT "${QUANTAPE_SOBOL_ASSET}"
        COMMAND $<TARGET_FILE:sobol_to_binary> --bits=32 "${QUANTAPE_SOBOL_TABLE}" "${QUANTAPE_SOBOL_ASSET}"
        DEPENDS sobol_to_binary "${QUANTAPE_SOBOL_TABLE}"
        COMMENT "Building Sobol binary asset from ${QUANTAPE_SOBOL_TABLE}")
    add_custom_target(sobol_table ALL DEPENDS "${QUANTAPE_SOBOL_ASSET}")
    target_compile_definitions(quantape PUBLIC
        QUANTAPE_SOBOL_DEFAULT_TABLE_PATH="${QUANTAPE_SOBOL_ASSET}")
    add_dependencies(quantape sobol_table)
else()
    message(STATUS "QUANTAPE_SOBOL_TABLE not found (${QUANTAPE_SOBOL_TABLE}); "
                   "SobolGenerator::fromDefaultTable will be unavailable")
endif()

# ============================================================================
# Executables
# ============================================================================

if(QUANTAPE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(QUANTAPE_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
if(QUANTAPE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(QUANTAPE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Single target depending on every executable (IDE-friendly build target).
get_property(QUANTAPE_EXECUTABLES GLOBAL PROPERTY QUANTAPE_EXECUTABLE_TARGETS)
if(QUANTAPE_EXECUTABLES)
    add_custom_target(build_all DEPENDS ${QUANTAPE_EXECUTABLES})
endif()

# ============================================================================
# Compiler options
#
# Debug and Release are the supported configurations (defined at the top of
# this file); a target's optimization follows its configuration. Tests are the
# single documented exception: quantape_register_test adds -O2 in optimized
# configurations (never in Debug) because the numerical test loops are
# impractical at -O0.
# ============================================================================

# ============================================================================
# Summary
# ============================================================================

message(STATUS "===========================================")
message(STATUS "Third-party libraries configuration:")
message(STATUS "  Eigen:      ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  Boost:      ${Boost_INCLUDE_DIRS}")
message(STATUS "  TBB:        ${TBB_INCLUDE_DIR}")
message(STATUS "  SUNDIALS:   ${SUNDIALS_INCLUDE_DIR}")
message(STATUS "  dlib:       ${THIRD_PARTY_DIR}/dlib (disabled)")
message(STATUS "  Stan Math:  ${STAN_MATH_INCLUDE_DIR}")
if(QUANTAPE_ENABLE_LOGGING)
    message(STATUS "  quill:      ${quill_SOURCE_DIR} (v13.0.0)")
else()
    message(STATUS "  quill:      disabled (QUANTAPE_ENABLE_LOGGING=OFF)")
endif()
message(STATUS "  zmij:       ${zmij_SOURCE_DIR} (v1.2)")
message(STATUS "===========================================")

# ============================================================================
# API documentation (Doxygen)
# ============================================================================
find_package(Doxygen QUIET)
if(Doxygen_FOUND)
    add_custom_target(doc
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/docs/html
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation (docs/html)"
        VERBATIM)
endif()

# Formatting automation (clang-format via format_code.sh).
add_custom_target(format
    COMMAND "${CMAKE_SOURCE_DIR}/format_code.sh"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Formatting C++ sources with clang-format")

add_custom_target(format-check
    COMMAND "${CMAKE_SOURCE_DIR}/format_code.sh" --check
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Checking C++ formatting with clang-format")
