cmake_minimum_required(VERSION 3.20)
project(lightape LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LIGHTAPE_BUILD_TESTS "Build lightape tests" OFF)
option(LIGHTAPE_BUILD_BENCHMARKS "Build lightape benchmarks" OFF)
option(LIGHTAPE_VECTOR_MATH "Vector math backend (libm|sleef)" "libm")

# CMake's `option()` is boolean, so it coerces the non-boolean default "libm"
# to OFF; restore the documented string default. A `-DLIGHTAPE_VECTOR_MATH=...`
# command-line value is preserved verbatim.
if(NOT LIGHTAPE_VECTOR_MATH)
    set(LIGHTAPE_VECTOR_MATH "libm")
endif()

if(NOT LIGHTAPE_VECTOR_MATH STREQUAL "libm" AND NOT LIGHTAPE_VECTOR_MATH STREQUAL "sleef")
    message(FATAL_ERROR "LIGHTAPE_VECTOR_MATH must be `libm` or `sleef` (got `${LIGHTAPE_VECTOR_MATH}`)")
endif()

# SLEEF backend (decision #28): pinned static build, explicit-ISA symbols,
# linked only into the test/bench executables that exercise VectorMath -- never
# into the `lightape` interface target, and the default `libm` configure never
# downloads it.
#
# Denormal / FTZ behaviour (decision #22): SLEEF 3.9.0 compiles libsleef with
# `-ffp-contract=off -fno-math-errno -fno-trapping-math` (Configure.cmake) and
# contains no MXCSR/FPCR manipulation, no `-ffast-math` on the libm objects
# (`-ffast-math` appears only on the unused DFT tests), and therefore preserves
# IEEE denormals; we add no FTZ/DAZ flags either. `-ffp-contract=off` stays on
# our test/bench TUs.
set(_lightape_sleef OFF)
if(LIGHTAPE_VECTOR_MATH STREQUAL "sleef")
    set(_lightape_sleef ON)
    enable_language(C)
    include(FetchContent)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    set(SLEEF_BUILD_SHARED_LIBS OFF CACHE BOOL "SLEEF static library" FORCE)
    set(SLEEF_BUILD_TESTS OFF CACHE BOOL "Skip SLEEF test suite" FORCE)
    set(SLEEF_BUILD_DFT OFF CACHE BOOL "Skip SLEEF DFT library" FORCE)
    set(SLEEF_BUILD_QUAD OFF CACHE BOOL "Skip SLEEF quad library" FORCE)
    set(SLEEF_BUILD_GNUABI_LIBS OFF CACHE BOOL "Skip SLEEF GNUABI alias library" FORCE)
    set(SLEEF_BUILD_SCALAR_LIB OFF CACHE BOOL "Skip SLEEF scalar-only library" FORCE)
    set(SLEEF_ENABLE_LTO OFF CACHE BOOL "No SLEEF LTO" FORCE)
    set(SLEEF_ENABLE_TLFLOAT OFF CACHE BOOL "Skip TLFloat probing" FORCE)
    # Keeps the suite hermetic: otherwise SLEEF registers its internal PSHA
    # self-tests against the system OpenSSL.
    set(SLEEF_DISABLE_SSL ON CACHE BOOL "Skip SLEEF/OpenSSL integration" FORCE)
    set(SLEEF_SHOW_CONFIG ON CACHE BOOL "Show SLEEF configuration" FORCE)
    FetchContent_Declare(sleef
        URL https://github.com/shibatch/sleef/archive/refs/tags/3.9.0.tar.gz
        URL_HASH SHA256=af60856abac08a3b5e72a8d156dd71fec1f7ac23de8ee67793f45f9edcdf0908
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(sleef)
    message(STATUS "lightape: vector math backend = sleef 3.9.0 (static; denormal-preserving)")
endif()

add_library(lightape INTERFACE)
add_library(lightape::lightape ALIAS lightape)
target_include_directories(lightape INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(lightape INTERFACE cxx_std_20)

# Per-ISA SLEEF kernel TUs (x86-64 only): every SLEEF call lives here so the
# portable header and the libm reference TUs never see sleef.h or ISA flags.
# SleefGeneric.cpp is the portable scalar plan; SleefAvx2.cpp hosts the 4-wide
# AVX2 and the 2-wide AVX2128 kernels (both fit the same -mavx2 -mfma flags);
# SleefAvx512.cpp the 8-wide AVX-512F kernels. On AArch64 (and every other
# arch) the kernels stay header-only: NEON is baseline, so no TU is needed.
set(_lightape_sleef_x86 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" OR
   CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
    set(_lightape_sleef_x86 "${_lightape_sleef}")
endif()
if(_lightape_sleef_x86)
    add_library(lightape_sleef_kernels OBJECT
        src/batch/SleefGeneric.cpp
        src/batch/SleefAvx2.cpp
        src/batch/SleefAvx512.cpp)
    target_link_libraries(lightape_sleef_kernels PRIVATE lightape::lightape)
    add_dependencies(lightape_sleef_kernels sleef) # sleef.h is a build output
    target_include_directories(lightape_sleef_kernels SYSTEM PRIVATE ${sleef_BINARY_DIR}/include)
    target_compile_definitions(lightape_sleef_kernels PRIVATE SLEEF_STATIC_LIBS=1)
    set_source_files_properties(src/batch/SleefGeneric.cpp PROPERTIES
        COMPILE_OPTIONS "-ffp-contract=off")
    set_source_files_properties(src/batch/SleefAvx2.cpp PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=off")
    set_source_files_properties(src/batch/SleefAvx512.cpp PROPERTIES
        COMPILE_OPTIONS
        "-mavx512f;-mavx512vl;-mavx512dq;-mavx512bw;-mavx2;-mfma;-ffp-contract=off")
endif()

# Attach the SLEEF backend to one test/bench executable. `sleef.h` (for the
# AArch64 header kernels) and the static libsleef link stay off the portable
# `lightape` interface target; the x86-64 kernel objects are added only here.
function(_lightape_link_sleef _target)
    target_link_libraries(${_target} PRIVATE sleef)
    target_include_directories(${_target} SYSTEM PRIVATE ${sleef_BINARY_DIR}/include)
    target_compile_definitions(${_target} PRIVATE LIGHTAPE_VECTOR_MATH_SLEEF=1
                                                  SLEEF_STATIC_LIBS=1)
    if(_lightape_sleef_x86)
        target_sources(${_target} PRIVATE $<TARGET_OBJECTS:lightape_sleef_kernels>)
    endif()
endfunction()

if(LIGHTAPE_BUILD_TESTS)
    enable_testing()
    if(NOT TARGET gtest_main)
        include(FetchContent)
        if(POLICY CMP0135)
            cmake_policy(SET CMP0135 NEW)
        endif()
        set(BUILD_GMOCK OFF CACHE BOOL "Build googlemock" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "Install GoogleTest" FORCE)
        set(GTEST_HAS_ABSL OFF CACHE BOOL "Build GoogleTest with Abseil" FORCE)
        set(gtest_build_samples OFF CACHE BOOL "Build GoogleTest samples" FORCE)
        set(gtest_build_tests OFF CACHE BOOL "Build GoogleTest tests" FORCE)
        set(gtest_disable_pthreads OFF CACHE BOOL "Disable pthreads in GoogleTest" FORCE)
        set(_lightape_build_testing_saved "${BUILD_TESTING}")
        set(BUILD_TESTING OFF)
        FetchContent_Declare(googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
            URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(googletest)
        set(BUILD_TESTING "${_lightape_build_testing_saved}")
    endif()

    set(_lightape_eigen_sources "")
    set(_lightape_eigen_include "")
    find_package(Eigen3 QUIET NO_MODULE)
    if(TARGET Eigen3::Eigen)
        set(_lightape_eigen_sources tests/test_eigen_scalars.cpp tests/test_eigen_numext.cpp
                                     tests/test_matrix_blocks.cpp tests/test_eigen_select.cpp)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../third-party/eigen/Eigen/Core")
        set(_lightape_eigen_sources tests/test_eigen_scalars.cpp tests/test_eigen_numext.cpp
                                     tests/test_matrix_blocks.cpp tests/test_eigen_select.cpp)
        set(_lightape_eigen_include "${CMAKE_CURRENT_SOURCE_DIR}/../third-party/eigen")
    else()
        message(STATUS "lightape: Eigen not found; skipping test_eigen_scalars.cpp")
    endif()

    add_executable(lightape_tests
        tests/test_lean_reverse.cpp
        tests/test_forward_scalar.cpp
        tests/test_forward_ops.cpp
        tests/test_forward_completion.cpp
        tests/test_reverse_ops.cpp
        tests/test_external_ops.cpp
        tests/test_softmax.cpp
        tests/test_hessian.cpp
        tests/test_second_order.cpp
        tests/test_callbacks.cpp
        tests/test_tape_lifecycle.cpp
        tests/test_p1_functions.cpp
        tests/test_bessel_ad.cpp
        tests/test_noncentral_chi_square_ad.cpp
        tests/test_bvn_ad.cpp
        tests/test_elementary_extras.cpp
        tests/test_record_opt.cpp
        tests/test_mode_mixing.cpp
        tests/test_polygamma_ad.cpp
        tests/test_batch_tape.cpp
        tests/test_batch_functions.cpp
        tests/test_vector_math.cpp
        tests/test_sparsity.cpp
        tests/test_p2_functions.cpp
        tests/test_third_order.cpp
        tests/test_determinism.cpp
        tests/test_inv_inc_beta.cpp
        tests/test_bessel_jy_ad.cpp
        tests/test_revolve.cpp
        tests/test_checkpoint.cpp
        tests/test_generic_ops.cpp
        tests/test_probability.cpp
        tests/test_digamma.cpp
        tests/test_exec.cpp
        tests/test_math_gaps.cpp
        tests/test_revtapet.cpp
        tests/test_revscalarn.cpp
        tests/test_all_headers.cpp
        ${_lightape_eigen_sources})
    target_link_libraries(lightape_tests PRIVATE lightape::lightape GTest::gtest_main)
    # Enables detail::setMathTableForTesting so the suite can pin per-function
    # resolved tables (fingerprints, forced-libm bitwise reference).
    target_compile_definitions(lightape_tests PRIVATE LIGHTAPE_VECTOR_MATH_TESTING=1)
    if(_lightape_eigen_sources)
        # test_all_headers.cpp pulls the Eigen adapters only when Eigen is here.
        target_compile_definitions(lightape_tests PRIVATE LIGHTAPE_TEST_ALL_HEADERS_EIGEN=1)
    endif()
    if(NOT MSVC)
        # Keep the libm bitwise gates in test_vector_math.cpp exact: without
        # this, LLVM may fuse a sin/cos pair into Darwin's `__sincos` in one
        # evaluation and not the other (1 ULP apart on M-series).
        set_source_files_properties(tests/test_vector_math.cpp PROPERTIES COMPILE_OPTIONS
                                                                       "-fno-builtin")
    endif()
    if(TARGET Eigen3::Eigen)
        target_link_libraries(lightape_tests PRIVATE Eigen3::Eigen)
    endif()
    if(_lightape_eigen_include)
        target_include_directories(lightape_tests PRIVATE ${_lightape_eigen_include})
    endif()
    if(NOT MSVC)
        target_compile_options(lightape_tests PRIVATE -Wall -Wextra -ffp-contract=off)
    endif()
    include(GoogleTest)
    gtest_discover_tests(lightape_tests)

    # SLEEF-backend test executable: the batch seam tests are compiled twice.
    # `lightape_tests` above stays on the libm bitwise reference (the full
    # suite, including the unmodified batch bitwise gates in
    # test_batch_functions.cpp / test_determinism.cpp), and this second binary
    # runs the same test_batch_tape.cpp plus test_vector_math.cpp against the
    # SLEEF plans with tolerance-gated assertions.
    if(_lightape_sleef)
        add_executable(lightape_tests_sleef
            tests/test_vector_math.cpp
            tests/test_batch_tape.cpp)
        target_link_libraries(lightape_tests_sleef PRIVATE lightape::lightape GTest::gtest_main)
        _lightape_link_sleef(lightape_tests_sleef)
        target_compile_definitions(lightape_tests_sleef PRIVATE LIGHTAPE_VECTOR_MATH_TESTING=1)
        if(NOT MSVC)
            target_compile_options(lightape_tests_sleef PRIVATE -Wall -Wextra -ffp-contract=off)
        endif()
        gtest_discover_tests(lightape_tests_sleef TEST_PREFIX "Sleef.")
    endif()
endif()

if(LIGHTAPE_BUILD_BENCHMARKS)
    add_executable(lightape_bench benchmarks/bench_lean_reverse.cpp)
    target_link_libraries(lightape_bench PRIVATE lightape::lightape)
    add_executable(lightape_batch_bench benchmarks/bench_batch_tape.cpp)
    target_link_libraries(lightape_batch_bench PRIVATE lightape::lightape)
    add_executable(lightape_vector_math_bench benchmarks/bench_vector_math.cpp)
    target_link_libraries(lightape_vector_math_bench PRIVATE lightape::lightape)
    # Enables detail::setMathTableForTesting so the harness can time explicit
    # per-function tables (libm / SLEEF ISA / generic) in one process.
    target_compile_definitions(lightape_vector_math_bench PRIVATE LIGHTAPE_VECTOR_MATH_TESTING=1)
    if(_lightape_sleef)
        _lightape_link_sleef(lightape_batch_bench)
        _lightape_link_sleef(lightape_vector_math_bench)
    endif()
    if(NOT MSVC)
        target_compile_options(lightape_bench PRIVATE -O3 -Wall -Wextra -ffp-contract=off)
        # No -march=native/-mavx*: the per-ISA kernels carry their flags in
        # their own TUs, and the direct `std::` reference loops must stay the
        # portable libm they claim to be.
        target_compile_options(lightape_batch_bench PRIVATE -O3 -Wall -Wextra -ffp-contract=off)
        # -fno-builtin: the direct `std::` reference loops must not be
        # re-lowered into `__sincos`/vector builtins, or the libm ULP column
        # would report LLVM's fusion, not the table (same reason as the test).
        target_compile_options(lightape_vector_math_bench PRIVATE -O3 -Wall -Wextra
                                                                 -ffp-contract=off -fno-builtin)
    endif()
endif()
