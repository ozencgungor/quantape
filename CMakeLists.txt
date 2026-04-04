cmake_minimum_required(VERSION 3.20)
project(generic_MC)

# Allow subdirectories with older CMake minimum versions
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Third-party libraries
# ============================================================================

# Set third-party directory
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third-party)

# ----------------------------------------------------------------------------
# Eigen (header-only linear algebra library)
# ----------------------------------------------------------------------------
set(EIGEN3_INCLUDE_DIR ${THIRD_PARTY_DIR}/eigen)
include_directories(${EIGEN3_INCLUDE_DIR})

# ----------------------------------------------------------------------------
# Boost (download to third-party directory if not present)
# ----------------------------------------------------------------------------
if(NOT EXISTS ${THIRD_PARTY_DIR}/boost/boost)
    message(STATUS "Downloading Boost headers to third-party...")
    include(FetchContent)
    FetchContent_Declare(
        Boost
        URL https://archives.boost.io/release/1.84.0/source/boost_1_84_0.tar.bz2
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_DIR ${THIRD_PARTY_DIR}/boost
    )
    FetchContent_MakeAvailable(Boost)
else()
    message(STATUS "Using existing Boost installation in third-party/boost")
endif()
set(Boost_INCLUDE_DIRS ${THIRD_PARTY_DIR}/boost)

# ----------------------------------------------------------------------------
# dlib (machine learning library) - DISABLED due to C++20 compatibility issues
# ----------------------------------------------------------------------------
# Uncomment when needed:
# message(STATUS "Fetching dlib...")
# FetchContent_Declare(
#     dlib
#     GIT_REPOSITORY https://github.com/davisking/dlib.git
#     GIT_TAG v19.24.4
#     GIT_SHALLOW TRUE
#     SOURCE_DIR ${THIRD_PARTY_DIR}/dlib
# )
# set(DLIB_NO_GUI_SUPPORT ON CACHE BOOL "Disable dlib GUI support")
# set(DLIB_USE_CUDA OFF CACHE BOOL "Disable CUDA support")
# FetchContent_MakeAvailable(dlib)

# ----------------------------------------------------------------------------
# Intel TBB (Threading Building Blocks, required by Stan Math)
# Download and build oneTBB from source
# ----------------------------------------------------------------------------
# Force TBB to build as a shared library, as recommended for Stan Math
set(BUILD_SHARED_LIBS ON)
if(NOT TARGET TBB::tbb)
    message(STATUS "Downloading and building Intel oneTBB from source...")
    include(FetchContent)

    # Set minimum CMake policy to allow TBB's older CMake version
    set(CMAKE_POLICY_DEFAULT_CMP0000 OLD)

    FetchContent_Declare(
        TBB
        URL https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.9.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    # Configure TBB build options
    set(TBB_TEST OFF CACHE BOOL "Build TBB tests" FORCE)
    set(TBB_EXAMPLES OFF CACHE BOOL "Build TBB examples" FORCE)
    set(TBB_STRICT OFF CACHE BOOL "Treat compiler warnings as errors" FORCE)
    set(TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON CACHE BOOL "Disable hwloc" FORCE)

    FetchContent_MakeAvailable(TBB)

    message(STATUS "TBB built successfully")
    set(TBB_LIBRARIES TBB::tbb)
else()
    message(STATUS "Using existing TBB target")
endif()
# Reset BUILD_SHARED_LIBS to default (OFF for this project) after building TBB
set(BUILD_SHARED_LIBS OFF)

# ----------------------------------------------------------------------------
# SUNDIALS (SUite of Nonlinear and DIfferential/ALgebraic equation Solvers)
# Required by Stan Math for ODE solving capabilities
# ----------------------------------------------------------------------------
if(NOT TARGET SUNDIALS::generic)
    message(STATUS "Downloading and building SUNDIALS...")
    include(FetchContent)
    FetchContent_Declare(
        SUNDIALS
        URL https://github.com/LLNL/sundials/releases/download/v6.6.0/sundials-6.6.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    # Configure SUNDIALS build options
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
    set(BUILD_STATIC_LIBS ON CACHE BOOL "Build static libraries" FORCE)
    set(EXAMPLES_ENABLE_C OFF CACHE BOOL "Build SUNDIALS C examples" FORCE)
    set(EXAMPLES_INSTALL OFF CACHE BOOL "Install SUNDIALS examples" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Build SUNDIALS tests" FORCE)

    FetchContent_MakeAvailable(SUNDIALS)

    message(STATUS "SUNDIALS built successfully")
else()
    message(STATUS "Using existing SUNDIALS target")
endif()

# ----------------------------------------------------------------------------
# Stan Math (statistical modeling library, depends on Eigen, Boost, TBB, and SUNDIALS)
# ----------------------------------------------------------------------------
set(STAN_MATH_INCLUDE_DIR ${THIRD_PARTY_DIR}/stan-math/math-4.9.0)
include_directories(${STAN_MATH_INCLUDE_DIR})

# Include SUNDIALS headers from the build
if(TARGET SUNDIALS::generic)
    get_target_property(SUNDIALS_INCLUDE_DIRS SUNDIALS::generic INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(${SUNDIALS_INCLUDE_DIRS})
    # Also include the source directories
    include_directories(${CMAKE_BINARY_DIR}/_deps/sundials-src/include)
    include_directories(${CMAKE_BINARY_DIR}/_deps/sundials-build/include)
endif()

# ============================================================================
# Include directories
# ============================================================================

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Models)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ScenarioGeneration)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Math)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Markets)

# Third-party includes
include_directories(${Boost_INCLUDE_DIRS})

# ============================================================================
# Project source files
# ============================================================================

# Add subdirectories to gather sources and headers
add_subdirectory(Models)
add_subdirectory(ScenarioGeneration)
add_subdirectory(Math)
add_subdirectory(Markets)

# Combine all sources and headers
set(LIB_SOURCES ${MODEL_SOURCES_ABS} ${SCENARIO_SOURCES_ABS})
set(LIB_HEADERS ${MODEL_HEADERS_ABS} ${SCENARIO_HEADERS_ABS} ${MATH_HEADERS_ABS} ${MARKETS_HEADERS_ABS})

# ============================================================================
# Library target
# ============================================================================

# Create static library
add_library(generic_MC_lib STATIC ${LIB_SOURCES} ${LIB_HEADERS})

# Set include directories for the library
target_include_directories(generic_MC_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Models
    ${CMAKE_CURRENT_SOURCE_DIR}/ScenarioGeneration
    ${CMAKE_CURRENT_SOURCE_DIR}/Math
    ${CMAKE_CURRENT_SOURCE_DIR}/Markets
    ${Boost_INCLUDE_DIRS}
)

# Link third-party libraries
# target_link_libraries(generic_MC_lib PUBLIC
#     Eigen3::Eigen
#     # dlib::dlib  # Disabled due to C++20 compatibility issues
# )

# ============================================================================
# Example executables
# ============================================================================

# Create executable for example usage
add_executable(example_usage example_usage.cpp)
target_link_libraries(example_usage generic_MC_lib)

# Create executable for Math library tests
add_executable(test_math test_math.cpp)
target_link_libraries(test_math generic_MC_lib)

# Create executable for Interpolation library tests
add_executable(test_interpolation test_interpolation.cpp)
target_link_libraries(test_interpolation generic_MC_lib)

# Create executable for Markets library tests
add_executable(test_markets test_markets.cpp)
target_link_libraries(test_markets generic_MC_lib)

# Create executable for CIR bridge accuracy tests
add_executable(test_cir_bridge test_cir_bridge.cpp)
target_link_libraries(test_cir_bridge generic_MC_lib)

# Create executable for CIR bridge distribution tests
add_executable(test_cir_bridge_distribution test_cir_bridge_distribution.cpp)
target_link_libraries(test_cir_bridge_distribution generic_MC_lib)

# Create executable for improved CIR bridge tests
add_executable(test_improved_bridges test_improved_bridges.cpp)
target_link_libraries(test_improved_bridges generic_MC_lib)

# Create executable for theoretical bridge variance tests
add_executable(test_theoretical_bridge_variance test_theoretical_bridge_variance.cpp)
target_link_libraries(test_theoretical_bridge_variance generic_MC_lib)

# Create executable for asymmetric bridge tests
add_executable(test_asymmetric_bridge test_asymmetric_bridge.cpp)
target_link_libraries(test_asymmetric_bridge generic_MC_lib)

# Create executable for analytical bridge variance tests
add_executable(test_analytical_bridge_variance test_analytical_bridge_variance.cpp)
target_link_libraries(test_analytical_bridge_variance generic_MC_lib)

# Create executable for GBM bridge tests
add_executable(test_gbm_bridge test_gbm_bridge.cpp)
target_link_libraries(test_gbm_bridge generic_MC_lib)

# Create executable for G2++ bridge tests
add_executable(test_g2pp_bridge test_g2pp_bridge.cpp)
target_link_libraries(test_g2pp_bridge generic_MC_lib)

# Create executable for analytical bridge tests
add_executable(test_analytical_bridges test_analytical_bridges.cpp)
target_link_libraries(test_analytical_bridges generic_MC_lib)

# Create executable for nonlinear correction tests
add_executable(test_nonlinear_corrections test_nonlinear_corrections.cpp)
target_link_libraries(test_nonlinear_corrections generic_MC_lib)

# Create executable for explaining effective kappa
add_executable(explain_effective_kappa explain_effective_kappa.cpp)

# Create executable for C++ references and pointers guide
add_executable(cpp_references_pointers_guide cpp_references_pointers_guide.cpp)

# Create executable for smart pointers and algorithms implementation
add_executable(smart_pointers_and_algorithms interview_prep/smart_pointers_and_algorithms.cpp)

# Create executable for special member functions demonstration
add_executable(special_member_functions interview_prep/special_member_functions.cpp)

# Create executable for new and delete explanation
add_executable(new_and_delete_explained interview_prep/new_and_delete_explained.cpp)

# Create executable for lambdas and algorithms
add_executable(lambdas_and_algorithms interview_prep/lambdas_and_algorithms.cpp)

# Create executable for advanced C++ topics
add_executable(advanced_cpp_topics interview_prep/advanced_cpp_topics.cpp)

# Create executable for numerical methods
add_executable(numerical_methods interview_prep/numerical_methods.cpp)

# Create executable for numpy patterns demonstration
add_executable(numpy_patterns interview_prep/numpy_patterns.cpp)

# Create executable for Eigen 3.4+ fancy indexing
add_executable(eigen_fancy_indexing interview_prep/eigen_fancy_indexing.cpp)

# Create executable for AD (Automatic Differentiation) tests
add_executable(test_ad test_ad.cpp)

# Link libraries for AD test
target_link_libraries(test_ad PUBLIC
    generic_MC_lib
    TBB::tbb
    SUNDIALS::nvecserial
    SUNDIALS::cvodes
)

# Configure Stan Math compile definitions for AD test
target_compile_definitions(test_ad PRIVATE
    STAN_THREADS=true       # Enable threading via TBB
    STAN_NO_RANGE_CHECKS
    _REENTRANT              # Required for thread-safe math functions
    TBB_INTERFACE_NEW       # Use new oneTBB interface
)

# Set rpath for TBB on macOS and Linux to ensure it's found at runtime
if(APPLE OR UNIX)
    # Set the rpath for the test_ad executable using generator expressions
    set_target_properties(test_ad PROPERTIES
        INSTALL_RPATH "$<TARGET_FILE_DIR:TBB::tbb>"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    # For Linux, also set the rpath to search relative to the executable
    if(UNIX AND NOT APPLE)
        set_target_properties(test_ad PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib:$<TARGET_FILE_DIR:TBB::tbb>"
        )
    endif()
endif()

# Create executable for Black-Scholes AD benchmark
add_executable(test_bs_ad_bench test_bs_ad_bench.cpp)

target_link_libraries(test_bs_ad_bench PUBLIC
    generic_MC_lib
    TBB::tbb
    SUNDIALS::nvecserial
    SUNDIALS::cvodes
)

target_compile_definitions(test_bs_ad_bench PRIVATE
    STAN_THREADS=true
    STAN_NO_RANGE_CHECKS
    _REENTRANT
    TBB_INTERFACE_NEW
)

target_compile_options(test_bs_ad_bench PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native -ffast-math)

if(APPLE OR UNIX)
    set_target_properties(test_bs_ad_bench PROPERTIES
        INSTALL_RPATH "$<TARGET_FILE_DIR:TBB::tbb>"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Create executable for interpolation AD benchmark
add_executable(test_interp_ad_bench test_interp_ad_bench.cpp)

target_link_libraries(test_interp_ad_bench PUBLIC
    generic_MC_lib
    TBB::tbb
    SUNDIALS::nvecserial
    SUNDIALS::cvodes
)

target_compile_definitions(test_interp_ad_bench PRIVATE
    STAN_THREADS=true
    STAN_NO_RANGE_CHECKS
    _REENTRANT
    TBB_INTERFACE_NEW
)

target_compile_options(test_interp_ad_bench PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native -ffast-math)

if(APPLE OR UNIX)
    set_target_properties(test_interp_ad_bench PROPERTIES
        INSTALL_RPATH "$<TARGET_FILE_DIR:TBB::tbb>"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Create executable for fast exp/log AD benchmark
add_executable(test_fast_exp_log test_fast_exp_log.cpp)

target_link_libraries(test_fast_exp_log PUBLIC
    generic_MC_lib
    TBB::tbb
    SUNDIALS::nvecserial
    SUNDIALS::cvodes
)

target_compile_definitions(test_fast_exp_log PRIVATE
    STAN_THREADS=true
    STAN_NO_RANGE_CHECKS
    _REENTRANT
    TBB_INTERFACE_NEW
)

target_compile_options(test_fast_exp_log PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native -ffast-math)

if(APPLE OR UNIX)
    set_target_properties(test_fast_exp_log PROPERTIES
        INSTALL_RPATH "$<TARGET_FILE_DIR:TBB::tbb>"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Create executable for bivariate normal CDF AD benchmark
add_executable(test_bvn_ad_bench test_bvn_ad_bench.cpp)

target_link_libraries(test_bvn_ad_bench PUBLIC
    generic_MC_lib
    TBB::tbb
    SUNDIALS::nvecserial
    SUNDIALS::cvodes
)

target_compile_definitions(test_bvn_ad_bench PRIVATE
    STAN_THREADS=true
    STAN_NO_RANGE_CHECKS
    _REENTRANT
    TBB_INTERFACE_NEW
)

target_compile_options(test_bvn_ad_bench PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native -ffast-math)

if(APPLE OR UNIX)
    set_target_properties(test_bvn_ad_bench PROPERTIES
        INSTALL_RPATH "$<TARGET_FILE_DIR:TBB::tbb>"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()


add_executable(test_bs_hessian_bench test_bs_hessian_bench.cpp)

target_link_libraries(test_bs_hessian_bench PUBLIC
        generic_MC_lib
        TBB::tbb
        SUNDIALS::nvecserial
        SUNDIALS::cvodes
)

target_compile_definitions(test_bs_hessian_bench PRIVATE
        STAN_THREADS=true
        STAN_NO_RANGE_CHECKS
        _REENTRANT
        TBB_INTERFACE_NEW
)

target_compile_options(test_bs_hessian_bench PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)

if(APPLE OR UNIX)
    set_target_properties(test_bs_hessian_bench PROPERTIES
            INSTALL_RPATH "$<TARGET_FILE_DIR:TBB::tbb>"
            BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Create executable for mixed hessian benchmark
add_executable(test_mixed_hessian test_mixed_hessian.cpp)

target_link_libraries(test_mixed_hessian PUBLIC
        generic_MC_lib
        TBB::tbb
        SUNDIALS::nvecserial
        SUNDIALS::cvodes
)

target_compile_definitions(test_mixed_hessian PRIVATE
        STAN_THREADS=true
        STAN_NO_RANGE_CHECKS
        _REENTRANT
        TBB_INTERFACE_NEW
)

target_compile_options(test_mixed_hessian PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)

if(APPLE OR UNIX)
    set_target_properties(test_mixed_hessian PROPERTIES
            INSTALL_RPATH "$<TARGET_FILE_DIR:TBB::tbb>"
            BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# precomputed_gradients vs custom vari example
add_executable(test_precomputed_gradients test_precomputed_gradients.cpp)

target_link_libraries(test_precomputed_gradients PUBLIC
        generic_MC_lib
        TBB::tbb
        SUNDIALS::nvecserial
        SUNDIALS::cvodes
)

target_compile_definitions(test_precomputed_gradients PRIVATE
        STAN_THREADS=true
        STAN_NO_RANGE_CHECKS
        _REENTRANT
        TBB_INTERFACE_NEW
)

target_compile_options(test_precomputed_gradients PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native -ferror-limit=0)

if(APPLE OR UNIX)
    set_target_properties(test_precomputed_gradients PROPERTIES
            INSTALL_RPATH "$<TARGET_FILE_DIR:TBB::tbb>"
            BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Multi-component AD chain example
add_executable(test_chain_ad test_chain_ad.cpp)

target_link_libraries(test_chain_ad PUBLIC
        generic_MC_lib
        TBB::tbb
        SUNDIALS::nvecserial
        SUNDIALS::cvodes
)

target_compile_definitions(test_chain_ad PRIVATE
        STAN_THREADS=true
        STAN_NO_RANGE_CHECKS
        _REENTRANT
        TBB_INTERFACE_NEW
)

target_compile_options(test_chain_ad PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native -ferror-limit=0)

if(APPLE OR UNIX)
    set_target_properties(test_chain_ad PROPERTIES
            INSTALL_RPATH "$<TARGET_FILE_DIR:TBB::tbb>"
            BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Create executable for pricer hierarchy hessian benchmark
add_executable(test_pricer_hessian test_pricer_hessian.cpp)

target_link_libraries(test_pricer_hessian PUBLIC
        generic_MC_lib
        TBB::tbb
        SUNDIALS::nvecserial
        SUNDIALS::cvodes
)

target_compile_definitions(test_pricer_hessian PRIVATE
        STAN_THREADS=true
        STAN_NO_RANGE_CHECKS
        _REENTRANT
        TBB_INTERFACE_NEW
)

target_compile_options(test_pricer_hessian PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)

if(APPLE OR UNIX)
    set_target_properties(test_pricer_hessian PROPERTIES
            INSTALL_RPATH "$<TARGET_FILE_DIR:TBB::tbb>"
            BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Create executable for ziggurat RNG tests
add_executable(test_ziggurat test_ziggurat.cpp)
target_link_libraries(test_ziggurat generic_MC_lib)
target_compile_options(test_ziggurat PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)

# Create executable for McFarland ziggurat tests
add_executable(test_mcfarland test_mcfarland.cpp)
target_link_libraries(test_mcfarland generic_MC_lib)
target_compile_options(test_mcfarland PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)

# ============================================================================
# Compiler options
# ============================================================================

# Enable warnings
if(MSVC)
    target_compile_options(generic_MC_lib PRIVATE /W4)
    target_compile_options(example_usage PRIVATE /W4)
    target_compile_options(test_math PRIVATE /W4)
    target_compile_options(test_interpolation PRIVATE /W4)
    target_compile_options(test_markets PRIVATE /W4)
    target_compile_options(test_gbm_bridge PRIVATE /W4)
    target_compile_options(test_g2pp_bridge PRIVATE /W4)
    target_compile_options(test_nonlinear_corrections PRIVATE /W4)
    target_compile_options(explain_effective_kappa PRIVATE /W4)
    target_compile_options(cpp_references_pointers_guide PRIVATE /W4)
    target_compile_options(smart_pointers_and_algorithms PRIVATE /W4)
    target_compile_options(special_member_functions PRIVATE /W4)
    target_compile_options(new_and_delete_explained PRIVATE /W4)
    target_compile_options(lambdas_and_algorithms PRIVATE /W4)
    target_compile_options(advanced_cpp_topics PRIVATE /W4)
    target_compile_options(numerical_methods PRIVATE /W4)
    target_compile_options(test_ad PRIVATE /W4)
else()
    target_compile_options(generic_MC_lib PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native -Rpass=loop-vectorize -ffast-math)
    target_compile_options(example_usage PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_math PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_interpolation PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_markets PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_gbm_bridge PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_g2pp_bridge PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_nonlinear_corrections PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(explain_effective_kappa PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(cpp_references_pointers_guide PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(smart_pointers_and_algorithms PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(special_member_functions PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(new_and_delete_explained PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(lambdas_and_algorithms PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(advanced_cpp_topics PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(numerical_methods PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(numpy_patterns PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)
    target_compile_options(eigen_fancy_indexing PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)
    target_compile_options(test_ad PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native -Rpass=loop-vectorize -ffast-math)
endif()

# ============================================================================
# Installation
# ============================================================================

message(STATUS "===========================================")
message(STATUS "Third-party libraries configuration:")
message(STATUS "  Eigen:      ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  Boost:      ${Boost_INCLUDE_DIRS}")
message(STATUS "  TBB:        ${TBB_INCLUDE_DIR}")
message(STATUS "  SUNDIALS:   ${SUNDIALS_INCLUDE_DIR}")
message(STATUS "  dlib:       ${THIRD_PARTY_DIR}/dlib")
message(STATUS "  Stan Math:  ${STAN_MATH_INCLUDE_DIR}")
message(STATUS "===========================================")
