# ============================================================================
# Third-party dependencies
#
# Header-only libraries live under third-party/ (downloaded once). TBB and
# SUNDIALS are fetched and configured by FetchContent, required by Stan Math.
# ============================================================================

set(THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/third-party)

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
# Stan Math (statistical modeling library, depends on Eigen, Boost, TBB, and
# SUNDIALS)
# ----------------------------------------------------------------------------
set(STAN_MATH_INCLUDE_DIR ${THIRD_PARTY_DIR}/stan-math/math-4.9.0)
if(NOT EXISTS "${STAN_MATH_INCLUDE_DIR}/stan/math.hpp")
    # layout changed: the checkout root is directly under stan-math/
    set(STAN_MATH_INCLUDE_DIR ${THIRD_PARTY_DIR}/stan-math)
endif()
include_directories(${STAN_MATH_INCLUDE_DIR})

# Include SUNDIALS headers from the build
if(TARGET SUNDIALS::generic)
    get_target_property(SUNDIALS_INCLUDE_DIRS SUNDIALS::generic INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(${SUNDIALS_INCLUDE_DIRS})
    # Also include the source directories
    include_directories(${CMAKE_BINARY_DIR}/_deps/sundials-src/include)
    include_directories(${CMAKE_BINARY_DIR}/_deps/sundials-build/include)
endif()

# ----------------------------------------------------------------------------
# quill (header-only low-latency asynchronous logging; bundles fmt)
# Only fetched when logging is enabled; zmij is always needed.
# ----------------------------------------------------------------------------
include(FetchContent)
if(QUANTAPE_ENABLE_LOGGING)
    set(QUILL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(QUILL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(QUILL_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(QUILL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(QUILL_DISABLE_NON_PREFIXED_MACROS ON CACHE BOOL "" FORCE)
    FetchContent_Declare(quill
        URL https://github.com/odygrd/quill/archive/refs/tags/v13.0.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(quill)
endif()

# ----------------------------------------------------------------------------
# zmij (fast float-to-string conversion)
# ----------------------------------------------------------------------------
set(ZMIJ_TEST OFF CACHE BOOL "" FORCE)
set(ZMIJ_EXAMPLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(zmij
    URL https://github.com/vitaut/zmij/archive/refs/tags/v1.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(zmij)
# AppleClang rejects zmij's C++14 relaxed-constexpr table generation
# ("read of element of array without known bound"); zmij's documented opt-out
# costs a little table-init time at first use.
target_compile_definitions(zmij PUBLIC ZMIJ_USE_CONSTEXPR=0)

find_package(Threads REQUIRED)
