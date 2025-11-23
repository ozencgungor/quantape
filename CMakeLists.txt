cmake_minimum_required(VERSION 3.20)
project(generic_MC)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Third-party libraries
# ============================================================================

include(FetchContent)

# Set third-party directory
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third-party)

# ----------------------------------------------------------------------------
# Eigen (header-only linear algebra library)
# ----------------------------------------------------------------------------
message(STATUS "Fetching Eigen...")
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    SOURCE_DIR ${THIRD_PARTY_DIR}/eigen
)
FetchContent_MakeAvailable(Eigen3)

# ----------------------------------------------------------------------------
# Boost (try system install first, otherwise download subset)
# ----------------------------------------------------------------------------
message(STATUS "Looking for Boost...")
find_package(Boost 1.75 QUIET)

if(NOT Boost_FOUND)
    message(STATUS "Boost not found on system. Downloading Boost headers...")
    FetchContent_Declare(
        Boost
        URL https://boostorg.jfrog.io/artifactory/main/release/1.84.0/source/boost_1_84_0.tar.gz
        SOURCE_DIR ${THIRD_PARTY_DIR}/boost
    )
    FetchContent_MakeAvailable(Boost)
    set(Boost_INCLUDE_DIRS ${THIRD_PARTY_DIR}/boost)
endif()

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
# Stan Math (statistical modeling library, depends on Eigen and Boost)
# ----------------------------------------------------------------------------
message(STATUS "Fetching Stan Math...")
FetchContent_Declare(
    stan_math
    GIT_REPOSITORY https://github.com/stan-dev/math.git
    GIT_TAG v4.9.0
    GIT_SHALLOW TRUE
    SOURCE_DIR ${THIRD_PARTY_DIR}/stan-math
)
FetchContent_GetProperties(stan_math)
if(NOT stan_math_POPULATED)
    FetchContent_Populate(stan_math)
    # Stan Math is header-only, just add include directory
    set(STAN_MATH_INCLUDE_DIR ${THIRD_PARTY_DIR}/stan-math)
endif()

# ============================================================================
# Include directories
# ============================================================================

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Models)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ScenarioGeneration)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Math)

# Third-party includes
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${STAN_MATH_INCLUDE_DIR})

# ============================================================================
# Project source files
# ============================================================================

# Model source files
set(MODEL_SOURCES
    Models/HestonModel.cpp
    Models/CIRModel.cpp
    Models/CIRPPModel.cpp
    Models/JCIRPPModel.cpp
)

# Scenario generation source files
set(SCENARIO_SOURCES
    ScenarioGeneration/BrownianMotion.cpp
    ScenarioGeneration/MonteCarloSimulator.cpp
    ScenarioGeneration/ModelSimulator.cpp
    ScenarioGeneration/ModelGenerator.cpp
    ScenarioGeneration/EquitySimulator.cpp
    ScenarioGeneration/EquityGenerator.cpp
    ScenarioGeneration/CreditSimulator.cpp
    ScenarioGeneration/CreditGenerator.cpp
)

# Model header files
set(MODEL_HEADERS
    Models/HestonModel.h
    Models/CIRModel.h
    Models/GBMModel.h
    Models/CIRPPModel.h
    Models/JCIRPPModel.h
)

# Scenario generation header files
set(SCENARIO_HEADERS
    ScenarioGeneration/BrownianMotion.h
    ScenarioGeneration/MonteCarloSimulator.h
    ScenarioGeneration/ModelSimulator.h
    ScenarioGeneration/ModelGenerator.h
    ScenarioGeneration/EquitySimulator.h
    ScenarioGeneration/EquityGenerator.h
    ScenarioGeneration/CreditSimulator.h
    ScenarioGeneration/CreditGenerator.h
)

# Math library header files (header-only)
set(MATH_HEADERS
    Math/NumericalMethods.h
    Math/Integrator.h
    Math/TrapezoidIntegrator.h
    Math/GaussianQuadrature.h
    Math/Solver1D.h
)

# Combine all sources and headers
set(LIB_SOURCES ${MODEL_SOURCES} ${SCENARIO_SOURCES})
set(LIB_HEADERS ${MODEL_HEADERS} ${SCENARIO_HEADERS} ${MATH_HEADERS})

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
    ${Boost_INCLUDE_DIRS}
    ${STAN_MATH_INCLUDE_DIR}
)

# Link third-party libraries
target_link_libraries(generic_MC_lib PUBLIC
    Eigen3::Eigen
    # dlib::dlib  # Disabled due to C++20 compatibility issues
)

# ============================================================================
# Example executables
# ============================================================================

# Create executable for example usage
add_executable(example_usage example_usage.cpp)
target_link_libraries(example_usage generic_MC_lib)

# Create executable for Math library tests
add_executable(test_math test_math.cpp)
target_link_libraries(test_math generic_MC_lib)

# ============================================================================
# Compiler options
# ============================================================================

# Enable warnings
if(MSVC)
    target_compile_options(generic_MC_lib PRIVATE /W4)
    target_compile_options(example_usage PRIVATE /W4)
    target_compile_options(test_math PRIVATE /W4)
else()
    target_compile_options(generic_MC_lib PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(example_usage PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_math PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ============================================================================
# Installation
# ============================================================================

message(STATUS "===========================================")
message(STATUS "Third-party libraries configuration:")
message(STATUS "  Eigen:      ${THIRD_PARTY_DIR}/eigen")
message(STATUS "  Boost:      ${Boost_INCLUDE_DIRS}")
message(STATUS "  dlib:       ${THIRD_PARTY_DIR}/dlib")
message(STATUS "  Stan Math:  ${STAN_MATH_INCLUDE_DIR}")
message(STATUS "===========================================")
