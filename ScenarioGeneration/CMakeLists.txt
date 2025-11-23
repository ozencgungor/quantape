# ScenarioGeneration library - Monte Carlo scenario generation

# Scenario generation source files
set(SCENARIO_SOURCES
    BrownianMotion.cpp
    MonteCarloSimulator.cpp
    ModelSimulator.cpp
    ModelGenerator.cpp
    EquitySimulator.cpp
    EquityGenerator.cpp
    CreditSimulator.cpp
    CreditGenerator.cpp
)

# Scenario generation header files
set(SCENARIO_HEADERS
    BrownianMotion.h
    MonteCarloSimulator.h
    ModelSimulator.h
    ModelGenerator.h
    EquitySimulator.h
    EquityGenerator.h
    CreditSimulator.h
    CreditGenerator.h
)

# Add sources to parent scope
set(SCENARIO_SOURCES ${SCENARIO_SOURCES} PARENT_SCOPE)
set(SCENARIO_HEADERS ${SCENARIO_HEADERS} PARENT_SCOPE)

# Transform to absolute paths for parent scope
foreach(src ${SCENARIO_SOURCES})
    list(APPEND SCENARIO_SOURCES_ABS ${CMAKE_CURRENT_SOURCE_DIR}/${src})
endforeach()
foreach(hdr ${SCENARIO_HEADERS})
    list(APPEND SCENARIO_HEADERS_ABS ${CMAKE_CURRENT_SOURCE_DIR}/${hdr})
endforeach()

set(SCENARIO_SOURCES_ABS ${SCENARIO_SOURCES_ABS} PARENT_SCOPE)
set(SCENARIO_HEADERS_ABS ${SCENARIO_HEADERS_ABS} PARENT_SCOPE)
