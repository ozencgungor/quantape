# Models library - Stochastic models for Monte Carlo simulation

# Model source files
set(MODEL_SOURCES
    HestonModel.cpp
    CIRModel.cpp
    CIRPPModel.cpp
    JCIRPPModel.cpp
)

# Model header files
set(MODEL_HEADERS
    HestonModel.h
    CIRModel.h
    GBMModel.h
    CIRPPModel.h
    JCIRPPModel.h
)

# Add sources to parent scope
set(MODEL_SOURCES ${MODEL_SOURCES} PARENT_SCOPE)
set(MODEL_HEADERS ${MODEL_HEADERS} PARENT_SCOPE)

# Transform to absolute paths for parent scope
foreach(src ${MODEL_SOURCES})
    list(APPEND MODEL_SOURCES_ABS ${CMAKE_CURRENT_SOURCE_DIR}/${src})
endforeach()
foreach(hdr ${MODEL_HEADERS})
    list(APPEND MODEL_HEADERS_ABS ${CMAKE_CURRENT_SOURCE_DIR}/${hdr})
endforeach()

set(MODEL_SOURCES_ABS ${MODEL_SOURCES_ABS} PARENT_SCOPE)
set(MODEL_HEADERS_ABS ${MODEL_HEADERS_ABS} PARENT_SCOPE)
