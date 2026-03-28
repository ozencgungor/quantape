# Markets library - Header-only market data infrastructure

# Markets library header files (header-only)
set(MARKETS_HEADERS
    MarketData.h
    # Descriptors
    Descriptors/IRCurveDescriptor.h
    Descriptors/YieldCurveDescriptor.h
    Descriptors/CreditDescriptor.h
    Descriptors/IRVolDescriptor.h
    Descriptors/EQDDescriptor.h
    Descriptors/FXDescriptor.h
    # Curves
    Curves/IRCurve.h
    Curves/YieldCurve.h
    Curves/SurvivalProbabilityCurve.h
    # Volatility
    Volatility/IRVolatility.h
    Volatility/EQDVolatility.h
    Volatility/FXVolatility.h
    # Data
    Data/EQDData.h
    Data/FXRate.h
)

# Add headers to parent scope
set(MARKETS_HEADERS ${MARKETS_HEADERS} PARENT_SCOPE)

# Transform to absolute paths for parent scope
foreach(hdr ${MARKETS_HEADERS})
    list(APPEND MARKETS_HEADERS_ABS ${CMAKE_CURRENT_SOURCE_DIR}/${hdr})
endforeach()

set(MARKETS_HEADERS_ABS ${MARKETS_HEADERS_ABS} PARENT_SCOPE)
