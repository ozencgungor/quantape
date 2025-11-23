# Math library - Header-only numerical methods and interpolation

# Math library header files (header-only)
set(MATH_HEADERS
    NumericalMethods.h
    Interpolations.h
    Integrals/Integrator.h
    Integrals/TrapezoidIntegrator.h
    Integrals/SimpsonIntegrator.h
    Integrals/GaussianQuadrature.h
    Integrals/GaussLobattoIntegrator.h
    Solvers/Solver1DBase.h
    Solvers/BisectionSolver.h
    Solvers/BrentSolver.h
    Solvers/SecantSolver.h
    Solvers/NewtonSolver.h
    Solvers/RidderSolver.h
    Solvers/FalsePositionSolver.h
    Solvers/TridiagonalSolver.h
    Interpolations/Interpolation.h
    Interpolations/LinearInterpolation.h
    Interpolations/LogLinearInterpolation.h
    Interpolations/CubicInterpolation.h
    Interpolations/Interpolation2D.h
    Interpolations/BilinearInterpolation.h
    Interpolations/BicubicInterpolation.h
)

# Add headers to parent scope
set(MATH_HEADERS ${MATH_HEADERS} PARENT_SCOPE)

# Transform to absolute paths for parent scope
foreach(hdr ${MATH_HEADERS})
    list(APPEND MATH_HEADERS_ABS ${CMAKE_CURRENT_SOURCE_DIR}/${hdr})
endforeach()

set(MATH_HEADERS_ABS ${MATH_HEADERS_ABS} PARENT_SCOPE)
