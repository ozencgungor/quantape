# Math library - Header-only numerical methods and interpolation

# Math library header files (header-only)
set(MATH_HEADERS
    NumericalMethods.h
    Integrals.h
    Interpolations.h
    Optimization.h
    Solvers.h
    StanPrimitives.h
    Autodiff/Hvp.h
    Autodiff/PrimalExtraction.h
    Integrals/Integrator.h
    Integrals/TrapezoidIntegrator.h
    Integrals/SimpsonIntegrator.h
    Integrals/GaussianQuadrature.h
    Integrals/GaussLobattoIntegrator.h
    Solvers/Solver1DBase.h
    Solvers/SolverPrimitives.h
    Solvers/SolverStanPrimitives.h
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
    Optimization/OptimizerPrimitives.h
    Optimization/OptimizerStanPrimitives.h
    Optimization/LineSearch.h
    Optimization/LBFGS.h
    Optimization/Constraint.h
    Optimization/QpSolver.h
    Optimization/SLSQP.h
    Optimization/AugLag.h
    Optimization/TNewton.h
    Random/PCGRandom.hpp
    Random/PCGExtras.hpp
    Random/PCGUint128.hpp
    Random/ZigguratNormal.h
    Random/McFarlandNormal.h
    Random/Sobol/CBCSearch.h
    Random/Sobol/DirectionNumbers.h
    Random/Sobol/GF2.h
)

# Add headers to parent scope
set(MATH_HEADERS ${MATH_HEADERS} PARENT_SCOPE)

# Transform to absolute paths for parent scope
foreach(hdr ${MATH_HEADERS})
    list(APPEND MATH_HEADERS_ABS ${CMAKE_CURRENT_SOURCE_DIR}/${hdr})
endforeach()

set(MATH_HEADERS_ABS ${MATH_HEADERS_ABS} PARENT_SCOPE)
