# Math Library - AD-Compatible Numerical Methods

A template-based numerical methods library supporting both regular floating-point computation (`double`) and automatic differentiation (`stan::math::var`).

## Overview

This library provides numerical integration and root-finding methods designed to work seamlessly with automatic differentiation (AD) frameworks. The implementation is inspired by QuantLib's design patterns but fully templated for AD compatibility.

## Features

### Integration Methods

- **Trapezoid Integration** with adaptive refinement
  - Default policy: doubles intervals per refinement
  - MidPoint policy: triples intervals per refinement

- **Gaussian Quadrature** for highly accurate integration
  - Gauss-Legendre quadrature (orders 2, 3, 4, 5, 6, 10, 20)
  - Exact for polynomials up to degree 2n-1 (for order n)

### Root Finding

- **Brent's Method** combining:
  - Bisection (robust bracketing)
  - Secant method (fast convergence)
  - Inverse quadratic interpolation (super-linear convergence)

- Automatic bracketing support

## Usage

### Basic Integration

```cpp
#include "Math/NumericalMethods.h"

using namespace Math;

// Define function to integrate
auto f = [](double x) { return x * x; };

// Trapezoid integration
TrapezoidIntegratorDefault<double> integrator(1e-8, 1000);
double result = integrator(f, 0.0, 1.0);  // ∫₀¹ x² dx = 1/3

// Gauss-Legendre integration (order 20)
GaussLegendreIntegrator<double> gauss(20);
double result_gauss = gauss(f, 0.0, 1.0);
```

### Root Finding

```cpp
// Find root of f(x) = x² - 2 (i.e., sqrt(2))
auto f = [](double x) { return x * x - 2.0; };

BrentSolver<double> solver;
solver.setMaxEvaluations(100);

// With explicit bracket [0, 3] and initial guess 1.5
double root = solver.solve(f, 1e-10, 1.5, 0.0, 3.0);

// With automatic bracketing from guess
double root2 = solver.solve(f, 1e-10, 1.5, 0.1);  // step size 0.1
```

### Automatic Differentiation Support

The library is designed to work with AD types like `stan::math::var`:

```cpp
#include <stan/math.hpp>
#include "Math/NumericalMethods.h"

using ADVariableT = stan::math::var;
using namespace Math;

// AD-compatible function
auto f_ad = [](ADVariableT x) { return x * x; };

// Integration with AD
TrapezoidIntegratorDefault<ADVariableT> integrator(1e-6, 1000);
ADVariableT result = integrator(f_ad, ADVariableT(0.0), ADVariableT(1.0));

// Root finding with AD
auto g_ad = [](ADVariableT x) { return x * x - ADVariableT(2.0); };
BrentSolver<ADVariableT> solver;
ADVariableT root = solver.solve(g_ad, 1e-10,
                                ADVariableT(1.5),
                                ADVariableT(0.0),
                                ADVariableT(3.0));
```

## Design Pattern

### Template Parameter `DoubleT`

All classes are templated on the numeric type `DoubleT`:

- `DoubleT = double` for regular computation
- `DoubleT = stan::math::var` for automatic differentiation
- Can be extended to other numeric types (e.g., `mpfr::mpreal`, `boost::multiprecision`)

### Policy-Based Integration

The `TrapezoidIntegrator` uses policy classes to define refinement strategies:

```cpp
// Default policy: N → 2N (doubles intervals)
TrapezoidIntegrator<double, DefaultPolicy> default_integrator(1e-8, 1000);

// MidPoint policy: N → 3N (triples intervals)
TrapezoidIntegrator<double, MidPointPolicy> midpoint_integrator(1e-8, 1000);
```

### CRTP for Solvers

The `Solver1D` base class uses the Curiously Recurring Template Pattern (CRTP) for static polymorphism:

```cpp
template<typename DoubleT, typename Impl>
class Solver1D { ... };

template<typename DoubleT>
class BrentSolver : public Solver1D<DoubleT, BrentSolver<DoubleT>> { ... };
```

## Components

### Integrator.h
Base class for all integrators with:
- Accuracy and max evaluations control
- Evaluation counting
- Error reporting

### TrapezoidIntegrator.h
Adaptive trapezoid integration with:
- Policy-based refinement strategies
- Convergence detection
- Customizable policies

### GaussianQuadrature.h
Gaussian quadrature integration with:
- Tabulated weights and abscissas for Gauss-Legendre
- Domain transformation for arbitrary intervals
- High accuracy with minimal function evaluations

### Solver1D.h
1D root finding with:
- Brent's method implementation
- Automatic bracketing
- Bounded search domains
- Hybrid bisection/interpolation approach

## Performance

From test results (see `test_math.cpp`):

### Integration Accuracy

- **Trapezoid (Default)**: ~10⁻⁹ error with ~8K evaluations
- **Gauss-Legendre (order 20)**: ~10⁻¹⁶ error with 20 evaluations

### Root Finding Convergence

- **Brent's method**: ~10⁻¹¹ error in typical cases
- Converges quadratically near the root
- Guaranteed to bracket (unlike pure Newton methods)

## Testing

Run the test suite:

```bash
cmake --build build --target test_math
./build/test_math
```

The test suite validates:
- Polynomial integration (exact for Gauss quadrature)
- Transcendental function integration (sin, exp)
- Root finding for various function types
- Domain transformation correctness

## Future Extensions

Planned additions for AD compatibility:

1. **Newton-Raphson solver** - leverages AD for automatic gradient computation
2. **Simpson's rule integrator** - higher-order accuracy
3. **Multi-dimensional integration** - for multivariate functions
4. **Adaptive Gauss-Kronrod** - adaptive Gaussian quadrature
5. **Romberg integration** - Richardson extrapolation

## References

- QuantLib library: https://www.quantlib.org/
- Stan Math library: https://mc-stan.org/math/
- Numerical Recipes in C (Press et al.)
- Gauss quadratures and orthogonal polynomials (Golub & Welsch, 1986)
