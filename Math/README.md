# Math Library - AD-Compatible Numerical Methods

A template-based numerical methods library supporting both regular floating-point computation (`double`) and automatic differentiation (`stan::math::var`).

## Overview

This library provides comprehensive numerical integration and root-finding methods designed to work seamlessly with automatic differentiation (AD) frameworks. The implementation is inspired by QuantLib's design patterns but fully templated for AD compatibility.

## Library Structure

```
Math/
├── Integrals/
│   ├── Integrator.h              # Base integrator class template
│   ├── TrapezoidIntegrator.h     # Trapezoid rule with adaptive refinement
│   ├── SimpsonIntegrator.h       # Simpson's rule (Richardson extrapolation)
│   └── GaussianQuadrature.h      # Gauss-Legendre quadrature
│
├── Solvers/
│   ├── Solver1DBase.h            # Base solver class with CRTP pattern
│   ├── BisectionSolver.h         # Bisection method (most robust)
│   ├── SecantSolver.h            # Secant method (no derivatives)
│   ├── NewtonSolver.h            # Newton-Raphson (AD-compatible)
│   ├── BrentSolver.h             # Brent's method (hybrid approach)
│   ├── RidderSolver.h            # Ridder's exponential formula
│   └── FalsePositionSolver.h     # False position (regula falsi)
│
└── NumericalMethods.h            # Main convenience header
```

## Features

### Integration Methods

| Method | Order | Convergence | Best For |
|--------|-------|-------------|----------|
| **Trapezoid** | O(h²) | Adaptive | General purpose |
| **Simpson** | O(h⁴) | Adaptive | Smooth functions |
| **Gauss-Legendre** | 2n-1 polynomial | Fixed order | High accuracy, few evaluations |

**Performance from tests:**
- Trapezoid: ~10⁻⁹ error with ~8K evaluations
- Simpson: Machine precision with ~65 evaluations
- Gauss-Legendre (order 20): ~10⁻¹⁶ error with 20 evaluations

### Root Finding Methods

| Method | Convergence | Derivatives | Robustness | Best For |
|--------|-------------|-------------|------------|----------|
| **Bisection** | Linear | No | Highest | Guaranteed convergence |
| **Secant** | ~1.618 | No | Good | Fast without derivatives |
| **Newton** | Quadratic | Yes | Medium | When derivatives available |
| **Brent** | Super-linear | No | High | Best general-purpose choice |
| **Ridder** | ~1.839 | No | High | Faster than bisection |
| **False Position** | Super-linear | No | High | Alternative to bisection |

**Performance from tests (finding √2):**
- Bisection: ~10⁻¹¹ error
- Secant: ~10⁻¹⁵ error
- Newton: Exact (0 error)
- Brent: ~10⁻¹¹ error
- Ridder: ~10⁻¹⁶ error
- False Position: ~10⁻¹⁵ error

## Usage

### Basic Integration

```cpp
#include "Math/NumericalMethods.h"

using namespace Math;

// Define function to integrate
auto f = [](double x) { return x * x; };

// Method 1: Trapezoid integration
TrapezoidIntegratorDefault<double> trap(1e-8, 1000);
double result = trap(f, 0.0, 1.0);  // ∫₀¹ x² dx = 1/3

// Method 2: Simpson's rule (higher order)
SimpsonIntegrator<double> simpson(1e-8, 1000);
result = simpson(f, 0.0, 1.0);

// Method 3: Gauss-Legendre (best accuracy)
GaussLegendreIntegrator<double> gauss(20);  // order 20
result = gauss(f, 0.0, 1.0);
```

### Root Finding

```cpp
auto f = [](double x) { return x * x - 2.0; };  // Find √2

// Simple bisection (most robust)
BisectionSolver<double> bisection;
double root = bisection.solve(f, 1e-10, 1.5, 0.0, 3.0);

// Brent's method (best general-purpose)
BrentSolver<double> brent;
root = brent.solve(f, 1e-10, 1.5, 0.0, 3.0);

// Newton with explicit derivative (fastest when available)
auto df = [](double x) { return 2.0 * x; };
NewtonSolverWithDerivative<double> newton;
newton.setDerivative(df);
root = newton.solve(f, 1e-10, 1.5, 0.0, 3.0);

// Automatic bracketing
root = brent.solve(f, 1e-10, 1.5, 0.1);  // Auto-bracket from guess
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
SimpsonIntegrator<ADVariableT> integrator(1e-6, 1000);
ADVariableT result = integrator(f_ad, ADVariableT(0.0), ADVariableT(1.0));

// Root finding with AD
auto g_ad = [](ADVariableT x) { return x * x - ADVariableT(2.0); };
BrentSolver<ADVariableT> solver;
ADVariableT root = solver.solve(g_ad, 1e-10,
                                ADVariableT(1.5),
                                ADVariableT(0.0),
                                ADVariableT(3.0));

// Newton with AD (automatic derivatives!)
NewtonSolver<ADVariableT> newton_ad;
root = newton_ad.solve(g_ad, 1e-10,
                       ADVariableT(1.5),
                       ADVariableT(0.0),
                       ADVariableT(3.0));
```

## Design Patterns

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

// Convenient typedefs
TrapezoidIntegratorDefault<double> trap1(1e-8, 1000);
TrapezoidIntegratorMidPoint<double> trap2(1e-8, 1000);
```

### CRTP for Solvers

The `Solver1D` base class uses the Curiously Recurring Template Pattern (CRTP) for static polymorphism:

```cpp
template<typename DoubleT, typename Impl>
class Solver1D { ... };

template<typename DoubleT>
class BrentSolver : public Solver1D<DoubleT, BrentSolver<DoubleT>> { ... };
```

This provides zero-overhead abstraction - as efficient as hand-coded implementations.

## Components

### Integrals

**Integrator.h**
- Base class for all integrators
- Accuracy and max evaluations control
- Evaluation counting and error reporting

**TrapezoidIntegrator.h**
- Adaptive trapezoid integration
- Policy-based refinement strategies (Default, MidPoint)
- Convergence detection
- Second-order accuracy (O(h²))

**SimpsonIntegrator.h**
- Simpson's 1/3 rule via Richardson extrapolation
- Fourth-order accuracy (O(h⁴))
- Formula: `(4*Trap_fine - Trap_coarse) / 3`
- Excellent for smooth functions

**GaussianQuadrature.h**
- Gauss-Legendre quadrature (orders 2, 3, 4, 5, 6, 10, 20)
- Exact for polynomials up to degree 2n-1
- Tabulated weights and abscissas
- Domain transformation for arbitrary intervals
- Best accuracy with minimal evaluations

### Solvers

**Solver1DBase.h**
- CRTP base class for all 1D solvers
- Automatic bracketing support
- Bounded search domains
- Common interface for all derived solvers

**BisectionSolver.h**
- Simplest and most robust method
- Guaranteed convergence for continuous functions
- Linear convergence (slow but sure)
- Use when robustness is critical

**SecantSolver.h**
- No derivatives required
- Super-linear convergence (~1.618)
- Faster than bisection
- May fail to converge in pathological cases

**NewtonSolver.h**
- Quadratic convergence near root
- Two variants:
  - `NewtonSolver`: Uses finite differences
  - `NewtonSolverWithDerivative`: Uses explicit derivative function
- Perfect for AD (automatic derivatives)
- Falls back to bisection if jumps outside brackets

**BrentSolver.h**
- Hybrid: bisection + secant + inverse quadratic interpolation
- Guaranteed convergence (like bisection)
- Super-linear convergence (like secant)
- **Recommended general-purpose solver**

**RidderSolver.h**
- Exponential formula method
- Convergence order ~1.839
- More robust than secant
- Good alternative to Brent

**FalsePositionSolver.h**
- Linear interpolation (regula falsi)
- Super-linear convergence
- Alternative to bisection
- Can be slow if one endpoint becomes "stuck"

## Performance Comparison

From `test_math.cpp` results:

### Integration: ∫₀¹ x² dx = 1/3

| Method | Error | Evaluations |
|--------|-------|-------------|
| Trapezoid (Default) | 2.5×10⁻⁹ | 8,193 |
| Trapezoid (MidPoint) | 2.1×10⁻⁹ | 129,140,164 |
| **Simpson** | **0 (exact)** | **65** |
| Gauss-Legendre (order 2) | 5.6×10⁻¹⁷ | 2 |
| Gauss-Legendre (order 20) | 5.6×10⁻¹⁷ | 20 |

**Winner:** Simpson's rule provides exact results with minimal evaluations for polynomial integrands.

### Root Finding: x² - 2 = 0 (√2)

| Method | Error | Convergence |
|--------|-------|-------------|
| Bisection | 7.6×10⁻¹¹ | Linear |
| Secant | 2.2×10⁻¹⁵ | ~1.618 |
| **Newton** | **0 (exact)** | **Quadratic** |
| Brent | 1.1×10⁻¹¹ | Super-linear |
| Ridder | 8.9×10⁻¹⁶ | ~1.839 |
| False Position | 2.2×10⁻¹⁵ | Super-linear |

**Winner:** Newton with exact derivatives achieves machine precision instantly.

## Testing

Run the comprehensive test suite:

```bash
cmake --build build --target test_math
./build/test_math
```

The test suite validates:
- Polynomial integration (exact for Gauss quadrature)
- Transcendental function integration (sin, exp)
- Root finding for various function types (polynomial, transcendental)
- Domain transformation correctness
- All solvers with different initial guesses
- Automatic bracketing
- Newton with explicit derivatives

## Integration with Stan Math

To enable full AD support:

1. Uncomment in `NumericalMethods.h`:
```cpp
#include <stan/math.hpp>
using ADVariableT = stan::math::var;
```

2. Link with Stan Math in your CMakeLists.txt (already configured in this project)

3. Use `ADVariableT` wherever you would use `double`

## Future Extensions

Planned additions for future releases:

### Integrals
- **Gauss-Kronrod** - adaptive Gaussian quadrature with error estimation
- **Romberg integration** - Richardson extrapolation on trapezoid rule
- **Adaptive Simpson** - error-controlled Simpson's rule
- **2D/3D integration** - multi-dimensional quadrature
- **Oscillatory integrals** - Filon method for ∫f(x)sin(ωx)dx

### Solvers
- **Halley's method** - cubic convergence (requires f'')
- **Muller's method** - finds complex roots
- **Powell's hybrid** - combines methods adaptively
- **Steffensen's method** - quadratic without derivatives

## References

- [QuantLib](https://www.quantlib.org/) - Design inspiration
- [Stan Math](https://mc-stan.org/math/) - Automatic differentiation framework
- Press et al., "Numerical Recipes in C" (2nd ed.) - Algorithm implementations
- Golub & Welsch (1986), "Gauss quadratures and orthogonal polynomials"
- Brent (1973), "Algorithms for Minimization Without Derivatives"

## License

Part of the generic_MC project. See top-level LICENSE file.
