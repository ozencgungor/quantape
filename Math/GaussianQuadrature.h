#ifndef GAUSSIAN_QUADRATURE_H
#define GAUSSIAN_QUADRATURE_H

#include "Integrator.h"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace Math {

/**
 * @brief Base class for Gaussian quadrature integration
 *
 * Gaussian quadrature integrates a function using weighted sums:
 * ∫f(x)dx ≈ Σ w_i * f(x_i)
 *
 * The weights (w_i) and abscissas (x_i) are precomputed based on
 * orthogonal polynomials.
 *
 * @tparam DoubleT Numeric type (double or stan::math::var)
 */
template<typename DoubleT>
class GaussianQuadrature {
public:
    using FunctionType = std::function<DoubleT(DoubleT)>;

    /**
     * @brief Construct quadrature with given order and points
     * @param x Abscissas (quadrature points)
     * @param w Weights corresponding to each point
     */
    GaussianQuadrature(const std::vector<double>& x, const std::vector<double>& w)
        : m_x(x), m_w(w) {
        if (x.size() != w.size()) {
            throw std::invalid_argument("Abscissas and weights must have same size");
        }
    }

    /**
     * @brief Evaluate integral using quadrature rule
     * @param f Function to integrate
     * @return Integral approximation
     */
    DoubleT operator()(const FunctionType& f) const {
        DoubleT sum = DoubleT(0.0);
        for (size_t i = 0; i < m_x.size(); ++i) {
            sum = sum + DoubleT(m_w[i]) * f(DoubleT(m_x[i]));
        }
        return sum;
    }

    /**
     * @brief Integrate from a to b by transforming to standard domain
     * @param f Function to integrate
     * @param a Lower bound
     * @param b Upper bound
     * @return Integral value
     */
    DoubleT integrate(const FunctionType& f, DoubleT a, DoubleT b) const {
        // Transform [-1,1] to [a,b]: x = ((b-a)*t + (b+a))/2
        // dx = (b-a)/2 * dt
        DoubleT scale = (b - a) / DoubleT(2.0);
        DoubleT shift = (b + a) / DoubleT(2.0);

        auto transformed = [&](double t) -> DoubleT {
            DoubleT x = scale * DoubleT(t) + shift;
            return f(x);
        };

        DoubleT sum = DoubleT(0.0);
        for (size_t i = 0; i < m_x.size(); ++i) {
            sum = sum + DoubleT(m_w[i]) * transformed(m_x[i]);
        }

        return scale * sum;
    }

    size_t order() const { return m_x.size(); }
    const std::vector<double>& weights() const { return m_w; }
    const std::vector<double>& abscissas() const { return m_x; }

protected:
    std::vector<double> m_x;  // Abscissas
    std::vector<double> m_w;  // Weights
};

/**
 * @brief Gauss-Legendre quadrature on [-1, 1]
 *
 * Tabulated weights and abscissas for common orders.
 * Weighting function w(x) = 1
 */
template<typename DoubleT>
class GaussLegendreQuadrature : public GaussianQuadrature<DoubleT> {
public:
    explicit GaussLegendreQuadrature(size_t order)
        : GaussianQuadrature<DoubleT>(getAbscissas(order), getWeights(order)) {}

private:
    static std::vector<double> getAbscissas(size_t order) {
        switch (order) {
            case 2:
                return {-0.5773502691896257, 0.5773502691896257};
            case 3:
                return {-0.7745966692414834, 0.0, 0.7745966692414834};
            case 4:
                return {-0.8611363115940526, -0.3399810435848563,
                        0.3399810435848563, 0.8611363115940526};
            case 5:
                return {-0.9061798459386640, -0.5384693101056831, 0.0,
                        0.5384693101056831, 0.9061798459386640};
            case 6:
                return {-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
                        0.2386191860831969, 0.6612093864662645, 0.9324695142031521};
            case 10:
                return {-0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
                        -0.4333953941292472, -0.1488743389816312, 0.1488743389816312,
                        0.4333953941292472, 0.6794095682990244, 0.8650633666889845,
                        0.9739065285171717};
            case 20:
                return {-0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
                        -0.8391169718222188, -0.7463319064601508, -0.6360536807265150,
                        -0.5108670019508271, -0.3737060887154195, -0.2277858511416451,
                        -0.0765265211334973, 0.0765265211334973, 0.2277858511416451,
                        0.3737060887154195, 0.5108670019508271, 0.6360536807265150,
                        0.7463319064601508, 0.8391169718222188, 0.9122344282513259,
                        0.9639719272779138, 0.9931285991850949};
            default:
                throw std::invalid_argument("Gauss-Legendre order " + std::to_string(order) + " not implemented");
        }
    }

    static std::vector<double> getWeights(size_t order) {
        switch (order) {
            case 2:
                return {1.0, 1.0};
            case 3:
                return {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};
            case 4:
                return {0.3478548451374538, 0.6521451548625461,
                        0.6521451548625461, 0.3478548451374538};
            case 5:
                return {0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                        0.4786286704993665, 0.2369268850561891};
            case 6:
                return {0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                        0.4679139345726910, 0.3607615730481386, 0.1713244923791704};
            case 10:
                return {0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
                        0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
                        0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
                        0.0666713443086881};
            case 20:
                return {0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
                        0.0832767415767048, 0.1019301198172404, 0.1181945319615184,
                        0.1316886384491766, 0.1420961093183820, 0.1491729864726037,
                        0.1527533871307258, 0.1527533871307258, 0.1491729864726037,
                        0.1420961093183820, 0.1316886384491766, 0.1181945319615184,
                        0.1019301198172404, 0.0832767415767048, 0.0626720483341091,
                        0.0406014298003869, 0.0176140071391521};
            default:
                throw std::invalid_argument("Gauss-Legendre order " + std::to_string(order) + " not implemented");
        }
    }
};

/**
 * @brief Gauss-Legendre integrator wrapper
 *
 * Wraps GaussLegendreQuadrature to match Integrator interface.
 */
template<typename DoubleT>
class GaussLegendreIntegrator : public Integrator<DoubleT> {
public:
    using FunctionType = typename Integrator<DoubleT>::FunctionType;

    explicit GaussLegendreIntegrator(size_t order)
        : Integrator<DoubleT>(0.0, 1)  // Quadrature doesn't iterate
        , m_quadrature(order) {}

protected:
    DoubleT integrate(const FunctionType& f, DoubleT a, DoubleT b) const override {
        this->setNumberOfEvaluations(m_quadrature.order());
        return m_quadrature.integrate(f, a, b);
    }

private:
    GaussLegendreQuadrature<DoubleT> m_quadrature;
};

} // namespace Math

#endif // GAUSSIAN_QUADRATURE_H
