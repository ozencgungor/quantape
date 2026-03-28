#ifndef SURVIVALPROBABILITYCURVE_H
#define SURVIVALPROBABILITYCURVE_H

#include "Markets/Descriptors/CreditDescriptor.h"
#include "Math/Interpolations/CubicInterpolation.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <memory>

namespace Markets {

/**
 * @brief Survival probability curve for credit risk
 *
 * Template class for credit curves compatible with AD.
 * Stores survival probabilities and provides default probabilities and hazard rates.
 *
 * @tparam DoubleT Numeric type (double or stan::math::var for AD)
 */
template <typename DoubleT>
class SurvivalProbabilityCurve {
public:
    /**
     * @brief Constructor with tenors and survival probabilities
     * @param tenors Time points (in years)
     * @param survivalProbs Survival probabilities (between 0 and 1)
     * @param descriptor Curve metadata
     */
    template <typename ContainerT>
    SurvivalProbabilityCurve(const ContainerT& tenors,
                             const ContainerT& survivalProbs,
                             const CreditDescriptor& descriptor = CreditDescriptor())
        : m_descriptor(descriptor) {
        // Convert to vectors
        m_tenors = toVector(tenors);
        m_survivalProbs = toVector(survivalProbs);

        if (m_tenors.size() != m_survivalProbs.size()) {
            throw std::runtime_error(
                "SurvivalProbabilityCurve: tenors and survivalProbs size mismatch");
        }
        if (m_tenors.size() < 2) {
            throw std::runtime_error(
                "SurvivalProbabilityCurve: need at least 2 points");
        }

        // Validate survival probabilities are in valid range
        for (size_t i = 0; i < m_survivalProbs.size(); ++i) {
            double sp = value_impl(m_survivalProbs[i]);
            if (sp < 0.0 || sp > 1.0) {
                throw std::runtime_error(
                    "SurvivalProbabilityCurve: survival probability must be in [0,1]");
            }
        }

        // Create interpolator for survival probabilities
        m_interpolator = std::make_unique<Math::CubicInterpolation<DoubleT>>(
            m_tenors, m_survivalProbs, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Copy constructor
     */
    SurvivalProbabilityCurve(const SurvivalProbabilityCurve& other)
        : m_descriptor(other.m_descriptor),
          m_tenors(other.m_tenors),
          m_survivalProbs(other.m_survivalProbs) {
        m_interpolator = std::make_unique<Math::CubicInterpolation<DoubleT>>(
            m_tenors, m_survivalProbs, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Copy assignment
     */
    SurvivalProbabilityCurve& operator=(const SurvivalProbabilityCurve& other) {
        if (this != &other) {
            m_descriptor = other.m_descriptor;
            m_tenors = other.m_tenors;
            m_survivalProbs = other.m_survivalProbs;
            m_interpolator = std::make_unique<Math::CubicInterpolation<DoubleT>>(
                m_tenors, m_survivalProbs, Math::CubicInterpolation<DoubleT>::Spline);
        }
        return *this;
    }

    /**
     * @brief Get survival probability at time t
     * @param t Time in years
     * @param allowExtrapolation Allow extrapolation beyond curve range
     * @return Survival probability S(t)
     */
    DoubleT survivalProb(DoubleT t, bool allowExtrapolation = true) const {
        return (*m_interpolator)(t, allowExtrapolation);
    }

    /**
     * @brief Get default probability at time t
     * @param t Time in years
     * @param allowExtrapolation Allow extrapolation beyond curve range
     * @return Default probability PD(t) = 1 - S(t)
     */
    DoubleT defaultProb(DoubleT t, bool allowExtrapolation = true) const {
        return DoubleT(1.0) - survivalProb(t, allowExtrapolation);
    }

    /**
     * @brief Get instantaneous hazard rate at time t
     * @param t Time in years
     * @param allowExtrapolation Allow extrapolation beyond curve range
     * @return Hazard rate h(t) = -d/dt[ln(S(t))] = -S'(t)/S(t)
     *
     * Note: Computed using finite difference approximation
     */
    DoubleT hazardRate(DoubleT t, bool allowExtrapolation = true) const {
        const double dt = 1e-6; // Small time step for numerical derivative
        DoubleT sp = survivalProb(t, allowExtrapolation);
        DoubleT sp_plus = survivalProb(t + DoubleT(dt), allowExtrapolation);

        // h(t) ≈ -[S(t+dt) - S(t)] / [S(t) * dt]
        DoubleT dS = sp_plus - sp;
        return -dS / (sp * DoubleT(dt));
    }

    /**
     * @brief Get average hazard rate from 0 to t
     * @param t Time in years
     * @return Average hazard rate λ(t) = -ln(S(t)) / t
     */
    DoubleT avgHazardRate(DoubleT t, bool allowExtrapolation = true) const {
        if (value_impl(t) <= 0.0) {
            throw std::runtime_error(
                "SurvivalProbabilityCurve::avgHazardRate: t must be positive");
        }
        DoubleT sp = survivalProb(t, allowExtrapolation);
        return -log_impl(sp) / t;
    }

    /**
     * @brief Get descriptor
     */
    const CreditDescriptor& descriptor() const { return m_descriptor; }

    /**
     * @brief Get tenors
     */
    const std::vector<DoubleT>& tenors() const { return m_tenors; }

    /**
     * @brief Get survival probabilities
     */
    const std::vector<DoubleT>& survivalProbs() const { return m_survivalProbs; }

private:
    CreditDescriptor m_descriptor;
    std::vector<DoubleT> m_tenors;
    std::vector<DoubleT> m_survivalProbs;
    std::unique_ptr<Math::CubicInterpolation<DoubleT>> m_interpolator;

    /**
     * @brief Convert container to vector
     */
    template <typename ContainerT>
    std::vector<DoubleT> toVector(const ContainerT& container) const {
        std::vector<DoubleT> result;
        result.reserve(container.size());
        for (const auto& val : container) {
            result.push_back(static_cast<DoubleT>(val));
        }
        return result;
    }

    /**
     * @brief AD-compatible logarithm
     */
    static DoubleT log_impl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return std::log(x);
        } else {
            using std::log;
            return log(x); // ADL for stan::math::log
        }
    }

    /**
     * @brief Extract value for comparison (handles both double and AD types)
     */
    static double value_impl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return x;
        } else {
            return x.val(); // For stan::math::var
        }
    }
};

} // namespace Markets

#endif // SURVIVALPROBABILITYCURVE_H
