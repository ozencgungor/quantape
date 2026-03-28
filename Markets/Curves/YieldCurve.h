#ifndef YIELDCURVE_H
#define YIELDCURVE_H

#include "Markets/Descriptors/YieldCurveDescriptor.h"
#include "Math/Interpolations/CubicInterpolation.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <memory>

namespace Markets {

/**
 * @brief Yield curve storing zero rates (for dividends, repo, etc.)
 *
 * Template class for yield curves compatible with AD.
 * Semantically different from IRCurve (used for dividends, repo, etc.)
 * but mathematically similar.
 *
 * @tparam DoubleT Numeric type (double or stan::math::var for AD)
 */
template <typename DoubleT>
class YieldCurve {
public:
    /**
     * @brief Constructor with tenors and zero rates
     * @param tenors Time points (in years)
     * @param zeroRates Continuously compounded zero rates (e.g., dividend yields)
     * @param descriptor Curve metadata
     */
    template <typename ContainerT>
    YieldCurve(const ContainerT& tenors, const ContainerT& zeroRates,
               const YieldCurveDescriptor& descriptor = YieldCurveDescriptor())
        : m_descriptor(descriptor) {
        // Convert to vectors
        m_tenors = toVector(tenors);
        m_zeroRates = toVector(zeroRates);

        if (m_tenors.size() != m_zeroRates.size()) {
            throw std::runtime_error("YieldCurve: tenors and zeroRates size mismatch");
        }
        if (m_tenors.size() < 2) {
            throw std::runtime_error("YieldCurve: need at least 2 points");
        }

        // Create interpolator for zero rates
        m_interpolator = std::make_unique<Math::CubicInterpolation<DoubleT>>(
            m_tenors, m_zeroRates, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Copy constructor
     */
    YieldCurve(const YieldCurve& other)
        : m_descriptor(other.m_descriptor),
          m_tenors(other.m_tenors),
          m_zeroRates(other.m_zeroRates) {
        m_interpolator = std::make_unique<Math::CubicInterpolation<DoubleT>>(
            m_tenors, m_zeroRates, Math::CubicInterpolation<DoubleT>::Spline);
    }

    /**
     * @brief Copy assignment
     */
    YieldCurve& operator=(const YieldCurve& other) {
        if (this != &other) {
            m_descriptor = other.m_descriptor;
            m_tenors = other.m_tenors;
            m_zeroRates = other.m_zeroRates;
            m_interpolator = std::make_unique<Math::CubicInterpolation<DoubleT>>(
                m_tenors, m_zeroRates, Math::CubicInterpolation<DoubleT>::Spline);
        }
        return *this;
    }

    /**
     * @brief Get continuously compounded yield at time t
     * @param t Time in years
     * @param allowExtrapolation Allow extrapolation beyond curve range
     * @return Yield y(t)
     */
    DoubleT yield(DoubleT t, bool allowExtrapolation = true) const {
        return (*m_interpolator)(t, allowExtrapolation);
    }

    /**
     * @brief Alias for yield (for consistency with IRCurve)
     */
    DoubleT zeroRate(DoubleT t, bool allowExtrapolation = true) const {
        return yield(t, allowExtrapolation);
    }

    /**
     * @brief Get discount factor at time t
     * @param t Time in years
     * @param allowExtrapolation Allow extrapolation beyond curve range
     * @return Discount factor DF(t) = exp(-y(t) * t)
     */
    DoubleT discountFactor(DoubleT t, bool allowExtrapolation = true) const {
        DoubleT y = yield(t, allowExtrapolation);
        return exp_impl(-y * t);
    }

    /**
     * @brief Get forward yield between two times
     * @param t1 Start time
     * @param t2 End time
     * @return Forward yield
     */
    DoubleT forwardYield(DoubleT t1, DoubleT t2,
                         bool allowExtrapolation = true) const {
        if (value_impl(t2) <= value_impl(t1)) {
            throw std::runtime_error("YieldCurve::forwardYield: t2 must be > t1");
        }
        DoubleT y1 = yield(t1, allowExtrapolation);
        DoubleT y2 = yield(t2, allowExtrapolation);
        return (y2 * t2 - y1 * t1) / (t2 - t1);
    }

    /**
     * @brief Get forward discount factor between two times
     * @param t1 Start time
     * @param t2 End time
     * @return Forward DF = DF(t2) / DF(t1)
     */
    DoubleT forwardDiscountFactor(DoubleT t1, DoubleT t2,
                                  bool allowExtrapolation = true) const {
        DoubleT df1 = discountFactor(t1, allowExtrapolation);
        DoubleT df2 = discountFactor(t2, allowExtrapolation);
        return df2 / df1;
    }

    /**
     * @brief Get descriptor
     */
    const YieldCurveDescriptor& descriptor() const { return m_descriptor; }

    /**
     * @brief Get tenors
     */
    const std::vector<DoubleT>& tenors() const { return m_tenors; }

    /**
     * @brief Get zero rates/yields
     */
    const std::vector<DoubleT>& yields() const { return m_zeroRates; }

private:
    YieldCurveDescriptor m_descriptor;
    std::vector<DoubleT> m_tenors;
    std::vector<DoubleT> m_zeroRates;
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
     * @brief AD-compatible exponential
     */
    static DoubleT exp_impl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return std::exp(x);
        } else {
            using std::exp;
            return exp(x); // ADL for stan::math::exp
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

#endif // YIELDCURVE_H
