#ifndef LOG_LINEAR_INTERPOLATION_H
#define LOG_LINEAR_INTERPOLATION_H

#include <cmath>

#include "Interpolation.h"

namespace Math {

template <typename DoubleT>
class LogLinearInterpolation : public Interpolation<DoubleT> {
public:
    /**
     * @brief Construct log-linear interpolation
     * @param x X coordinates (accepts std::vector, Eigen::Vector, etc.)
     * @param y Y coordinates (accepts std::vector, Eigen::Vector, etc.)
     */
    template <typename ContainerX, typename ContainerY>
    LogLinearInterpolation(const ContainerX& x, const ContainerY& y) {
        this->m_x = this->toVector(x);
        this->m_y = this->toVector(y);
        this->validate();
    }

private:
    // AD-compatible log function
    static DoubleT log_impl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return std::log(x);
        } else {
            using std::log;
            return log(x); // ADL will find stan::math::log
        }
    }

    // AD-compatible exp function
    static DoubleT exp_impl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return std::exp(x);
        } else {
            using std::exp;
            return exp(x); // ADL will find stan::math::exp
        }
    }

    DoubleT valueImpl(DoubleT x) const override {
        size_t i = this->locate(x);
        DoubleT x1 = this->m_x[i];
        DoubleT y1 = this->m_y[i];
        DoubleT x2 = this->m_x[i + 1];
        DoubleT y2 = this->m_y[i + 1];

        DoubleT log_y1 = log_impl(y1);
        DoubleT log_y2 = log_impl(y2);
        DoubleT log_result = log_y1 + (log_y2 - log_y1) * (x - x1) / (x2 - x1);
        return exp_impl(log_result);
    }

    DoubleT derivativeImpl(DoubleT x) const override {
        size_t i = this->locate(x);
        DoubleT x1 = this->m_x[i];
        DoubleT y1 = this->m_y[i];
        DoubleT x2 = this->m_x[i + 1];
        DoubleT y2 = this->m_y[i + 1];

        DoubleT val = valueImpl(x);
        DoubleT log_y1 = log_impl(y1);
        DoubleT log_y2 = log_impl(y2);
        return val * (log_y2 - log_y1) / (x2 - x1);
    }
};

} // namespace Math

#endif // LOG_LINEAR_INTERPOLATION_H
