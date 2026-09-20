#ifndef LOG_LINEAR_INTERPOLATION_H
#define LOG_LINEAR_INTERPOLATION_H

#include <cmath>

#include "Interpolation.h"

namespace Math {

/**
 * @brief Log-linear interpolation: linear in log(y), exponential result
 *
 * f(x) = exp( (1-t)*log(y_i) + t*log(y_{i+1}) )
 * where t = (x - x_i) / (x_{i+1} - x_i)
 *
 * This is NOT linear in y, so the Hessian is non-zero.
 * var and fvar<var> specializations are in InterpolationStanPrimitives.h.
 */
template <typename DoubleT>
class LogLinearInterpolation : public Interpolation<DoubleT, LogLinearInterpolation<DoubleT>> {
    using Base = Interpolation<DoubleT, LogLinearInterpolation<DoubleT>>;
    friend Base;

public:
    template <typename ContainerX, typename ContainerY>
    LogLinearInterpolation(const ContainerX& x, const ContainerY& y) {
        this->m_x = this->toDoubleVector(x);
        this->m_y = this->toVector(y);
        this->validate();
    }

    DoubleT valueImpl(DoubleT x) const {
        size_t i = this->locate(x);

        // Grid coordinates are double — off tape
        double x1 = this->m_x[i];
        double x2 = this->m_x[i + 1];
        double xv = this->extractDouble(x);
        double t = (xv - x1) / (x2 - x1);

        // Node values are DoubleT — on tape
        DoubleT log_y1 = log_impl(this->m_y[i]);
        DoubleT log_y2 = log_impl(this->m_y[i + 1]);
        DoubleT log_result = (1.0 - t) * log_y1 + t * log_y2;
        return exp_impl(log_result);
    }

    DoubleT derivativeImpl(DoubleT x) const {
        size_t i = this->locate(x);

        double x1 = this->m_x[i];
        double x2 = this->m_x[i + 1];
        double inv_dx = 1.0 / (x2 - x1);

        DoubleT val = valueImpl(x);
        DoubleT log_y1 = log_impl(this->m_y[i]);
        DoubleT log_y2 = log_impl(this->m_y[i + 1]);
        return val * (log_y2 - log_y1) * inv_dx;
    }

private:
    static DoubleT log_impl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return std::log(x);
        } else {
            using std::log;
            return log(x);
        }
    }

    static DoubleT exp_impl(const DoubleT& x) {
        if constexpr (std::is_same_v<DoubleT, double>) {
            return std::exp(x);
        } else {
            using std::exp;
            return exp(x);
        }
    }
};

} // namespace Math

#endif // LOG_LINEAR_INTERPOLATION_H
