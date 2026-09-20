#ifndef INTERPOLATION2D_H
#define INTERPOLATION2D_H

#include <stdexcept>
#include <vector>

namespace Math {

/**
 * @brief CRTP base class for 2D interpolations
 *
 * Derived classes must implement (non-virtual, accessible via friend):
 * - valueImpl(DoubleT x, DoubleT y) const -> DoubleT
 * - isInRange(DoubleT x, DoubleT y) const -> bool
 *
 * @tparam DoubleT Numeric type (double, stan::math::var, stan::math::fvar<var>)
 * @tparam Derived CRTP derived class
 */
template <typename DoubleT, typename Derived>
class Interpolation2D {
public:
    Interpolation2D() = default;

    DoubleT operator()(DoubleT x, DoubleT y, bool allowExtrapolation = false) const {
        if (!allowExtrapolation && !derived().isInRange(x, y)) {
            throw std::runtime_error("Interpolation2D: (x, y) is out of range");
        }
        return derived().valueImpl(x, y);
    }

    // ── Value extraction ──

    static double extractDouble(double x) { return x; }

    template <typename T>
    static double extractDouble(const T& x) {
        return extractDouble(x.val());
    }

    // ── Container conversion helpers ──

    template <typename Container>
    static std::vector<DoubleT> toVector(const Container& container) {
        if constexpr (std::is_same_v<Container, std::vector<DoubleT>>) {
            return container;
        } else {
            std::vector<DoubleT> result;
            result.reserve(container.size());
            for (size_t i = 0; i < container.size(); ++i) {
                result.push_back(static_cast<DoubleT>(container[i]));
            }
            return result;
        }
    }

    template <typename Container>
    static std::vector<double> toDoubleVector(const Container& container) {
        if constexpr (std::is_same_v<Container, std::vector<double>>) {
            return container;
        } else {
            std::vector<double> result;
            result.reserve(container.size());
            for (size_t i = 0; i < container.size(); ++i) {
                result.push_back(extractDouble(container[i]));
            }
            return result;
        }
    }

    template <typename Container2D>
    static std::vector<std::vector<DoubleT>> toVector2D(const Container2D& container) {
        std::vector<std::vector<DoubleT>> result;

        if constexpr (requires {
                          container.rows();
                          container.cols();
                      }) {
            result.resize(container.rows());
            for (size_t i = 0; i < static_cast<size_t>(container.rows()); ++i) {
                result[i].resize(container.cols());
                for (size_t j = 0; j < static_cast<size_t>(container.cols()); ++j) {
                    result[i][j] = static_cast<DoubleT>(container(i, j));
                }
            }
        } else if constexpr (requires {
                                 container.size();
                                 container[0].size();
                             }) {
            result.resize(container.size());
            for (size_t i = 0; i < container.size(); ++i) {
                result[i].resize(container[i].size());
                for (size_t j = 0; j < container[i].size(); ++j) {
                    result[i][j] = static_cast<DoubleT>(container[i][j]);
                }
            }
        }

        return result;
    }

private:
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

} // namespace Math

#endif // INTERPOLATION2D_H
