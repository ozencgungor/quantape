#ifndef INTERPOLATION2D_H
#define INTERPOLATION2D_H

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace Math {

template <typename DoubleT>
class Interpolation2D {
public:
    Interpolation2D() = default;
    virtual ~Interpolation2D() = default;

    DoubleT operator()(DoubleT x, DoubleT y, bool allowExtrapolation = false) const {
        if (!allowExtrapolation && !isInRange(x, y)) {
            throw std::runtime_error("Interpolation2D: (x, y) is out of range");
        }
        return valueImpl(x, y);
    }

protected:
    virtual DoubleT valueImpl(DoubleT x, DoubleT y) const = 0;
    virtual bool isInRange(DoubleT x, DoubleT y) const = 0;

    /**
     * @brief Helper to convert 1D container to std::vector
     */
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

    /**
     * @brief Helper to convert 2D container to std::vector<std::vector<>>
     * Works with std::vector<std::vector<>>, Eigen::Matrix, etc.
     */
    template <typename Container2D>
    static std::vector<std::vector<DoubleT>> toVector2D(const Container2D& container) {
        std::vector<std::vector<DoubleT>> result;

        // For Eigen matrices or similar: use rows() and cols()
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
        }
        // For std::vector<std::vector<>> or similar
        else if constexpr (requires {
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
};

} // namespace Math

#endif // INTERPOLATION2D_H
