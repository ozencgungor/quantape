#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace quantape::math {

/**
 * @brief CRTP base class for 1D interpolations
 *
 * Grid coordinates (m_x) are always double — they are not AD-active.
 * Node values (m_y) are DoubleT — these are the AD leaves.
 *
 * This separation is critical for AD performance: grid operations (locate,
 * weight computation) stay off the tape, while the weights themselves are
 * DoubleT so the query coordinate is differentiated by default.
 *
 * Derived classes must implement (non-virtual, accessible via friend):
 * - valueImpl(DoubleT x) const -> DoubleT
 * - derivativeImpl(DoubleT x) const -> DoubleT
 * - valueFixedImpl(DoubleT x) const -> DoubleT      (passive-abscissa policy)
 * - derivativeFixedImpl(DoubleT x) const -> DoubleT (passive-abscissa policy)
 *
 * Explicit template specializations of the Fixed methods for stan::math::var
 * and stan::math::fvar<var> go in InterpolationStanPrimitives.h, following the
 * same pattern as Pricing/StanPrimitives.h for Black76.
 *
 * @tparam DoubleT Numeric type (double, stan::math::var, stan::math::fvar<var>)
 * @tparam Derived CRTP derived class
 */
template <typename DoubleT, typename Derived>
class Interpolation {
public:
    Interpolation() = default;

    // ── Container conversion helpers ──

    /// Convert any container to std::vector<DoubleT> (for node values m_y)
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

    /// Convert any container to std::vector<double> (for grid coordinates m_x)
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

    // ── Public interface (dispatches to Derived via CRTP) ──

    DoubleT operator()(DoubleT x, bool allowExtrapolation = false) const {
        if (!allowExtrapolation && !isInRange(x)) {
            throw std::runtime_error("Interpolation: x is out of range");
        }
        return derived().valueImpl(x);
    }

    DoubleT derivative(DoubleT x, bool allowExtrapolation = false) const {
        if (!allowExtrapolation && !isInRange(x)) {
            throw std::runtime_error("Interpolation: x is out of range for derivative");
        }
        return derived().derivativeImpl(x);
    }

    /**
     * @brief Evaluate treating the query coordinate as a passive abscissa
     *
     * The default operator()/derivative() build weights as DoubleT, so the
     * query point is on the AD tape and grad/Hessian include df/dx and mixed
     * df/dxdy blocks. These Fixed variants restore the node-minimal callback
     * paths (InterpolationStanPrimitives.h) for hot loops where x is known to
     * be a constant: the x adjoint is intentionally not pushed there.
     */
    DoubleT evaluateFixed(DoubleT x, bool allowExtrapolation = false) const {
        if (!allowExtrapolation && !isInRange(x)) {
            throw std::runtime_error("Interpolation: x is out of range");
        }
        return derived().valueFixedImpl(x);
    }

    DoubleT derivativeFixed(DoubleT x, bool allowExtrapolation = false) const {
        if (!allowExtrapolation && !isInRange(x)) {
            throw std::runtime_error("Interpolation: x is out of range for derivative");
        }
        return derived().derivativeFixedImpl(x);
    }

    double xMin() const {
        if (m_x.empty())
            throw std::runtime_error("Interpolation: no data");
        return m_x.front();
    }

    double xMax() const {
        if (m_x.empty())
            throw std::runtime_error("Interpolation: no data");
        return m_x.back();
    }

    size_t size() const { return m_x.size(); }

    bool isInRange(DoubleT x) const {
        if (m_x.empty())
            return false;
        double xVal = extractDouble(x);
        return xVal >= m_x.front() && xVal <= m_x.back();
    }

    /// Access grid coordinates (always double)
    const std::vector<double>& xGrid() const { return m_x; }

    /// Access node values (DoubleT, AD-active)
    const std::vector<DoubleT>& yValues() const { return m_y; }

    // ── Value extraction ──

    /// Recursively extract the innermost double from any AD type.
    /// double -> double, var -> var.val() -> double,
    /// fvar<var> -> fvar.val() -> var -> var.val() -> double
    static double extractDouble(double x) { return x; }

    template <typename T>
    static double extractDouble(const T& x) {
        return extractDouble(x.val());
    }

protected:
    std::vector<double> m_x;  ///< Grid coordinates (double, never AD)
    std::vector<DoubleT> m_y; ///< Node values (DoubleT, AD-active)

    /**
     * @brief Shared snapshot of m_y for the Fixed-path callbacks
     *
     * The passive-abscissa callbacks (InterpolationStanPrimitives.h) need the
     * node values after the interpolator goes out of scope, so they capture
     * them by value. Capturing m_y directly copies the vector on EVERY
     * evaluation; caching one shared snapshot at construction turns that
     * into a shared_ptr copy (refcount bump). The elements alias the same
     * varis as m_y, so adjoint accumulation is unchanged. The vector itself
     * is never resized after caching; element adjoints are the only writes.
     */
    std::shared_ptr<std::vector<DoubleT>> m_y_shared;

    /// Snapshot m_y for the Fixed-path callbacks (call after m_y is final)
    void cacheSharedValues() { m_y_shared = std::make_shared<std::vector<DoubleT>>(m_y); }

    // ── Grid operations (all in double, never on tape) ──

    /// Locate interval: returns i such that m_x[i] <= x < m_x[i+1]
    size_t locate(DoubleT x) const {
        if (m_x.size() < 2)
            throw std::runtime_error("Interpolation: need at least 2 points");

        double xVal = extractDouble(x);

        if (xVal <= m_x.front())
            return 0;
        if (xVal >= m_x.back())
            return m_x.size() - 2;

        // Binary search (pure double, no tape)
        size_t left = 0;
        size_t right = m_x.size() - 1;
        while (right - left > 1) {
            size_t mid = left + (right - left) / 2;
            if (m_x[mid] <= xVal)
                left = mid;
            else
                right = mid;
        }
        return left;
    }

    void validate() const {
        if (m_x.size() != m_y.size())
            throw std::runtime_error("Interpolation: x and y must have same size");
        if (m_x.size() < 2)
            throw std::runtime_error("Interpolation: need at least 2 points");
        for (size_t i = 1; i < m_x.size(); ++i) {
            if (m_x[i] <= m_x[i - 1])
                throw std::runtime_error("Interpolation: x values must be strictly increasing");
        }
    }

private:
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

} // namespace quantape::math

#endif // INTERPOLATION_H
