#pragma once

#include "quantape/markets/Curves/SpreadCurve.h"

#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quantape::sde {
/**
 * @file StateCurveMap.h
 * @brief Analytic state-to-curve evaluation seam
 *
 * A `StateCurve` turns a flat factor state into discount factors at
 * caller-supplied dates. Pricing holds evaluation indices into a precomputed
 * schedule instead of curve objects, so the hot path has no interpolation, no
 * key lookup and no allocation: one factor dot product plus one exponential
 * per coupon.
 */

/// State-to-curve contract: `bond(k, j, state)` is the discount factor
/// P(t_k, T_j) for the k-th observation date and j-th payment date, and
/// `annuity` sums coupon accruals times bonds over a window. Implementations
/// expose the precomputed schedule through `obsTimes`/`payTimes` and the
/// factor count through `stateSize`.
template <typename Map, typename Scalar>
concept StateCurve =
    requires(const Map& map, std::size_t k, std::size_t j, std::span<const Scalar> state) {
        { map.bond(k, j, state) } -> std::same_as<Scalar>;
        { map.annuity(k, k, state, std::span<const double>{}) } -> std::same_as<Scalar>;
        { map.stateSize() } -> std::convertible_to<std::size_t>;
        { map.obsTimes() } -> std::convertible_to<std::span<const double>>;
        { map.payTimes() } -> std::convertible_to<std::span<const double>>;
    };

/// Gaussian multi-factor state-to-curve map on caller-supplied dates.
///
///     P(t_k, T_j, x) = (P0(T_j) / P0(t_k)) * exp(-B(t_k,T_j) . x - Omega/2),
///     Omega(t_k,T_j) = sum_il rho_il * B_i(t_k,T_j) * B_l(t_k,T_j).
///
/// The base discounts P0, the variance terms Omega and the loading blocks B
/// are precomputed once at construction into one contiguous record per node
/// (base ratio, Omega, B_0..B_{q-1}) for exactly the requested observation and
/// payment dates; evaluation is one O(q) dot product plus one exponential per
/// bond on a flat record stride. A deterministic block (all B_i = 0, hence Omega = 0) makes the
/// zero state reproduce the base curve ratio P0(T_j)/P0(t_k) exactly. For a nonzero block the
/// -Omega/2 term centers the log-deviation, so a state x ~ N(0, C) has E[P(t,T,x)] = P0(T)/P0(t).
class GaussianStateCurve {
public:
    /// SoA loading blocks: `factor[i][k * payCount() + j]` holds B_i(t_k,T_j);
    /// `correlation` is the q x q factor correlation matrix (row-major).
    struct Loadings {
        std::vector<std::vector<double>> factor;
        std::vector<double> correlation;
    };

    /// Build from precomputed base discounts `baseAtObs[k] = P0(t_k)` and
    /// `baseAtPay[j] = P0(T_j)`. Throws `std::invalid_argument` on empty or
    /// unsorted dates, non-positive base discounts, mismatched sizes or a
    /// malformed loading block.
    GaussianStateCurve(std::vector<double> obsTimes, std::vector<double> payTimes,
                       std::vector<double> baseAtObs, std::vector<double> baseAtPay,
                       Loadings loadings);

    /// Freeze any double curve provider (OIS, spread or forecast chain) on the
    /// requested dates and attach `loadings`.
    template <typename Provider>
        requires markets::CurveProvider<Provider, double>
    static GaussianStateCurve fromCurve(const Provider& base, std::vector<double> obsTimes,
                                        std::vector<double> payTimes, Loadings loadings) {
        std::vector<double> baseAtObs;
        baseAtObs.reserve(obsTimes.size());
        for (const double t : obsTimes) {
            baseAtObs.push_back(base.discount(t));
        }
        std::vector<double> baseAtPay;
        baseAtPay.reserve(payTimes.size());
        for (const double t : payTimes) {
            baseAtPay.push_back(base.discount(t));
        }
        return GaussianStateCurve(std::move(obsTimes), std::move(payTimes), std::move(baseAtObs),
                                  std::move(baseAtPay), std::move(loadings));
    }

    std::size_t stateSize() const noexcept { return m_factorCount; }
    std::size_t obsCount() const noexcept { return m_obsTimes.size(); }
    std::size_t payCount() const noexcept { return m_payTimes.size(); }
    std::span<const double> obsTimes() const noexcept { return m_obsTimes; }
    std::span<const double> payTimes() const noexcept { return m_payTimes; }

    /// P(t_k, T_j, state); `state` must have `stateSize()` entries. The
    /// evaluation touches only the precomputed arrays and allocates nothing.
    template <typename Scalar>
    Scalar bond(std::size_t k, std::size_t j, std::span<const Scalar> state) const {
        requireIndex(k, j);
        requireStateSize(state.size(), "bond");
        return evaluateRecord(rowAt(k * payCount() + j), m_factorCount, state.data());
    }

    /// Sum of `accruals[j - firstPay] * bond(k, j, state)` over the coupon
    /// window `[firstPay, firstPay + accruals.size())`.
    template <typename Scalar>
    Scalar annuity(std::size_t k, std::size_t firstPay, std::span<const Scalar> state,
                   std::span<const double> accruals) const {
        if (k >= obsCount() || firstPay + accruals.size() > payCount()) {
            throw std::out_of_range("GaussianStateCurve::annuity: index out of range");
        }
        requireStateSize(state.size(), "annuity");
        Scalar sum(0.0);
        const double* record = rowAt(k * payCount() + firstPay);
        const std::size_t stride = rowStride();
        for (std::size_t j = 0; j < accruals.size(); ++j) {
            sum += accruals[j] * evaluateRecord(record, m_factorCount, state.data());
            record += stride;
        }
        return sum;
    }

    /// Write every bond P(t_k, T_j) at observation `k` into `out`; `out` must
    /// have `payCount()` entries and is reused across calls.
    template <typename Scalar>
    void bondsInto(std::size_t k, std::span<const Scalar> state, std::span<Scalar> out) const {
        if (k >= obsCount()) {
            throw std::out_of_range("GaussianStateCurve::bondsInto: index out of range");
        }
        requireStateSize(state.size(), "bondsInto");
        if (out.size() != payCount()) {
            throw std::invalid_argument("GaussianStateCurve::bondsInto: output size mismatch");
        }
        const double* record = rowAt(k * payCount());
        const Scalar* values = state.data();
        const std::size_t stride = rowStride();
        for (std::size_t j = 0; j < out.size(); ++j) {
            out[j] = evaluateRecord(record, m_factorCount, values);
            record += stride;
        }
    }

    /// Base curve ratio P0(T_j)/P0(t_k); the exact zero state of a
    /// deterministic loading block.
    double baseRatio(std::size_t k, std::size_t j) const {
        requireIndex(k, j);
        return rowAt(k * payCount() + j)[0];
    }

    /// Variance term Omega(t_k,T_j) = sum_il rho_il B_i B_l.
    double omega(std::size_t k, std::size_t j) const {
        requireIndex(k, j);
        return rowAt(k * payCount() + j)[1];
    }

    /// Loading B_i(t_k, T_j).
    double loading(std::size_t i, std::size_t k, std::size_t j) const {
        if (i >= stateSize()) {
            throw std::out_of_range("GaussianStateCurve::loading: factor index out of range");
        }
        requireIndex(k, j);
        return rowAt(k * payCount() + j)[2 + i];
    }

    /// Gaussian bond from explicit loading and state spans:
    /// `baseRatio * exp(-sum_i loading_i * state_i - variance / 2)`. `loading`
    /// and `state` may be double, `var` or `fvar<var>`, so derivatives flow
    /// through either side.
    template <typename LoadingScalar, typename StateScalar>
    static auto analyticBond(double baseRatio, std::span<const LoadingScalar> loading,
                             std::span<const StateScalar> state, double variance)
        -> decltype(loading[0] * state[0]) {
        if (loading.size() != state.size()) {
            throw std::invalid_argument("GaussianStateCurve::analyticBond: size mismatch");
        }
        using ResultT = decltype(loading[0] * state[0]);
        ResultT dot(0.0);
        for (std::size_t i = 0; i < loading.size(); ++i) {
            dot += loading[i] * state[i];
        }
        using std::exp;
        return baseRatio * exp(-dot - 0.5 * variance);
    }

private:
    /// Per-node record: `[P0(T_j)/P0(t_k), Omega(t_k,T_j), B_0 .. B_{q-1}]`.
    /// One contiguous record per (k, j) keeps the base ratio, the variance
    /// term and the loading dot on the same cache line, so a row walk
    /// (`bondsInto`/`annuity`) streams one record stride instead of three
    /// parallel arrays.
    std::size_t rowStride() const noexcept { return m_factorCount + 2; }

    const double* rowAt(std::size_t row) const noexcept {
        return m_rows.data() + row * rowStride();
    }

    /// One loading dot plus one exponential on a precomputed record. The dot
    /// starts from the negated first product and subtracts every later factor
    /// in index order (identical rounding to summing the products and negating
    /// afterward, by sign symmetry), and the variance is folded in with a
    /// single add, so the result is bit-for-bit the same as
    /// `base * exp(-dot - Omega/2)` computed on a plain double loop. Starting
    /// from the first product keeps the sum's leading constant off the tape
    /// on the var/fvar<var> tracks (a constant stays a constant). The dominant
    /// factor counts (4 and 2) are unrolled in the same order; other counts
    /// fall back to the loop.
    template <typename Scalar>
    static Scalar evaluateRecord(const double* record, std::size_t factors, const Scalar* values) {
        if (factors == 0) {
            return Scalar(record[0]);
        }
        Scalar dot((-record[2]) * values[0]);
        if (factors == 4) {
            dot -= record[3] * values[1];
            dot -= record[4] * values[2];
            dot -= record[5] * values[3];
        } else if (factors == 2) {
            dot -= record[3] * values[1];
        } else {
            for (std::size_t i = 1; i < factors; ++i) {
                dot -= record[2 + i] * values[i];
            }
        }
        using std::exp;
        return record[0] * exp(dot - 0.5 * record[1]);
    }

    void requireIndex(std::size_t k, std::size_t j) const {
        if (k >= obsCount() || j >= payCount()) [[unlikely]] {
            raiseIndex();
        }
    }

    void requireStateSize(std::size_t size, const char* caller) const {
        if (size != stateSize()) [[unlikely]] {
            raiseStateSize(caller);
        }
    }

    /// Cold failure paths kept out of line so the hot checks inline to one
    /// compare and a predicted branch; the thrown types and messages are
    /// unchanged.
    [[noreturn]] [[gnu::noinline, gnu::cold]] void raiseIndex() const {
        throw std::out_of_range("GaussianStateCurve: date index out of range");
    }

    [[noreturn]] [[gnu::noinline, gnu::cold]] void raiseStateSize(const char* caller) const {
        throw std::invalid_argument(std::string("GaussianStateCurve::") + caller +
                                    ": state size mismatch");
    }

    std::vector<double> m_obsTimes;
    std::vector<double> m_payTimes;
    std::size_t m_factorCount = 0; ///< number of loading factors
    std::vector<double> m_rows;    ///< record per (k,j), row-major [k * payCount + j]
};

namespace detail {

inline void requireSchedule(const std::vector<double>& times, const char* label) {
    if (times.empty()) {
        throw std::invalid_argument(std::string("GaussianStateCurve: empty ") + label + " times");
    }
    if (times.front() < 0.0) {
        throw std::invalid_argument(std::string("GaussianStateCurve: negative ") + label + " time");
    }
    for (std::size_t i = 1; i < times.size(); ++i) {
        if (!(times[i] > times[i - 1])) {
            throw std::invalid_argument(std::string("GaussianStateCurve: ") + label +
                                        " times must be strictly increasing");
        }
    }
}

inline void requirePositiveDiscounts(const std::vector<double>& values, const char* label) {
    for (const double value : values) {
        if (!(value > 0.0)) {
            throw std::invalid_argument(std::string("GaussianStateCurve: non-positive ") + label +
                                        " base discount");
        }
    }
}

} // namespace detail

inline GaussianStateCurve::GaussianStateCurve(std::vector<double> obsTimes,
                                              std::vector<double> payTimes,
                                              std::vector<double> baseAtObs,
                                              std::vector<double> baseAtPay, Loadings loadings)
    : m_obsTimes(std::move(obsTimes)), m_payTimes(std::move(payTimes)) {
    if (m_obsTimes.size() != baseAtObs.size()) {
        throw std::invalid_argument("GaussianStateCurve: observation times/base size mismatch");
    }
    if (m_payTimes.size() != baseAtPay.size()) {
        throw std::invalid_argument("GaussianStateCurve: payment times/base size mismatch");
    }
    detail::requireSchedule(m_obsTimes, "observation");
    detail::requireSchedule(m_payTimes, "payment");
    detail::requirePositiveDiscounts(baseAtObs, "observation");
    detail::requirePositiveDiscounts(baseAtPay, "payment");

    const std::size_t q = loadings.factor.size();
    if (q == 0) {
        if (!loadings.correlation.empty()) {
            throw std::invalid_argument("GaussianStateCurve: correlation without loading factors");
        }
    } else if (loadings.correlation.size() != q * q) {
        throw std::invalid_argument("GaussianStateCurve: correlation size mismatch");
    }
    const std::size_t rows = m_obsTimes.size() * m_payTimes.size();
    for (const std::vector<double>& block : loadings.factor) {
        if (block.size() != rows) {
            throw std::invalid_argument("GaussianStateCurve: loading block size mismatch");
        }
    }

    const std::size_t nPay = m_payTimes.size();
    const std::size_t stride = q + 2;
    m_factorCount = q;
    m_rows.assign(rows * stride, 0.0);
    double* const rowsData = m_rows.data();
    for (std::size_t k = 0; k < m_obsTimes.size(); ++k) {
        const double baseObs = baseAtObs[k];
        const std::size_t rowBase = k * nPay;
        std::size_t j = 0;
        // Four records per step with one accumulator each: every record keeps
        // the exact i-major/l-major accumulation order of the scalar loop (so
        // the stored bits are unchanged), while the four dependency chains
        // interleave and keep the FMA pipeline busy.
        for (; j + 4 <= nPay; j += 4) {
            const std::size_t r0 = rowBase + j;
            double* const record0 = rowsData + r0 * stride;
            double* const record1 = record0 + stride;
            double* const record2 = record0 + 2 * stride;
            double* const record3 = record0 + 3 * stride;
            record0[0] = baseAtPay[j] / baseObs;
            record1[0] = baseAtPay[j + 1] / baseObs;
            record2[0] = baseAtPay[j + 2] / baseObs;
            record3[0] = baseAtPay[j + 3] / baseObs;
            for (std::size_t i = 0; i < q; ++i) {
                const double* const block = loadings.factor[i].data();
                record0[2 + i] = block[r0];
                record1[2 + i] = block[r0 + 1];
                record2[2 + i] = block[r0 + 2];
                record3[2 + i] = block[r0 + 3];
            }
            double sum0 = 0.0;
            double sum1 = 0.0;
            double sum2 = 0.0;
            double sum3 = 0.0;
            for (std::size_t i = 0; i < q; ++i) {
                const double* const blockI = loadings.factor[i].data();
                const double factorI0 = blockI[r0];
                const double factorI1 = blockI[r0 + 1];
                const double factorI2 = blockI[r0 + 2];
                const double factorI3 = blockI[r0 + 3];
                for (std::size_t l = 0; l < q; ++l) {
                    const double c = loadings.correlation[i * q + l];
                    const double* const blockL = loadings.factor[l].data();
                    sum0 += c * factorI0 * blockL[r0];
                    sum1 += c * factorI1 * blockL[r0 + 1];
                    sum2 += c * factorI2 * blockL[r0 + 2];
                    sum3 += c * factorI3 * blockL[r0 + 3];
                }
            }
            record0[1] = sum0;
            record1[1] = sum1;
            record2[1] = sum2;
            record3[1] = sum3;
        }
        for (; j < nPay; ++j) {
            const std::size_t r = rowBase + j;
            double* const record = rowsData + r * stride;
            record[0] = baseAtPay[j] / baseObs;
            for (std::size_t i = 0; i < q; ++i) {
                record[2 + i] = loadings.factor[i][r];
            }
            double sum = 0.0;
            for (std::size_t i = 0; i < q; ++i) {
                for (std::size_t l = 0; l < q; ++l) {
                    sum += loadings.correlation[i * q + l] * loadings.factor[i][r] *
                           loadings.factor[l][r];
                }
            }
            record[1] = sum;
        }
    }
}

static_assert(StateCurve<GaussianStateCurve, double>);

/// Fallback seam for state-to-curve maps that cannot use the analytic Gaussian
/// bond formula (non-Gaussian or exotic models). Contract:
///  - the frozen log-discount surface log P0(t_k,T_j) and the model's factor
///    loading blocks are precomputed before the run;
///  - an implementation shifts one node by its factor block,
///    log P(t_k,T_j,x) = log P0(t_k,T_j) - shift_kj(x), and returns the
///    shifted log-discount (one exponential per node stays with the consumer);
///  - the zero state returns log P0 exactly and evaluation allocates nothing.
/// No model-specific shift is implemented yet; the interface pins the seam so
/// later models can land without changing its consumers.
template <typename Scalar>
class GridDeformationMap {
public:
    virtual ~GridDeformationMap() = default;

    /// Shifted log-discount at frozen node (k, j) for `state`.
    virtual Scalar logDiscount(std::size_t k, std::size_t j,
                               std::span<const Scalar> state) const = 0;
};

} // namespace quantape::sde
