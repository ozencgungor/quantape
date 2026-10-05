#pragma once

#include "quantape/datetime/Calendar.h"
#include "quantape/datetime/Schedule.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/math/LinearAlgebra/DenseSolve.h"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace quantape::markets {
/**
 * @file CurveRisk.h
 * @brief Quote-space risk for a single bootstrap curve
 *
 * Transforms zero-node sensitivities `∂V/∂z_i` into quote sensitivities
 * `∂V/∂r_j` through the instrument Jacobian `F = ∂r/∂z`:
 *
 *   `∂V/∂r = (F^T)^{-1} ∂V/∂z`,
 *
 * assembled analytically from the instrument definitions and the curve node
 * weights. The system is solved densely because stencil spillover (Akima /
 * tension splines) can put entries just above the diagonal. This is the
 * per-curve partial transform (parents frozen); the full stack derivative
 * adds the tree pass implemented in `StackRisk`.
 *
 * Capability notes: in-range risk weights are analytic for Linear, Akima,
 * TensionSpline and MixedLinearCubic, while MonotoneCubic and HymanSpline
 * throw. Extrapolated weights are analytic for every scheme. Cross-currency
 * rows are finite-difference based.
 */

namespace detail {

/// Partial-pivoted LU factorization of a row-major dense matrix: one O(n^3)
/// elimination reused across right-hand sides, each back-substitution O(n^2).
/// Row swaps touch only the active submatrix, matching the interleaved
/// elimination in `quantape::math::solveDense`; `solveDenseLu` replays the
/// swaps on the right-hand side in the same order.
struct DenseLu {
    std::size_t dim = 0;
    std::vector<double> lu;          ///< Row-major; L below the diagonal, U on/above
    std::vector<std::size_t> pivots; ///< Row swapped into position k at step k
};

/// Factor `matrix` (row-major `dim x dim`) with partial pivoting, consuming
/// the scratch copy. Throws when the matrix is singular.
DenseLu factorDenseLu(std::vector<double> matrix, std::size_t dim);

/// Solve `A x = rhs` from a factorization: pivot swaps, forward substitution
/// against the stored multipliers, then back substitution against U. The
/// arithmetic follows `quantape::math::solveDense` step for step.
std::vector<double> solveDenseLu(const DenseLu& factors, const std::vector<double>& rhs);

/// Solve `A x = rhs` from a factorization into a caller-owned buffer; the
/// buffer is resized to the factorization dimension and reused across solves
/// so a full inverse costs one allocation instead of one per column.
void solveDenseLuInto(const DenseLu& factors, const std::vector<double>& rhs,
                      std::vector<double>& x);

/// Cache key of a generated schedule: the constructor arguments that
/// determine the dates, with the calendar identified by the address of its
/// rule-set name (stable for each rule set, distinct between them).
struct ScheduleKey {
    std::int32_t effective = 0;
    std::int32_t maturity = 0;
    int tenorLength = 0;
    datetime::TimeUnit tenorUnit = datetime::TimeUnit::Days;
    const std::string* calendarName = nullptr;
    datetime::BusinessDayConvention convention = datetime::BusinessDayConvention::Following;
};

/// Per-thread cache of generated forward schedules. The row builders rebuild
/// the same schedule on every finite-difference assembly; the cache keeps one
/// generated schedule per distinct key, so repeated bumps reuse the dates.
class ScheduleCache {
public:
    const datetime::Schedule& get(const datetime::Date& effective, const datetime::Date& maturity,
                                  const datetime::Period& tenor, const datetime::Calendar& calendar,
                                  datetime::BusinessDayConvention convention) {
        // A risk run cycles through a bounded set of schedules, so the cap only
        // trips for a long-lived caller that keeps inventing new reference
        // dates; dropping the entries then bounds the footprint.
        if (m_entries.size() >= kMaxEntries) {
            m_entries.clear();
        }
        const ScheduleKey key{effective.serial(), maturity.serial(), tenor.length(),
                              tenor.unit(),       &calendar.name(),  convention};
        const auto generate = [&] {
            return datetime::Schedule(effective, maturity, tenor, calendar, convention,
                                      datetime::DateGeneration::Forward, false,
                                      datetime::BusinessDayConvention::Unadjusted);
        };
        for (CachedSchedule& entry : m_entries) {
            if (entry.key.effective != key.effective || entry.key.maturity != key.maturity ||
                entry.key.tenorLength != key.tenorLength || entry.key.tenorUnit != key.tenorUnit ||
                entry.key.calendarName != key.calendarName ||
                entry.key.convention != key.convention) {
                continue;
            }
            if (entry.calendarAddress == &calendar) {
                return entry.schedule;
            }
            // A different object can share the rule-set name (for example
            // after `withExtraHolidays`), so compare the two calendars on the
            // dates that determine this schedule. A schedule rolls exactly the
            // unadjusted dates, and those follow from date arithmetic alone,
            // so equal rolled dates make the two calendars interchangeable for
            // this request.
            if (sameRolledDates(entry, calendar)) {
                entry.calendarAddress = &calendar;
                return entry.schedule;
            }
            entry.schedule = generate();
            entry.calendar = calendar;
            entry.calendarAddress = &calendar;
            return entry.schedule;
        }
        CachedSchedule entry;
        entry.key = key;
        entry.calendar = calendar;
        entry.calendarAddress = &calendar;
        entry.schedule = generate();
        m_entries.push_back(std::move(entry));
        return m_entries.back().schedule;
    }

private:
    static constexpr std::size_t kMaxEntries = 1024;

    struct CachedSchedule {
        ScheduleKey key;
        datetime::Calendar calendar;
        const datetime::Calendar* calendarAddress = nullptr;
        datetime::Schedule schedule;
    };

    static bool sameRolledDates(const CachedSchedule& entry, const datetime::Calendar& calendar) {
        const datetime::Schedule& schedule = entry.schedule;
        const std::vector<datetime::Date>& unadjusted = schedule.unadjustedDates();
        for (std::size_t i = 0; i < unadjusted.size(); ++i) {
            const datetime::BusinessDayConvention convention =
                i == 0 ? schedule.effectiveConvention()
                       : (i + 1 == unadjusted.size() ? schedule.terminationConvention()
                                                     : schedule.businessDayConvention());
            if (entry.calendar.adjust(unadjusted[i], convention) !=
                calendar.adjust(unadjusted[i], convention)) {
                return false;
            }
        }
        return true;
    }

    // A deque keeps references to cached schedules stable while it grows, so
    // two lookups in one row cannot dangle each other.
    std::deque<CachedSchedule> m_entries;
};

/// Per-thread scratch vectors of the pillar row builder, so repeated rows
/// reuse capacity instead of re-allocating the same working vectors.
struct PillarRowScratch {
    std::vector<double> weights;
    std::vector<double> weights2;
    std::vector<double> weightsStart;
    std::vector<double> weightsEnd;
    std::vector<double> payTimes;
    std::vector<double> startTimes;
    std::vector<double> endTimes;
    std::vector<double> taus;
    std::vector<double> payDf;
    std::vector<double> startDf;
    std::vector<double> endDf;
    std::vector<double> rowFloating;
    std::vector<double> rowAnnuity;
    ScheduleCache schedules;
};

} // namespace detail

struct QuoteRisk {
    std::vector<double> quoteDeltas; ///< ∂V/∂r_j per pillar
};

/// Full quote-space Hessian `H_r = J^T H_zeta J` with `J = F^{-1}` under the
/// Gauss-Newton approximation (bootstrap-curvature term omitted). `hessian` is
/// row-major `m x m` over pillars; diagonal entries are the diagonal gamma and
/// off-diagonals the cross gamma needed for IR ladder risk.
struct QuoteGamma {
    std::size_t dim = 0;
    std::vector<double> hessian;
    double at(std::size_t i, std::size_t j) const { return hessian[i * dim + j]; }
};

/// Analytic instrument Jacobian row contributions for one pillar.
/// Returns false when the pillar is numerically degenerate (non-positive
/// annuity or discount).
template <typename CurveT>
    requires std::same_as<
        std::remove_cvref_t<decltype(std::declval<const CurveT&>().discount(0.0))>, double>
bool pillarJacobianRow(const CurvePillar& pillar, const datetime::Date& referenceDate,
                       const CurveT& curve, std::vector<double>& row) {
    const std::size_t n = curve.size();
    row.assign(n - 1, 0.0); // solved nodes only (node 0 is fixed)
    static thread_local detail::PillarRowScratch scratch;
    std::vector<double>& weights = scratch.weights;
    const auto accumulateInto = [&](std::vector<double>& target, double dRdD, double t, double df) {
        if (t <= 0.0) {
            return; // D(0) = 1 carries no node risk
        }
        curve.zeroNodeWeights(t, weights);
        const double factor = dRdD * (-t * df);
        for (std::size_t i = 1; i < n; ++i) {
            target[i - 1] += factor * weights[i];
        }
    };
    const auto accumulate = [&](double dRdD, double t, double df) {
        accumulateInto(row, dRdD, t, df);
    };

    switch (pillar.kind) {
        case PillarKind::Repo:
        case PillarKind::Deposit: {
            const datetime::Date maturity = adjustedMaturity(pillar);
            const double tau =
                datetime::yearFraction(referenceDate, maturity, pillar.quoteDayCounter);
            const double t =
                datetime::yearFraction(referenceDate, maturity, curve.zeroDayCounter());
            const double df = curve.discount(t);
            accumulate(-1.0 / (tau * df * df), t, df);
            return true;
        }
        case PillarKind::Fra: {
            const datetime::Date start = adjustedStart(pillar);
            const datetime::Date maturity = adjustedMaturity(pillar);
            const double t1 = datetime::yearFraction(referenceDate, start, curve.zeroDayCounter());
            const double t2 =
                datetime::yearFraction(referenceDate, maturity, curve.zeroDayCounter());
            const double tau = datetime::yearFraction(start, maturity, pillar.quoteDayCounter);
            const double d1 = curve.discount(t1);
            const double d2 = curve.discount(t2);
            if (!(tau > 0.0) || !(d2 > 0.0)) {
                return false;
            }
            curve.zeroNodeWeights(t1, weights);
            std::vector<double>& weights2 = scratch.weights2;
            curve.zeroNodeWeights(t2, weights2);
            // dR/df = exp(C) for the shifted-lognormal convexity convention.
            const double scale = std::exp(pillar.fraConvexityExponent) * d1 / (tau * d2);
            for (std::size_t i = 1; i < n; ++i) {
                row[i - 1] += scale * (-t1 * weights[i] + t2 * weights2[i]);
            }
            return true;
        }
        case PillarKind::Future: {
            if (pillar.futureStyle == FutureStyle::Averaged &&
                pillar.averagingStyle == AveragingStyle::Arithmetic) {
                const std::vector<datetime::Date> fixings =
                    businessDayFixings(pillar.calendar, pillar.start, pillar.maturity);
                const std::size_t periods = fixings.size() - 1;
                if (periods == 0) {
                    return false;
                }
                std::vector<double>& weightsStart = scratch.weightsStart;
                std::vector<double>& weightsEnd = scratch.weightsEnd;
                for (std::size_t k = 0; k < periods; ++k) {
                    const double tau =
                        datetime::yearFraction(fixings[k], fixings[k + 1], pillar.quoteDayCounter);
                    const double t1 =
                        datetime::yearFraction(referenceDate, fixings[k], curve.zeroDayCounter());
                    const double t2 = datetime::yearFraction(referenceDate, fixings[k + 1],
                                                             curve.zeroDayCounter());
                    const double d1 = curve.discount(t1);
                    const double d2 = curve.discount(t2);
                    if (!(tau > 0.0) || !(d2 > 0.0)) {
                        return false;
                    }
                    curve.zeroNodeWeights(t1, weightsStart);
                    curve.zeroNodeWeights(t2, weightsEnd);
                    const double ratio = d1 / d2;
                    const double scale = 1.0 / (tau * d2);
                    for (std::size_t i = 1; i < n; ++i) {
                        row[i - 1] +=
                            scale * (-t1 * d1 * weightsStart[i] + ratio * t2 * d2 * weightsEnd[i]);
                    }
                }
                for (std::size_t i = 0; i < n - 1; ++i) {
                    row[i] /= static_cast<double>(periods);
                }
                return true;
            }
            // Averaged+Compounded falls through here: the daily compounding
            // product telescopes to the period discount ratio, so the
            // simple-rate row is exact.
            const double t1 =
                datetime::yearFraction(referenceDate, pillar.start, curve.zeroDayCounter());
            const double t2 =
                datetime::yearFraction(referenceDate, pillar.maturity, curve.zeroDayCounter());
            const double tau =
                datetime::yearFraction(pillar.start, pillar.maturity, pillar.quoteDayCounter);
            const double d1 = curve.discount(t1);
            const double d2 = curve.discount(t2);
            if (!(tau > 0.0) || !(d2 > 0.0)) {
                return false;
            }
            curve.zeroNodeWeights(t1, weights);
            std::vector<double>& weights2 = scratch.weights2;
            curve.zeroNodeWeights(t2, weights2);
            const double scale = d1 / (tau * d2);
            for (std::size_t i = 1; i < n; ++i) {
                row[i - 1] += scale * (-t1 * weights[i] + t2 * weights2[i]);
            }
            return true;
        }
        case PillarKind::OisSwap: {
            const datetime::Date effective =
                pillar.start.serial() != 0 ? pillar.start : referenceDate;
            const datetime::Schedule& schedule =
                scratch.schedules.get(effective, pillar.maturity, pillar.fixedTenor,
                                      pillar.calendar, pillar.businessDayConvention);
            const std::vector<datetime::Date>& dates = schedule.dates();
            const std::size_t payments = dates.size() - 1;
            std::vector<double>& payTimes = scratch.payTimes;
            std::vector<double>& startTimes = scratch.startTimes;
            std::vector<double>& endTimes = scratch.endTimes;
            std::vector<double>& taus = scratch.taus;
            std::vector<double>& payDf = scratch.payDf;
            std::vector<double>& startDf = scratch.startDf;
            std::vector<double>& endDf = scratch.endDf;
            payTimes.resize(payments);
            startTimes.resize(payments);
            endTimes.resize(payments);
            taus.resize(payments);
            payDf.resize(payments);
            startDf.resize(payments);
            endDf.resize(payments);
            double annuity = 0.0;
            double floating = 0.0;
            for (std::size_t k = 0; k < payments; ++k) {
                const datetime::Date payDate = pillar.calendar.advance(
                    dates[k + 1], datetime::Period(pillar.paymentLag, datetime::TimeUnit::Days),
                    pillar.businessDayConvention);
                payTimes[k] =
                    datetime::yearFraction(referenceDate, payDate, curve.zeroDayCounter());
                startTimes[k] =
                    datetime::yearFraction(referenceDate, dates[k], curve.zeroDayCounter());
                endTimes[k] =
                    datetime::yearFraction(referenceDate, dates[k + 1], curve.zeroDayCounter());
                taus[k] = datetime::yearFraction(dates[k], dates[k + 1], pillar.quoteDayCounter);
                payDf[k] = curve.discount(payTimes[k]);
                startDf[k] = curve.discount(startTimes[k]);
                endDf[k] = curve.discount(endTimes[k]);
                annuity += taus[k] * payDf[k];
                if (k == 0 && pillar.firstCouponFixed) {
                    floating += payDf[k] * taus[k] * pillar.firstCouponRate;
                } else {
                    floating += payDf[k] * (startDf[k] / endDf[k] - 1.0);
                }
            }
            if (!(annuity > 0.0)) {
                return false;
            }
            std::vector<double>& rowFloating = scratch.rowFloating;
            std::vector<double>& rowAnnuity = scratch.rowAnnuity;
            rowFloating.assign(n - 1, 0.0);
            rowAnnuity.assign(n - 1, 0.0);
            for (std::size_t k = 0; k < payments; ++k) {
                if (k == 0 && pillar.firstCouponFixed) {
                    // Only the payment-date discounting varies for a fixed coupon.
                    accumulateInto(rowFloating, taus[k] * pillar.firstCouponRate, payTimes[k],
                                   payDf[k]);
                } else {
                    accumulateInto(rowFloating, startDf[k] / endDf[k] - 1.0, payTimes[k], payDf[k]);
                    accumulateInto(rowFloating, payDf[k] / endDf[k], startTimes[k], startDf[k]);
                    accumulateInto(rowFloating, -payDf[k] * startDf[k] / (endDf[k] * endDf[k]),
                                   endTimes[k], endDf[k]);
                }
                accumulateInto(rowAnnuity, taus[k], payTimes[k], payDf[k]);
            }
            const double denominator = annuity * annuity;
            for (std::size_t i = 0; i < n - 1; ++i) {
                row[i] += (rowFloating[i] * annuity - floating * rowAnnuity[i]) / denominator;
            }
            return true;
        }
    }
    return false;
}

/// Assemble the instrument Jacobian `F = ∂r/∂z` over the solved nodes
/// (row-major `m x m`). Shared by the delta and gamma transforms.
template <typename CurveT>
void assembleQuoteJacobian(const CurveT& curve, const std::vector<CurvePillar>& pillars,
                           const datetime::Date& referenceDate, std::vector<double>& f,
                           std::vector<double>& rowScratch) {
    const std::size_t m = pillars.size();
    if (curve.size() != m + 1) {
        throw std::invalid_argument(
            "assembleQuoteJacobian: curve nodes must match the pillar count");
    }
    f.assign(m * m, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        if (!pillarJacobianRow(pillars[j], referenceDate, curve, rowScratch)) {
            throw std::invalid_argument("assembleQuoteJacobian: unsupported pillar");
        }
        for (std::size_t i = 0; i < m; ++i) {
            f[j * m + i] = rowScratch[i];
        }
    }
}

/// Transform zero-node sensitivities (`dVdZeros`, same size as `curve.zeroNodeWeights`)
/// into quote deltas. Throws when sizes mismatch or a pillar is unsupported.
template <typename CurveT>
QuoteRisk transformQuoteRisk(const CurveT& curve, const std::vector<CurvePillar>& pillars,
                             const datetime::Date& referenceDate,
                             const std::vector<double>& dVdZeros) {
    const std::size_t m = pillars.size();
    if (curve.size() != m + 1 || dVdZeros.size() != curve.size()) {
        throw std::invalid_argument("transformQuoteRisk: size mismatch");
    }
    std::vector<double> f;
    std::vector<double> row;
    assembleQuoteJacobian(curve, pillars, referenceDate, f, row);
    std::vector<double> fTranspose(m * m);
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            fTranspose[i * m + j] = f[j * m + i];
        }
    }
    std::vector<double> g(m);
    for (std::size_t i = 0; i < m; ++i) {
        g[i] = dVdZeros[i + 1];
    }
    QuoteRisk result;
    result.quoteDeltas = quantape::math::solveDense(std::move(fTranspose), m, g);
    return result;
}

/**
 * @brief Quote-space Hessian from zero-node second-order sensitivities.
 *
 * `HVdZeros` is the `n x n` row-major Hessian `d^2 V / dz_i dz_j` over curve
 * nodes (node 0 fixed; only rows/columns 1..n-1 are used). The transform is
 * `H_r = 0.5 (J^T H_zeta J + (J^T H_zeta J)^T)` with `J = F^{-1}`, so the
 * returned matrix is exactly symmetric (Gauss-Newton approximation; the
 * bootstrap-curvature term `sum_i dV/dz_i * d^2 z_i / dr^2` is omitted, which
 * is the standard approximation for smooth curves and is reported as an error
 * bar in the fuller report).
 */
template <typename CurveT>
QuoteGamma transformQuoteGamma(const CurveT& curve, const std::vector<CurvePillar>& pillars,
                               const datetime::Date& referenceDate,
                               const std::vector<double>& HVdZeros) {
    const std::size_t m = pillars.size();
    const std::size_t n = curve.size();
    if (n != m + 1 || HVdZeros.size() != n * n) {
        throw std::invalid_argument("transformQuoteGamma: size mismatch");
    }
    std::vector<double> f;
    std::vector<double> row;
    assembleQuoteJacobian(curve, pillars, referenceDate, f, row);
    // J = F^{-1} by one partial-pivoted LU factorization plus m
    // back-substitutions, instead of m independent dense solves.
    const detail::DenseLu factors = detail::factorDenseLu(f, m);
    std::vector<double> jacobianInverse(m * m, 0.0);
    std::vector<double> unit(m, 0.0);
    std::vector<double> solution;
    for (std::size_t column = 0; column < m; ++column) {
        std::fill(unit.begin(), unit.end(), 0.0);
        unit[column] = 1.0;
        detail::solveDenseLuInto(factors, unit, solution);
        for (std::size_t k = 0; k < m; ++k) {
            jacobianInverse[k * m + column] = solution[k];
        }
    }
    // H_r = J^T H_zeta J, restricting H_zeta to the solved nodes.
    std::vector<double> tmp(m * m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t b = 0; b < m; ++b) {
            double sum = 0.0;
            for (std::size_t a = 0; a < m; ++a) {
                sum += HVdZeros[(i + 1) * n + (a + 1)] * jacobianInverse[a * m + b];
            }
            tmp[i * m + b] = sum;
        }
    }
    QuoteGamma result;
    result.dim = m;
    result.hessian.assign(m * m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            double sum = 0.0;
            for (std::size_t a = 0; a < m; ++a) {
                sum += jacobianInverse[a * m + i] * tmp[a * m + j];
            }
            result.hessian[i * m + j] = sum;
        }
    }
    // Symmetrize: round-off in the two matrix products can leave an
    // asymmetric residual even though the analytic transform is symmetric.
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = i + 1; j < m; ++j) {
            const double symmetric = 0.5 * (result.hessian[i * m + j] + result.hessian[j * m + i]);
            result.hessian[i * m + j] = symmetric;
            result.hessian[j * m + i] = symmetric;
        }
    }
    return result;
}

} // namespace quantape::markets
