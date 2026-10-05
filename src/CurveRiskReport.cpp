#include "quantape/markets/Curves/CurveRiskReport.h"

#include "quantape/markets/Curves/CurveRisk.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace quantape::markets {
CurveRiskReport
curveRiskReport(const DiscountCurve<double>& curve, const std::vector<CurvePillar>& pillars,
                const datetime::Date& referenceDate, const std::vector<double>& dVdZeros,
                const CurveRiskReportOptions& options, const std::vector<double>* HVdZeros) {
    const QuoteRisk risk = transformQuoteRisk(curve, pillars, referenceDate, dVdZeros);
    CurveRiskReport report;
    report.quoteDeltas = risk.quoteDeltas;
    report.quoteLabels.reserve(pillars.size());
    const auto addBucket = [](std::vector<RiskBucket>& buckets, std::string_view label,
                              double delta) {
        for (RiskBucket& bucket : buckets) {
            if (bucket.label == label) {
                bucket.delta += delta;
                return;
            }
        }
        buckets.push_back(RiskBucket{std::string(label), delta});
    };
    std::vector<std::string> tags(pillars.size());
    for (std::size_t j = 0; j < pillars.size(); ++j) {
        const CurvePillar& pillar = pillars[j];
        const datetime::Date maturity = pillarRiskMaturity(pillar);
        const double t = datetime::yearFraction(referenceDate, maturity, curve.zeroDayCounter());
        std::string tag = riskMaturityTag(maturity, t);
        const std::string_view kind = pillarKindName(pillar.kind);
        std::string& label = report.quoteLabels.emplace_back();
        label.reserve(kind.size() + 1 + tag.size());
        label.assign(kind);
        label.push_back(' ');
        label += tag;
        addBucket(report.byKind, kind, risk.quoteDeltas[j]);
        addBucket(report.byEnd, t <= options.shortEndYears ? "ShortEnd" : "LongEnd",
                  risk.quoteDeltas[j]);
        addBucket(report.byYear, tag, risk.quoteDeltas[j]);
        tags[j] = std::move(tag);
        report.horizonYears = std::max(report.horizonYears, t);
    }
    if (HVdZeros != nullptr) {
        if (HVdZeros->size() != curve.size() * curve.size()) {
            throw std::invalid_argument("curveRiskReport: HVdZeros size mismatch");
        }
        const QuoteGamma gamma = transformQuoteGamma(curve, pillars, referenceDate, *HVdZeros);
        report.gammaDiagonal.resize(pillars.size());
        for (std::size_t j = 0; j < pillars.size(); ++j) {
            report.gammaDiagonal[j] = gamma.at(j, j);
        }
        // Maturity-tag buckets in first-appearance order (matches byYear labels).
        for (const RiskBucket& bucket : report.byYear) {
            report.gammaBucketLabels.push_back(bucket.label);
        }
        const std::size_t buckets = report.gammaBucketLabels.size();
        report.gammaDiagByYear.assign(buckets, RiskBucket{});
        report.gammaCrossByYear.assign(buckets * buckets, 0.0);
        std::vector<std::size_t> quoteBucket(pillars.size(), 0);
        for (std::size_t j = 0; j < pillars.size(); ++j) {
            bool found = false;
            for (std::size_t b = 0; b < buckets; ++b) {
                if (report.gammaBucketLabels[b] == tags[j]) {
                    quoteBucket[j] = b;
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::invalid_argument(
                    "curveRiskReport: gamma pillar label is not in the year buckets");
            }
        }
        for (std::size_t b = 0; b < buckets; ++b) {
            report.gammaDiagByYear[b].label = report.gammaBucketLabels[b];
        }
        for (std::size_t i = 0; i < pillars.size(); ++i) {
            report.gammaDiagByYear[quoteBucket[i]].delta += gamma.at(i, i);
            for (std::size_t j = 0; j < pillars.size(); ++j) {
                report.gammaCrossByYear[quoteBucket[i] * buckets + quoteBucket[j]] +=
                    gamma.at(i, j);
            }
        }
    }
    return report;
}

std::vector<RiskBucket>
aggregateByRoleAndKind(const std::vector<std::pair<CurveRole, CurveRiskReport>>& reports) {
    std::vector<RiskBucket> buckets;
    for (const auto& entry : reports) {
        const std::string role(curveRoleName(entry.first));
        for (const RiskBucket& kind : entry.second.byKind) {
            const std::string label = role + "/" + kind.label;
            bool merged = false;
            for (RiskBucket& bucket : buckets) {
                if (bucket.label == label) {
                    bucket.delta += kind.delta;
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                buckets.push_back(RiskBucket{label, kind.delta});
            }
        }
    }
    return buckets;
}

namespace detail {

QrLeastSquares householderQrLeastSquares(std::vector<double> matrix, std::size_t rows,
                                         std::size_t cols, std::vector<double> rhs) {
    std::vector<std::size_t> pivots(cols);
    std::iota(pivots.begin(), pivots.end(), std::size_t{0});
    QrLeastSquares result;
    std::size_t rank = 0;
    std::vector<double> reflector(rows);
    const std::size_t steps = std::min(rows, cols);
    for (std::size_t k = 0; k < steps; ++k) {
        std::size_t pivot = k;
        double pivotNormSquared = -1.0;
        for (std::size_t j = k; j < cols; ++j) {
            double sum = 0.0;
            for (std::size_t i = k; i < rows; ++i) {
                const double value = matrix[i * cols + j];
                sum += value * value;
            }
            if (sum > pivotNormSquared) {
                pivotNormSquared = sum;
                pivot = j;
            }
        }
        if (pivot != k) {
            for (std::size_t i = 0; i < rows; ++i) {
                std::swap(matrix[i * cols + k], matrix[i * cols + pivot]);
            }
            std::swap(pivots[k], pivots[pivot]);
        }
        const double norm = std::sqrt(std::max(pivotNormSquared, 0.0));
        result.scale = std::max(result.scale, norm);
        if (!(norm > kQrRankTolerance * result.scale)) {
            break;
        }
        const double pivotValue = matrix[k * cols + k];
        const double alpha = pivotValue >= 0.0 ? -norm : norm;
        for (std::size_t i = k; i < rows; ++i) {
            reflector[i] = matrix[i * cols + k];
        }
        reflector[k] -= alpha;
        double reflectorNormSquared = 0.0;
        for (std::size_t i = k; i < rows; ++i) {
            reflectorNormSquared += reflector[i] * reflector[i];
        }
        for (std::size_t j = k; j < cols; ++j) {
            double dot = 0.0;
            for (std::size_t i = k; i < rows; ++i) {
                dot += reflector[i] * matrix[i * cols + j];
            }
            const double factor = 2.0 * dot / reflectorNormSquared;
            for (std::size_t i = k; i < rows; ++i) {
                matrix[i * cols + j] -= factor * reflector[i];
            }
        }
        double dot = 0.0;
        for (std::size_t i = k; i < rows; ++i) {
            dot += reflector[i] * rhs[i];
        }
        const double factor = 2.0 * dot / reflectorNormSquared;
        for (std::size_t i = k; i < rows; ++i) {
            rhs[i] -= factor * reflector[i];
        }
        matrix[k * cols + k] = alpha;
        rank = k + 1;
    }
    std::vector<double> solution(cols, 0.0);
    for (std::size_t step = rank; step-- > 0;) {
        double sum = rhs[step];
        for (std::size_t j = step + 1; j < cols; ++j) {
            sum -= matrix[step * cols + j] * solution[pivots[j]];
        }
        solution[pivots[step]] = sum / matrix[step * cols + step];
    }
    result.solution = std::move(solution);
    result.rank = rank;
    return result;
}

} // namespace detail

std::vector<double> solveHedge(const std::vector<double>& jacobian, std::size_t m, std::size_t h,
                               const std::vector<double>& delta, double ridge) {
    if (h == 0 || jacobian.size() != m * h || delta.size() != m) {
        throw std::invalid_argument("solveHedge: size mismatch");
    }
    if (m < h) {
        throw std::invalid_argument("solveHedge: need at least as many rows as hedge instruments");
    }
    if (!std::isfinite(ridge) || ridge < 0.0) {
        throw std::invalid_argument("solveHedge: ridge must be finite and non-negative");
    }
    const std::size_t rows = m + h;
    std::vector<double> augmented(rows * h, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t a = 0; a < h; ++a) {
            augmented[i * h + a] = jacobian[i * h + a];
        }
    }
    std::vector<double> rhs(rows, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        rhs[i] = -delta[i];
    }
    const auto solveAugmented = [&](double effectiveRidge) {
        std::vector<double> matrix = augmented;
        const double root = std::sqrt(effectiveRidge);
        for (std::size_t a = 0; a < h; ++a) {
            matrix[(m + a) * h + a] = root;
        }
        return detail::householderQrLeastSquares(std::move(matrix), rows, h, rhs);
    };
    detail::QrLeastSquares qr = solveAugmented(ridge);
    if (qr.rank < h) {
        // Only a truly singular J reaches this with ridge == 0. Retry on a
        // ridge-like perturbation proportional to the matrix scale so the
        // null direction is damped instead of amplified.
        const double perturbation = 100.0 * detail::kQrRankTolerance * std::max(qr.scale, 1.0);
        qr = solveAugmented(std::max(ridge, perturbation * perturbation));
    }
    return std::move(qr.solution);
}

DualSchemeRisk dualSchemeQuoteRisk(const DiscountCurve<double>& pricingCurve,
                                   const DiscountCurve<double>& riskCurve,
                                   const std::vector<CurvePillar>& pillars,
                                   const datetime::Date& referenceDate,
                                   const std::vector<double>& dVdZeros) {
    DualSchemeRisk out;
    out.audit = transformQuoteRisk(pricingCurve, pillars, referenceDate, dVdZeros).quoteDeltas;
    out.hedge = transformQuoteRisk(riskCurve, pillars, referenceDate, dVdZeros).quoteDeltas;
    out.basis.resize(out.audit.size());
    for (std::size_t j = 0; j < out.audit.size(); ++j) {
        out.basis[j] = out.hedge[j] - out.audit[j];
    }
    return out;
}

std::vector<RiskBucket>
aggregateByRole(const std::vector<std::pair<CurveRole, std::vector<double>>>& quoteDeltasByCurve) {
    std::vector<RiskBucket> buckets;
    for (const auto& entry : quoteDeltasByCurve) {
        double sum = 0.0;
        for (const double delta : entry.second) {
            sum += delta;
        }
        const std::string label = std::string(curveRoleName(entry.first));
        bool merged = false;
        for (RiskBucket& bucket : buckets) {
            if (bucket.label == label) {
                bucket.delta += sum;
                merged = true;
                break;
            }
        }
        if (!merged) {
            buckets.push_back(RiskBucket{label, sum});
        }
    }
    return buckets;
}
} // namespace quantape::markets
