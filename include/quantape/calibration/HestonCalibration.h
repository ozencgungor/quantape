#ifndef QUANTAPE_MODELS_HESTON_CALIBRATION_H
#define QUANTAPE_MODELS_HESTON_CALIBRATION_H

#include "quantape/calibration/CalibrationProblem.h"
#include "quantape/models/HestonModel.h"
#include "quantape/models/HestonStanPrimitives.h"

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace quantape::models {
/**
 * @file HestonCalibration.h
 * @brief `CalibrationProblem` adapter: Heston analytic pricer -> generic
 *        calibration + IFT/KKT risk layer (`calibration/CalibrationProblem.h`)
 *
 * Mapping:
 *   b = [v0, kappa, theta, sigma, rho]            (5 model parameters)
 *   a = [spot, rate, dividend]                    (3 shared market parameters)
 *   quote = {strike, tMax, target, weight}        (strike/time are quote data)
 *
 * Residual `r_i = sqrt(weight_i) (P_i(b, a) - target_i)` with derivatives
 * taken from the full 9-parameter complex-step gradient/Hessian
 * (`HestonModel::fullGradient/fullHessian`, strike held at the quote's).
 * The Feller inequality `g = sigma^2 - 2 kappa theta <= 0` is exposed for
 * `AugLag`/`SLSQP` and for the constrained IFT; it does not depend on `a`,
 * which is the v1 scope of the generic IFT layer.
 */
struct HestonCalibrationQuote {
    double strike = 1.0;
    double tMax = 1.0;
    double target = 0.0; ///< target price
    double weight = 1.0; ///< diagonal LS weight (e.g. 1 / vega^2)
};

class HestonCalibrationProblem {
public:
    HestonCalibrationProblem(const HestonModel& model, std::vector<HestonCalibrationQuote> quotes)
        : m_model(&model), m_quotes(std::move(quotes)) {}

    std::size_t numModelParams() const { return 5; }
    std::size_t numMarketParams() const { return 3; } ///< spot, rate, dividend
    std::size_t numQuotes() const { return m_quotes.size(); }
    std::size_t numInequalities() const { return 1; } ///< Feller

    /// Full 9-parameter point `[v0..rho, spot, strike, rate, dividend]`.
    HestonFullPoint point(std::size_t i, const Eigen::VectorXd& b, const Eigen::VectorXd& a) const {
        HestonFullPoint p{};
        for (std::size_t k = 0; k < 5; ++k) {
            p[k] = b(static_cast<Eigen::Index>(k));
        }
        p[HESTON_SPOT] = a(0);
        p[HESTON_STRIKE] = m_quotes[i].strike;
        p[HESTON_RATE] = a(1);
        p[HESTON_DIVIDEND] = a(2);
        return p;
    }

    double price(std::size_t i, const Eigen::VectorXd& b, const Eigen::VectorXd& a) const {
        return m_model->callFull(point(i, b, a), m_quotes[i].tMax);
    }

    // ── Scalar-generic (AD) view ─────────────────────────────────────────
    //
    // `residual<S>`/`inequalities<S>` work for double, `stan::math::var` and
    // `stan::math::fvar<stan::math::var>`; the AD scalars price through the
    // `make_callback_var` overloads (models/HestonStanPrimitives.h), so an
    // objective over all quotes adds one tape node per quote.

    template <typename S>
    S price(std::size_t i, const std::vector<S>& b, const std::vector<S>& a) const {
        return priceScalar<S>(i, b, a);
    }

    template <typename S>
    S residual(std::size_t i, const std::vector<S>& b, const std::vector<S>& a) const {
        const S priceValue = priceScalar<S>(i, b, a);
        return S(std::sqrt(m_quotes[i].weight)) * (priceValue - S(m_quotes[i].target));
    }

    template <typename S>
    void inequalities(const std::vector<S>& b, std::vector<S>& out) const {
        out.resize(1);
        out[0] = b[3] * b[3] - S(2.0) * b[1] * b[2]; // Feller: sigma^2 - 2 kappa theta <= 0
    }

    void fillResidual(std::size_t i, const Eigen::VectorXd& b, const Eigen::VectorXd& a,
                      quantape::math::CalibrationOrder order,
                      quantape::math::CalibrationResidualBlock& out) const {
        const HestonCalibrationQuote& q = m_quotes[i];
        const double sw = std::sqrt(q.weight);
        const HestonFullPoint p = point(i, b, a);
        out.value = sw * (m_model->callFull(p, q.tMax) - q.target);
        if (order >= quantape::math::CalibrationOrder::Gradient) {
            const auto g = m_model->fullGradient(p, q.tMax);
            out.gradientB.resize(5);
            out.gradientA.resize(3);
            for (std::size_t k = 0; k < 5; ++k) {
                out.gradientB(static_cast<Eigen::Index>(k)) = sw * g(0, static_cast<int>(k));
            }
            out.gradientA(0) = sw * g(0, HESTON_SPOT);
            out.gradientA(1) = sw * g(0, HESTON_RATE);
            out.gradientA(2) = sw * g(0, HESTON_DIVIDEND);
        }
        if (order >= quantape::math::CalibrationOrder::Hessian) {
            const auto H = m_model->fullHessian(p, q.tMax);
            out.hessianBB.resize(5, 5);
            out.hessianBA.resize(5, 3);
            for (std::size_t k = 0; k < 5; ++k) {
                for (std::size_t l = 0; l < 5; ++l) {
                    out.hessianBB(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(l)) =
                        sw * H(static_cast<int>(k), static_cast<int>(l));
                }
            }
            const int market[3] = {HESTON_SPOT, HESTON_RATE, HESTON_DIVIDEND};
            for (std::size_t k = 0; k < 5; ++k) {
                for (int m = 0; m < 3; ++m) {
                    out.hessianBA(static_cast<Eigen::Index>(k), m) =
                        sw * H(static_cast<int>(k), market[m]);
                }
            }
        }
    }

    /// Feller: g(b) = sigma^2 - 2 kappa theta <= 0.
    void fillInequality(std::size_t /*k*/, const Eigen::VectorXd& b,
                        quantape::math::CalibrationInequalityBlock& out) const {
        const double kappa = b(1);
        const double theta = b(2);
        const double sigma = b(3);
        out.value = sigma * sigma - 2.0 * kappa * theta;
        out.gradientB.resize(5);
        out.gradientB << 0.0, -2.0 * theta, -2.0 * kappa, 2.0 * sigma, 0.0;
        out.hessianBB.setZero(5, 5);
        out.hessianBB(1, 2) = -2.0;
        out.hessianBB(2, 1) = -2.0;
        out.hessianBB(3, 3) = 2.0;
    }

    const HestonModel& model() const { return *m_model; }
    const std::vector<HestonCalibrationQuote>& quotes() const { return m_quotes; }

private:
    template <typename S>
    S priceScalar(std::size_t i, const std::vector<S>& b, const std::vector<S>& a) const {
        if constexpr (std::is_same_v<S, double>) {
            HestonFullPoint p{};
            for (std::size_t k = 0; k < 5; ++k) {
                p[k] = b[k];
            }
            p[HESTON_SPOT] = a[0];
            p[HESTON_STRIKE] = m_quotes[i].strike;
            p[HESTON_RATE] = a[1];
            p[HESTON_DIVIDEND] = a[2];
            return m_model->callFull(p, m_quotes[i].tMax);
        } else if constexpr (std::is_same_v<S, stan::math::var>) {
            // Primal-point cache: repeated evaluations at the same (b, a)
            // (HVPs, mixed passes, line-search restarts) reuse one build.
            Eigen::Matrix<stan::math::var, HESTON_PARAM_COUNT, 1> v;
            for (std::size_t k = 0; k < 5; ++k) {
                v(static_cast<Eigen::Index>(k)) = b[k];
            }
            v(HESTON_SPOT) = a[0];
            v(HESTON_STRIKE) = m_quotes[i].strike;
            v(HESTON_RATE) = a[1];
            v(HESTON_DIVIDEND) = a[2];
            return m_cache.price(v, m_quotes[i].tMax);
        } else if constexpr (std::is_same_v<S, stan::math::fvar<stan::math::var>>) {
            Eigen::Matrix<stan::math::fvar<stan::math::var>, HESTON_PARAM_COUNT, 1> v;
            for (std::size_t k = 0; k < 5; ++k) {
                v(static_cast<Eigen::Index>(k)) = b[k];
            }
            v(HESTON_SPOT) = a[0];
            v(HESTON_STRIKE) = m_quotes[i].strike;
            v(HESTON_RATE) = a[1];
            v(HESTON_DIVIDEND) = a[2];
            return m_cache.price(v, m_quotes[i].tMax);
        } else {
            static_assert(std::is_same_v<S, double>,
                          "HestonCalibrationProblem: supported scalars are double, var, fvar<var>");
            return S(0.0);
        }
    }

    const HestonModel* m_model;
    std::vector<HestonCalibrationQuote> m_quotes;
    /// One cached 9-parameter build per (b, a) primal point (mutable: the
    /// scalar-generic view is logically const). One adapter per thread.
    mutable quantape::models::HestonSourceCache m_cache{*m_model};
};

/**
 * @brief Fixed-market variant: `b` = the five model parameters only
 *        (`numMarketParams() == 0`).
 *
 * Same generic contract, but the derivatives use the model-only analytic
 * kernels (`call`/`callGradient`/`callHessian`): 1 + 10 + 31 complex
 * evaluations per quote instead of 1 + 18 + 91 for the nine-parameter form.
 * This is the calibration hot path when S, r and q are treated as known.
 * AD scalars price through the five-parameter `make_callback_var` overloads.
 */
class HestonModelCalibrationProblem {
public:
    HestonModelCalibrationProblem(const HestonModel& model,
                                  std::vector<HestonCalibrationQuote> quotes, double spot,
                                  double rate, double dividend)
        : m_model(&model), m_quotes(std::move(quotes)), m_spot(spot), m_rate(rate),
          m_dividend(dividend) {}

    std::size_t numModelParams() const { return 5; }
    std::size_t numMarketParams() const { return 0; }
    std::size_t numQuotes() const { return m_quotes.size(); }
    std::size_t numInequalities() const { return 1; }

    HestonMarket market(std::size_t i) const {
        return {m_spot, m_quotes[i].strike, m_rate, m_dividend, m_quotes[i].tMax};
    }

    void fillResidual(std::size_t i, const Eigen::VectorXd& b, const Eigen::VectorXd&,
                      quantape::math::CalibrationOrder order,
                      quantape::math::CalibrationResidualBlock& out) const {
        const HestonCalibrationQuote& q = m_quotes[i];
        const double sw = std::sqrt(q.weight);
        const HestonParams params{b(0), b(1), b(2), b(3), b(4)};
        const HestonMarket m = market(i);
        out.value = sw * (m_model->call(params, m) - q.target);
        if (order >= quantape::math::CalibrationOrder::Gradient) {
            const auto g = m_model->callGradient(params, m);
            out.gradientB.resize(5);
            for (Eigen::Index k = 0; k < 5; ++k) {
                out.gradientB(k) = sw * g(0, k);
            }
            out.gradientA.resize(0);
        }
        if (order >= quantape::math::CalibrationOrder::Hessian) {
            const auto H = m_model->callHessian(params, m);
            out.hessianBB.resize(5, 5);
            for (Eigen::Index k = 0; k < 5; ++k) {
                for (Eigen::Index l = 0; l < 5; ++l) {
                    out.hessianBB(k, l) = sw * H(k, l);
                }
            }
            out.hessianBA.resize(5, 0);
        }
    }

    void fillInequality(std::size_t, const Eigen::VectorXd& b,
                        quantape::math::CalibrationInequalityBlock& out) const {
        const double kappa = b(1), theta = b(2), sigma = b(3);
        out.value = sigma * sigma - 2.0 * kappa * theta;
        out.gradientB.resize(5);
        out.gradientB << 0.0, -2.0 * theta, -2.0 * kappa, 2.0 * sigma, 0.0;
        out.hessianBB.setZero(5, 5);
        out.hessianBB(1, 2) = -2.0;
        out.hessianBB(2, 1) = -2.0;
        out.hessianBB(3, 3) = 2.0;
    }

    // ── Scalar-generic (AD) view: the five-parameter model-only prices ──

    template <typename S>
    S residual(std::size_t i, const std::vector<S>& b, const std::vector<S>&) const {
        const HestonCalibrationQuote& q = m_quotes[i];
        return S(std::sqrt(q.weight)) * (priceScalar<S>(i, b) - S(q.target));
    }

    template <typename S>
    void inequalities(const std::vector<S>& b, std::vector<S>& out) const {
        out.resize(1);
        out[0] = b[3] * b[3] - S(2.0) * b[1] * b[2];
    }

    const HestonModel& model() const { return *m_model; }

private:
    template <typename S>
    S priceScalar(std::size_t i, const std::vector<S>& b) const {
        const HestonMarket m = market(i);
        if constexpr (std::is_same_v<S, double>) {
            return m_model->call(HestonParams{b[0], b[1], b[2], b[3], b[4]}, m);
        } else if constexpr (std::is_same_v<S, stan::math::var>) {
            Eigen::Matrix<stan::math::var, 5, 1> v;
            for (int k = 0; k < 5; ++k) {
                v(k) = b[static_cast<std::size_t>(k)];
            }
            return m_cache.price(v, m);
        } else if constexpr (std::is_same_v<S, stan::math::fvar<stan::math::var>>) {
            Eigen::Matrix<stan::math::fvar<stan::math::var>, 5, 1> v;
            for (int k = 0; k < 5; ++k) {
                v(k) = b[static_cast<std::size_t>(k)];
            }
            return m_cache.price(v, m);
        } else {
            static_assert(std::is_same_v<S, double>,
                          "HestonModelCalibrationProblem: supported scalars are double, var, "
                          "fvar<var>");
            return S(0.0);
        }
    }

    const HestonModel* m_model;
    std::vector<HestonCalibrationQuote> m_quotes;
    double m_spot = 1.0;
    double m_rate = 0.0;
    double m_dividend = 0.0;
    /// Cached model-only builds keyed by (params, market); see above.
    mutable quantape::models::HestonSourceCache m_cache{*m_model};
};

/**
 * @brief Quote-price chain: `b` = the five model parameters, `a` = the quote
 *        prices themselves (`numMarketParams() == numQuotes()`).
 *
 * Residual `r_i = sqrt(w_i) (P_i(b) - a_i)`: the data move with the quotes
 * (`dI/da = -sqrt(w_i) e_i`), which is the standard calibration-chain setup
 * for market-risk propagation (Savine; `test_market_risk_ift`). The fixed
 * market (S, r, q) is supplied at construction.
 */
class HestonQuotePriceCalibrationProblem {
public:
    HestonQuotePriceCalibrationProblem(const HestonModel& model,
                                       std::vector<HestonCalibrationQuote> quotes, double spot,
                                       double rate, double dividend)
        : m_model(&model), m_quotes(std::move(quotes)), m_spot(spot), m_rate(rate),
          m_dividend(dividend) {}

    std::size_t numModelParams() const { return 5; }
    std::size_t numMarketParams() const { return m_quotes.size(); }
    std::size_t numQuotes() const { return m_quotes.size(); }
    std::size_t numInequalities() const { return 1; }

    void fillResidual(std::size_t i, const Eigen::VectorXd& b, const Eigen::VectorXd& a,
                      quantape::math::CalibrationOrder order,
                      quantape::math::CalibrationResidualBlock& out) const {
        const HestonCalibrationQuote& q = m_quotes[i];
        const double sw = std::sqrt(q.weight);
        const HestonParams params{b(0), b(1), b(2), b(3), b(4)};
        const HestonMarket market{m_spot, q.strike, m_rate, m_dividend, q.tMax};
        out.value = sw * (m_model->call(params, market) - a(static_cast<Eigen::Index>(i)));
        if (order >= quantape::math::CalibrationOrder::Gradient) {
            const auto g = m_model->callGradient(params, market);
            out.gradientB.resize(5);
            for (Eigen::Index k = 0; k < 5; ++k) {
                out.gradientB(k) = sw * g(0, k);
            }
            out.gradientA.setZero(static_cast<Eigen::Index>(m_quotes.size()));
            out.gradientA(static_cast<Eigen::Index>(i)) = -sw;
        }
        if (order >= quantape::math::CalibrationOrder::Hessian) {
            const auto H = m_model->callHessian(params, market);
            out.hessianBB = sw * H;
            out.hessianBA.setZero(5, static_cast<Eigen::Index>(m_quotes.size()));
        }
    }

    void fillInequality(std::size_t, const Eigen::VectorXd& b,
                        quantape::math::CalibrationInequalityBlock& out) const {
        const double kappa = b(1), theta = b(2), sigma = b(3);
        out.value = sigma * sigma - 2.0 * kappa * theta;
        out.gradientB.resize(5);
        out.gradientB << 0.0, -2.0 * theta, -2.0 * kappa, 2.0 * sigma, 0.0;
        out.hessianBB.setZero(5, 5);
        out.hessianBB(1, 2) = -2.0;
        out.hessianBB(2, 1) = -2.0;
        out.hessianBB(3, 3) = 2.0;
    }

    /// Scalar-generic view (quote prices are data; the market is fixed).
    template <typename S>
    S residual(std::size_t i, const std::vector<S>& b, const std::vector<S>& a) const {
        const HestonCalibrationQuote& q = m_quotes[i];
        const HestonMarket market{m_spot, q.strike, m_rate, m_dividend, q.tMax};
        const S price = priceScalar<S>(i, b);
        (void)market;
        return S(std::sqrt(q.weight)) * (price - a[i]);
    }

    template <typename S>
    void inequalities(const std::vector<S>& b, std::vector<S>& out) const {
        out.resize(1);
        out[0] = b[3] * b[3] - S(2.0) * b[1] * b[2];
    }

private:
    template <typename S>
    S priceScalar(std::size_t i, const std::vector<S>& b) const {
        const HestonMarket market{m_spot, m_quotes[i].strike, m_rate, m_dividend, m_quotes[i].tMax};
        if constexpr (std::is_same_v<S, double>) {
            return m_model->call(HestonParams{b[0], b[1], b[2], b[3], b[4]}, market);
        } else if constexpr (std::is_same_v<S, stan::math::var>) {
            Eigen::Matrix<stan::math::var, 5, 1> v;
            for (int k = 0; k < 5; ++k) {
                v(k) = b[static_cast<std::size_t>(k)];
            }
            return m_cache.price(v, market);
        } else if constexpr (std::is_same_v<S, stan::math::fvar<stan::math::var>>) {
            Eigen::Matrix<stan::math::fvar<stan::math::var>, 5, 1> v;
            for (int k = 0; k < 5; ++k) {
                v(k) = b[static_cast<std::size_t>(k)];
            }
            return m_cache.price(v, market);
        } else {
            static_assert(std::is_same_v<S, double>,
                          "HestonQuotePriceCalibrationProblem: supported scalars are double, var, "
                          "fvar<var>");
            return S(0.0);
        }
    }

    const HestonModel* m_model;
    std::vector<HestonCalibrationQuote> m_quotes;
    double m_spot = 1.0;
    double m_rate = 0.0;
    double m_dividend = 0.0;
    mutable quantape::models::HestonSourceCache m_cache{*m_model};
};

} // namespace quantape::models

#endif // QUANTAPE_MODELS_HESTON_CALIBRATION_H
