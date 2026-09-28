#ifndef QUANTAPE_MODELS_HESTON_STAN_PRIMITIVES_H
#define QUANTAPE_MODELS_HESTON_STAN_PRIMITIVES_H

#include "quantape/math/StanMath.h"

#include "quantape/models/HestonModel.h"

#include <array>

namespace quantape::models {
/**
 * @file HestonStanPrimitives.h
 * @brief Stan AD instantiation of the Heston pricer (GBS-style callback vars)
 *
 * The value, gradient and Hessian are computed in double by
 * `HestonModel` (complex-step Jacobians, no tape through the quadrature)
 * and stitched into the Stan tape with `make_callback_var`, exactly as the
 * `GBS`/`Black76` specializations in `pricing/StanPrimitives.h`:
 *
 *  - `var`: value node plus one callback that pushes the analytical
 *    gradient into the nine leaves; the tape holds a single node instead of
 *    thousands of nodes through exp/log/sqrt of every quadrature node.
 *  - `fvar<var>`: adds one tangent node whose adjoint pushes `H . d`
 *    (the Hessian column combination `stan::math::hessian` walks).
 *
 * The full parameter vector is `[v0, kappa, theta, sigma, rho, S, K, r, q]`
 * (`HestonFullPoint`); `tMax` is a quote-level double.
 */

using HestonStanVector = Eigen::Matrix<stan::math::var, HESTON_PARAM_COUNT, 1>;
using HestonStanFvarVector =
    Eigen::Matrix<stan::math::fvar<stan::math::var>, HESTON_PARAM_COUNT, 1>;

/// Plain-double entry point (mirrors `callFull` on an Eigen vector).
inline double hestonCall(const HestonModel& model,
                         const Eigen::Matrix<double, HESTON_PARAM_COUNT, 1>& x, double tMax) {
    HestonFullPoint point{};
    for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
        point[static_cast<std::size_t>(i)] = x(i);
    }
    return model.callFull(point, tMax);
}

/// Value + analytical adjoint on a 1-node tape.
inline stan::math::var hestonCall(const HestonModel& model, const HestonStanVector& x,
                                  double tMax) {
    using stan::math::make_callback_var;
    using stan::math::var;

    HestonFullPoint point{};
    for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
        point[static_cast<std::size_t>(i)] = x(i).val();
    }
    const double value = model.callFull(point, tMax);
    const Eigen::Matrix<double, 1, HESTON_PARAM_COUNT> g1 = model.fullGradient(point, tMax);

    std::array<stan::math::vari*, HESTON_PARAM_COUNT> leaves{};
    for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
        leaves[static_cast<std::size_t>(i)] = x(i).vi_;
    }
    return make_callback_var(value, [leaves, g1](auto& vi) {
        const double adj = vi.adj();
        for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
            leaves[static_cast<std::size_t>(i)]->adj_ += adj * g1(i);
        }
    });
}

/// Nested analytical form used by `stan::math::hessian`: value node plus a
/// single tangent node that pushes `H . d` into the leaves.
inline stan::math::fvar<stan::math::var> hestonCall(const HestonModel& model,
                                                    const HestonStanFvarVector& x, double tMax) {
    using stan::math::fvar;
    using stan::math::make_callback_var;
    using stan::math::var;

    HestonFullPoint point{};
    for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
        point[static_cast<std::size_t>(i)] = x(i).val_.val();
    }
    const double value = model.callFull(point, tMax);
    const Eigen::Matrix<double, 1, HESTON_PARAM_COUNT> g1 = model.fullGradient(point, tMax);
    const Eigen::Matrix<double, HESTON_PARAM_COUNT, HESTON_PARAM_COUNT> g2 =
        model.fullHessian(point, tMax);

    std::array<stan::math::vari*, HESTON_PARAM_COUNT> leaves{};
    for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
        leaves[static_cast<std::size_t>(i)] = x(i).val_.vi_;
    }
    // Value node ALSO carries the analytical adjoint: nonlinear compositions
    // of the price (objectives, payoffs) then get the outer-curvature term
    // (d2f/dp2)(dp/dx)(dp/dx) from the value pass, while the tangent node
    // supplies (df/dp)(d2p/dx2).
    var price = make_callback_var(value, [leaves, g1](auto& vi) {
        const double a = vi.adj();
        for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
            leaves[static_cast<std::size_t>(i)]->adj_ += a * g1(i);
        }
    });

    std::array<double, HESTON_PARAM_COUNT> w{};
    double tval = 0.0;
    for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
        w[static_cast<std::size_t>(i)] = x(i).d_.val();
        tval += g1(i) * w[static_cast<std::size_t>(i)];
    }

    var tangent = make_callback_var(tval, [leaves, g2, w](auto& vi) {
        const double a = vi.adj();
        for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
            double row = 0.0;
            for (int j = 0; j < HESTON_PARAM_COUNT; ++j) {
                row += g2(i, j) * w[static_cast<std::size_t>(j)];
            }
            leaves[static_cast<std::size_t>(i)]->adj_ += a * row;
        }
    });

    return fvar<var>(price, tangent);
}

// ── Model-only (market-fixed) form used by Heston calibrations ──

using HestonStanModelVector = Eigen::Matrix<stan::math::var, 5, 1>;
using HestonStanModelFvarVector = Eigen::Matrix<stan::math::fvar<stan::math::var>, 5, 1>;

/// `var` price with the market leg held fixed: value + precomputed 5x5
/// gradient (`callGradient`), one tape node.
inline stan::math::var hestonCall(const HestonModel& model, const HestonStanModelVector& x,
                                  const HestonMarket& market) {
    using stan::math::var;

    const HestonParams params{x(0).val(), x(1).val(), x(2).val(), x(3).val(), x(4).val()};
    const double value = model.call(params, market);
    const Eigen::Matrix<double, 1, 5> g1 = model.callGradient(params, market);

    std::array<stan::math::vari*, 5> leaves{};
    for (int i = 0; i < 5; ++i) {
        leaves[static_cast<std::size_t>(i)] = x(i).vi_;
    }
    return stan::math::make_callback_var(value, [leaves, g1](auto& vi) {
        const double adj = vi.adj();
        for (int i = 0; i < 5; ++i) {
            leaves[static_cast<std::size_t>(i)]->adj_ += adj * g1(i);
        }
    });
}

/// `fvar<var>` model-only price: value node plus a tangent node pushing
/// `H . d` (5x5 `callHessian`) for `stan::math::hessian`.
inline stan::math::fvar<stan::math::var> hestonCall(const HestonModel& model,
                                                    const HestonStanModelFvarVector& x,
                                                    const HestonMarket& market) {
    using stan::math::fvar;
    using stan::math::var;

    const HestonParams params{x(0).val_.val(), x(1).val_.val(), x(2).val_.val(), x(3).val_.val(),
                              x(4).val_.val()};
    const double value = model.call(params, market);
    const Eigen::Matrix<double, 1, 5> g1 = model.callGradient(params, market);
    const Eigen::Matrix<double, 5, 5> g2 = model.callHessian(params, market);

    std::array<stan::math::vari*, 5> leaves{};
    for (int i = 0; i < 5; ++i) {
        leaves[static_cast<std::size_t>(i)] = x(i).val_.vi_;
    }
    // Value node carries the analytical adjoint too (see the 9-parameter
    // overload): required for nonlinear compositions under fvar<var>.
    var price = stan::math::make_callback_var(value, [leaves, g1](auto& vi) {
        const double a = vi.adj();
        for (int i = 0; i < 5; ++i) {
            leaves[static_cast<std::size_t>(i)]->adj_ += a * g1(i);
        }
    });

    std::array<double, 5> w{};
    double tval = 0.0;
    for (int i = 0; i < 5; ++i) {
        w[static_cast<std::size_t>(i)] = x(i).d_.val();
        tval += g1(i) * w[static_cast<std::size_t>(i)];
    }

    var tangent = stan::math::make_callback_var(tval, [leaves, g2, w](auto& vi) {
        const double a = vi.adj();
        for (int i = 0; i < 5; ++i) {
            double row = 0.0;
            for (int j = 0; j < 5; ++j) {
                row += g2(i, j) * w[static_cast<std::size_t>(j)];
            }
            leaves[static_cast<std::size_t>(i)]->adj_ += a * row;
        }
    });

    return fvar<var>(price, tangent);
}

// ── Primal-point cache: one build, many tangent wires ──
//
// `stan::math::hessian` calls the functor once per column plus the base
// evaluation, always at the same primal point. Rebuilding value + gradient +
// Hessian in double for each call dominates the wall time (the tape itself
// is 2 nodes). `HestonStanCache` memoizes the double layer keyed by the
// primal point and only rewires the callback graph, turning the 9x9 Hessian
// from ten builds into one. One instance per thread/objective; not
// thread-safe.

class HestonSourceCache {
public:
    explicit HestonSourceCache(const HestonModel& model) : model_(&model) {}

    /// Nine-parameter price (`[v0, kappa, theta, sigma, rho, S, K, r, q]`).
    stan::math::var price(const HestonStanVector& x, double tMax) {
        ensure(buildPoint(x), tMax);
        std::array<stan::math::vari*, HESTON_PARAM_COUNT> leaves{};
        for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
            leaves[static_cast<std::size_t>(i)] = x(i).vi_;
        }
        return stan::math::make_callback_var(value_, [leaves, g1 = g1_](auto& vi) {
            const double a = vi.adj();
            for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
                leaves[static_cast<std::size_t>(i)]->adj_ += a * g1(i);
            }
        });
    }

    stan::math::fvar<stan::math::var> price(const HestonStanFvarVector& x, double tMax) {
        ensure(buildPoint(x), tMax);
        std::array<stan::math::vari*, HESTON_PARAM_COUNT> leaves{};
        std::array<double, HESTON_PARAM_COUNT> w{};
        double tval = 0.0;
        for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
            leaves[static_cast<std::size_t>(i)] = x(i).val_.vi_;
            w[static_cast<std::size_t>(i)] = x(i).d_.val();
            tval += g1_(i) * w[static_cast<std::size_t>(i)];
        }
        stan::math::var value = stan::math::make_callback_var(value_, [leaves, g1 = g1_](auto& vi) {
            const double a = vi.adj();
            for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
                leaves[static_cast<std::size_t>(i)]->adj_ += a * g1(i);
            }
        });
        stan::math::var tangent =
            stan::math::make_callback_var(tval, [leaves, g2 = g2_, w](auto& vi) {
                const double a = vi.adj();
                for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
                    double row = 0.0;
                    for (int j = 0; j < HESTON_PARAM_COUNT; ++j) {
                        row += g2(i, j) * w[static_cast<std::size_t>(j)];
                    }
                    leaves[static_cast<std::size_t>(i)]->adj_ += a * row;
                }
            });
        return stan::math::fvar<stan::math::var>(value, tangent);
    }

    /// Five-parameter model-only price (market fixed).
    stan::math::var price(const HestonStanModelVector& x, const HestonMarket& market) {
        ensure(buildPoint(x), market);
        std::array<stan::math::vari*, 5> leaves{};
        for (int i = 0; i < 5; ++i) {
            leaves[static_cast<std::size_t>(i)] = x(i).vi_;
        }
        return stan::math::make_callback_var(valueModel_, [leaves, g1 = g1Model_](auto& vi) {
            const double a = vi.adj();
            for (int i = 0; i < 5; ++i) {
                leaves[static_cast<std::size_t>(i)]->adj_ += a * g1(i);
            }
        });
    }

    stan::math::fvar<stan::math::var> price(const HestonStanModelFvarVector& x,
                                            const HestonMarket& market) {
        ensure(buildPoint(x), market);
        std::array<stan::math::vari*, 5> leaves{};
        std::array<double, 5> w{};
        double tval = 0.0;
        for (int i = 0; i < 5; ++i) {
            leaves[static_cast<std::size_t>(i)] = x(i).val_.vi_;
            w[static_cast<std::size_t>(i)] = x(i).d_.val();
            tval += g1Model_(i) * w[static_cast<std::size_t>(i)];
        }
        stan::math::var value =
            stan::math::make_callback_var(valueModel_, [leaves, g1 = g1Model_](auto& vi) {
                const double a = vi.adj();
                for (int i = 0; i < 5; ++i) {
                    leaves[static_cast<std::size_t>(i)]->adj_ += a * g1(i);
                }
            });
        stan::math::var tangent =
            stan::math::make_callback_var(tval, [leaves, g2 = g2Model_, w](auto& vi) {
                const double a = vi.adj();
                for (int i = 0; i < 5; ++i) {
                    double row = 0.0;
                    for (int j = 0; j < 5; ++j) {
                        row += g2(i, j) * w[static_cast<std::size_t>(j)];
                    }
                    leaves[static_cast<std::size_t>(i)]->adj_ += a * row;
                }
            });
        return stan::math::fvar<stan::math::var>(value, tangent);
    }

    std::size_t rebuilds() const { return rebuilds_; }

private:
    static HestonFullPoint buildPoint(const HestonStanVector& x) {
        HestonFullPoint p{};
        for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
            p[static_cast<std::size_t>(i)] = x(i).val();
        }
        return p;
    }
    static HestonFullPoint buildPoint(const HestonStanFvarVector& x) {
        HestonFullPoint p{};
        for (int i = 0; i < HESTON_PARAM_COUNT; ++i) {
            p[static_cast<std::size_t>(i)] = x(i).val_.val();
        }
        return p;
    }
    static HestonParams buildParams(const HestonStanModelVector& x) {
        return {x(0).val(), x(1).val(), x(2).val(), x(3).val(), x(4).val()};
    }
    static HestonParams buildParams(const HestonStanModelFvarVector& x) {
        return {x(0).val_.val(), x(1).val_.val(), x(2).val_.val(), x(3).val_.val(),
                x(4).val_.val()};
    }
    static HestonFullPoint buildPoint(const HestonStanModelVector& x) {
        HestonFullPoint p{};
        for (int i = 0; i < 5; ++i) {
            p[static_cast<std::size_t>(i)] = x(i).val();
        }
        return p;
    }
    static HestonFullPoint buildPoint(const HestonStanModelFvarVector& x) {
        HestonFullPoint p{};
        for (int i = 0; i < 5; ++i) {
            p[static_cast<std::size_t>(i)] = x(i).val_.val();
        }
        return p;
    }

    void ensure(const HestonFullPoint& point, double tMax) {
        if (validFull_ && tMax == lastFullTMax_ && samePoint(point)) {
            return;
        }
        value_ = model_->callFull(point, tMax);
        g1_ = model_->fullGradient(point, tMax);
        g2_ = model_->fullHessian(point, tMax);
        lastFull_ = point;
        lastFullTMax_ = tMax;
        validFull_ = true;
        ++rebuilds_;
    }

    void ensure(const HestonFullPoint& point, const HestonMarket& market) {
        const HestonParams params{point[0], point[1], point[2], point[3], point[4]};
        const bool sameMarket =
            validModel_ && sameParams(params) && market.spot == lastMarket_.spot &&
            market.strike == lastMarket_.strike && market.rate == lastMarket_.rate &&
            market.dividend == lastMarket_.dividend && market.tMax == lastMarket_.tMax;
        if (sameMarket) {
            return;
        }
        valueModel_ = model_->call(params, market);
        g1Model_ = model_->callGradient(params, market);
        g2Model_ = model_->callHessian(params, market);
        lastParams_ = params;
        lastMarket_ = market;
        validModel_ = true;
        ++rebuilds_;
    }

    bool samePoint(const HestonFullPoint& point) const {
        for (std::size_t i = 0; i < HESTON_PARAM_COUNT; ++i) {
            if (point[i] != lastFull_[i]) {
                return false;
            }
        }
        return true;
    }
    bool sameParams(const HestonParams& params) const {
        return params.v0 == lastParams_.v0 && params.kappa == lastParams_.kappa &&
               params.theta == lastParams_.theta && params.sigma == lastParams_.sigma &&
               params.rho == lastParams_.rho;
    }

    const HestonModel* model_;
    bool validFull_ = false;
    HestonFullPoint lastFull_{};
    double lastFullTMax_ = 0.0;
    double value_ = 0.0;
    Eigen::Matrix<double, 1, HESTON_PARAM_COUNT> g1_;
    Eigen::Matrix<double, HESTON_PARAM_COUNT, HESTON_PARAM_COUNT> g2_;
    bool validModel_ = false;
    HestonParams lastParams_{};
    HestonMarket lastMarket_{};
    double valueModel_ = 0.0;
    Eigen::Matrix<double, 1, 5> g1Model_;
    Eigen::Matrix<double, 5, 5> g2Model_;
    std::size_t rebuilds_ = 0;
};

} // namespace quantape::models

#endif // QUANTAPE_MODELS_HESTON_STAN_PRIMITIVES_H
