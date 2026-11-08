#ifndef QUANTAPE_MATH_DISTRIBUTIONS_CHI_SQUARE_H
#define QUANTAPE_MATH_DISTRIBUTIONS_CHI_SQUARE_H

#include "quantape/math/Distributions/Gamma.h"
#include "quantape/math/Distributions/detail/IncompleteGamma.h"

#include <cmath>
#include <type_traits>

namespace quantape::math {
/**
 * @file ChiSquare.h
 * @brief Chi-square law: density, CDF, log tails and quantile.
 *
 * `ChiSquare(df)` is `Gamma(shape = df/2, rate = 1/2)`:
 *   `pdf(x) = x^(df/2-1) e^(-x/2) / (2^(df/2) Gamma(df/2))`,
 *   `cdf(x) = P(df/2, x/2)`, `ccdf(x) = Q(df/2, x/2)`,
 *   `inv(p)` = Wilson-Hilferty + safeguarded Newton (see `Gamma::inv`).
 *
 * The law delegates to the Gamma kernels, so accuracy, tail behavior and the
 * scalar-generic `T` handling are exactly those documented in Gamma.h;
 * `ChiSquare<double>` dispatches every method to the named
 * `detail::chiSquare*Double` entry (the Gamma double tier at `df/2`, rate
 * `1/2`), other scalars use the generic `Gamma<T>` path -- including `inv`'s
 * double solve followed by the taped final Newton steps (adjoint
 * `dQ/dp = 1/pdf(Q)`). The double-only derivative packs
 * `detail::chiSquarePdfD1D2` / `detail::chiSquareCdfD1D2` are the Gamma packs
 * at the same parameters.
 */
namespace detail {

/// ChiSquare(df) density at `T = double`: `gammaPdfDouble(df/2, 1/2, x)`.
inline double chiSquarePdfDouble(double df, double x) {
    return gammaPdfDouble(df / 2.0, 0.5, x);
}

/// ChiSquare(df) log density at `T = double`.
inline double chiSquareLogPdfDouble(double df, double x) {
    return gammaLogPdfDouble(df / 2.0, 0.5, x);
}

/// ChiSquare(df) CDF at `T = double`: `gammaCdfDouble(df/2, 1/2, x)`.
inline double chiSquareCdfDouble(double df, double x) {
    return gammaCdfDouble(df / 2.0, 0.5, x);
}

/// ChiSquare(df) log CDF at `T = double`.
inline double chiSquareLogCdfDouble(double df, double x) {
    return gammaLogCdfDouble(df / 2.0, 0.5, x);
}

/// ChiSquare(df) ccdf at `T = double`.
inline double chiSquareCcdfDouble(double df, double x) {
    return gammaCcdfDouble(df / 2.0, 0.5, x);
}

/// ChiSquare(df) log ccdf at `T = double`.
inline double chiSquareLogCcdfDouble(double df, double x) {
    return gammaLogCcdfDouble(df / 2.0, 0.5, x);
}

/// ChiSquare(df) quantile at `T = double`: `gammaInvDouble(df/2, 1/2, p)`.
inline double chiSquareInvDouble(double df, double p) {
    return gammaInvDouble(df / 2.0, 0.5, p);
}

} // namespace detail

namespace detail {
struct ChiSquareInvGradHess;
ChiSquareInvGradHess chiSquareInvGradHess(double df, double p);
} // namespace detail

/**
 * @brief ChiSquare(df) with pdf/cdf/ccdf/log forms and quantile.
 *
 * `DoubleT = double` dispatches to the named `detail::chiSquare*Double`
 * entries; other scalars delegate to `Gamma<T>` (generic scalar path).
 */
template <class T>
struct ChiSquare {
    T df;

    T pdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::chiSquarePdfDouble(df, x);
        } else {
            return law().pdf(x);
        }
    }

    T logPdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::chiSquareLogPdfDouble(df, x);
        } else {
            return law().logPdf(x);
        }
    }

    T cdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::chiSquareCdfDouble(df, x);
        } else {
            return law().cdf(x);
        }
    }

    T logCdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::chiSquareLogCdfDouble(df, x);
        } else {
            return law().logCdf(x);
        }
    }

    T ccdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::chiSquareCcdfDouble(df, x);
        } else {
            return law().ccdf(x);
        }
    }

    T logCcdf(T x) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::chiSquareLogCcdfDouble(df, x);
        } else {
            return law().logCcdf(x);
        }
    }

    T inv(T p) const {
        if constexpr (std::is_same_v<T, double>) {
            return detail::chiSquareInvDouble(df, p);
        } else {
            return law().inv(p);
        }
    }

    /// double-only: full gradient/Hessian of the quantile in (p, df).
    detail::ChiSquareInvGradHess invGradHess(double p) const;

private:
    Gamma<T> law() const { return Gamma<T>{df / T(2), T(1) / T(2)}; }
};

namespace detail {
/**
 * @brief `{f, f', f''}` for the ChiSquare(df) density.
 *
 * `f'/f = (df/2 - 1)/x - 1/2` and `f''/f = q^2 - (df/2 - 1)/x^2`, i.e. the
 * `gammaPdfD1D2(df/2, 1/2, x)` pack. All derivatives analytic.
 */
inline ValueD1D2 chiSquarePdfD1D2(double df, double x) {
    return gammaPdfD1D2(df / 2.0, 0.5, x);
}

/// `{F, F', F''}` for the ChiSquare(df) CDF: `gammaCdfD1D2(df/2, 1/2, x)`.
inline ValueD1D2 chiSquareCdfD1D2(double df, double x) {
    return gammaCdfD1D2(df / 2.0, 0.5, x);
}

// ---------------------------------------------------------------------------
// ChiSquare(df) parameter-derivative packs (central chi-square).
//
// `ChiSquare(df) = Gamma(shape = df/2, rate = 1/2)`, so every entry is the
// corresponding Gamma pack entry at `a = df/2`, `rate = 1/2`, scaled by the
// chain rule `d/ddf = (1/2) d/da`:
//   `gDf = gShape / 2`, `hDfDf = hShapeShape / 4`, `hDfX = hShapeX / 2`;
// the x entries are unchanged. Edges follow the Gamma packs (NaN for invalid
// df or x = 0, zero for x < 0, one at x = +Inf for the CDF).
// ---------------------------------------------------------------------------

/// (df, x) value + gradient + Hessian for the ChiSquare CDF.
struct ChiSquareCdfGradHess {
    double value = 0.0;
    double gDf = 0.0;
    double gx = 0.0;
    double hDfDf = 0.0;
    double hDfX = 0.0;
    double hxx = 0.0;
};

/// Exact gradient/Hessian of the ChiSquare(df) CDF at `x`; see section comment.
inline ChiSquareCdfGradHess chiSquareCdfGradHess(double df, double x) noexcept {
    const GammaCdfGradHess g = gammaCdfGradHess(df / 2.0, 0.5, x);
    ChiSquareCdfGradHess out{};
    out.value = g.value;
    out.gDf = 0.5 * g.gShape;
    out.gx = g.gx;
    out.hDfDf = 0.25 * g.hShapeShape;
    out.hDfX = 0.5 * g.hShapeX;
    out.hxx = g.hxx;
    return out;
}

/// (df, x) value + gradient + Hessian for the ChiSquare density.
struct ChiSquarePdfGradHess {
    double value = 0.0;
    double gDf = 0.0;
    double gx = 0.0;
    double hDfDf = 0.0;
    double hDfX = 0.0;
    double hxx = 0.0;
};

/// Exact gradient/Hessian of the ChiSquare(df) density at `x`; see section
/// comment. `gDf`/`hDfDf`/`hDfX` are the Gamma shape entries scaled by
/// `1/2`/`1/4`/`1/2`; `value`, `gx`, `hxx` are the Gamma density entries.
inline ChiSquarePdfGradHess chiSquarePdfGradHess(double df, double x) noexcept {
    const GammaPdfGradHess g = gammaPdfGradHess(df / 2.0, 0.5, x);
    ChiSquarePdfGradHess out{};
    out.value = g.value;
    out.gDf = 0.5 * g.gShape;
    out.gx = g.gx;
    out.hDfDf = 0.25 * g.hShapeShape;
    out.hDfX = 0.5 * g.hShapeX;
    out.hxx = g.hxx;
    return out;
}

// ---------------------------------------------------------------------------
// ChiSquare quantile parameter-derivative pack.
//
// Same implicit differentiation as in Normal.h: with `f = F_x`, `f' = F_xx`
// at `x = Q`, `Q_p = 1/f`, `Q_df = -F_df/f`,
// `Q_pp = -f'/f^3`, `Q_p,df = -(F_df,x + f' Q_df)/f^2`,
// `Q_dfdf = -(F_dfdf + 2 F_df,x Q_df + f' Q_df^2)/f`, with the df partials
// from `chiSquareCdfGradHess` at the quantile. The `value` is bitwise
// `ChiSquare::inv(p)`; outside the open unit interval the derivative entries
// are NaN (`value` keeps the 0/+Inf endpoint convention). No finite
// differences.
// ---------------------------------------------------------------------------

/// (p, df) value + gradient + Hessian of the chi-square quantile.
struct ChiSquareInvGradHess {
    double value = 0.0;
    double gP = 0.0;
    double gDf = 0.0;
    double hPP = 0.0;
    double hPDf = 0.0;
    double hDfDf = 0.0;
};

/// `{Q, Q_p, Q_df, Q_pp, Q_p,df, Q_dfdf}` for the chi-square quantile; see
/// the section comment.
inline ChiSquareInvGradHess chiSquareInvGradHess(double df, double p) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ChiSquareInvGradHess out;
    out.value = chiSquareInvDouble(df, p);
    if (!(p > 0.0) || !(p < 1.0)) {
        out.gP = nan;
        out.gDf = nan;
        out.hPP = nan;
        out.hPDf = nan;
        out.hDfDf = nan;
        return out;
    }
    const ChiSquareCdfGradHess c = chiSquareCdfGradHess(df, out.value);
    const double f = c.gx;
    const double fp = c.hxx;
    out.gP = 1.0 / f;
    out.gDf = -c.gDf / f;
    out.hPP = -fp / (f * f * f);
    out.hPDf = -(c.hDfX + fp * out.gDf) / (f * f);
    out.hDfDf = -(c.hDfDf + 2.0 * c.hDfX * out.gDf + fp * out.gDf * out.gDf) / f;
    return out;
}

} // namespace detail

template <class T>
detail::ChiSquareInvGradHess ChiSquare<T>::invGradHess(double p) const {
    static_assert(std::is_same_v<T, double>, "invGradHess analytic pack requires T = double");
    return detail::chiSquareInvGradHess(df, p);
}

} // namespace quantape::math

#endif // QUANTAPE_MATH_DISTRIBUTIONS_CHI_SQUARE_H
