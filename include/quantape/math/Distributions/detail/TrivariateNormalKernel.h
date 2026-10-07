#ifndef QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_TRIVARIATE_NORMAL_KERNEL_H
#define QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_TRIVARIATE_NORMAL_KERNEL_H

#include "quantape/math/Distributions/detail/BivariateNormalKernel.h"

#include <cmath>
#include <limits>

namespace quantape::math::detail {

/**
 * @file TrivariateNormalKernel.h
 * @brief Trivariate normal CDF value and analytic derivative packs.
 *
 * Phi3(h1,h2,h3; r12,r13,r23) = P(X1 <= h1, X2 <= h2, X3 <= h3) for standard
 * normal margins.  The value follows the published Genz (2004) reduction
 * (clean-room implementation, no Genz source copied): after a partial sort
 * that places the largest-magnitude correlation in r23, the identity
 *
 *     Phi3 = Phi(h1) Phi2(h2,h3; r23)
 *          + (1/2pi) Integral_0^1 f(x) dx ,
 *
 * is integrated, where with rua = asin(r12), rub = asin(r13),
 * r12(x) = sin(rua x), r13(x) = sin(rub x) and the auxiliary kernel
 * (Genz's `pntgnd`)
 *
 *     f(x) = rua pntgnd(h1,h2,h3; r13(x),r23,r12(x), 1-r12(x)^2)
 *          + rub pntgnd(h1,h3,h2; r12(x),r23,r13(x), 1-r13(x)^2),
 *
 *     pntgnd(ba,bb,bc; ra,rb,r,rrd) = exp(-ft/2) Phi(bt)   (guarded),
 *     ft = (ba - r bb)^2/rrd + bb^2,
 *     bt = (bc rrd + ba(r rb - ra) + bb(r ra - rb)) / sqrt(dt),
 *     dt = rrd (rrd - (ra-rb)^2 - 2 ra rb (1-r)).
 *
 * dt = (1-r^2) det(R) on the path; the published partial sort places the
 * largest |r| in the fixed slot and keeps the path inside the positive-definite
 * cone for all valid triples tested (2e6 random triples scanned along the
 * path), so dt > 0 throughout (up to rounding).  bt is the conditional
 * z-score of the third variable and ft the exponent of the product of the two
 * normal densities.  The quadrature is a
 * deterministic composite Gauss-Kronrod 15-point rule (QUADPACK constants)
 * with bisection of the largest-error panel until the summed embedded error
 * estimate drops below 1e-15 or 128 panels are used.
 *
 * Degenerate limits are exact: any h_i = -inf gives 0, h_i = +inf
 * marginalizes to the lower-dimensional CDF, r = 0 factors, |r| = 1 uses the
 * singular reductions, and |h1|+|h2|+|h3| ~ 0 uses the exact orthant form
 * 1/8 + (asin r12 + asin r13 + asin r23)/(4 pi).  The final value is clamped
 * to [0,1]; NaN inputs, |r_ij| > 1 and correlation triples with determinant
 * below -1e-12 return NaN.
 *
 * Derivatives (double, exact; no finite differences):
 *  - g_i = d/dh_i = phi(h_i) Phi2(A_i,B_i; rho_i) with the conditional
 *    standardization of the other two variables given X_i = h_i:
 *      A_1 = (h2-r12 h1)/s12, B_1 = (h3-r13 h1)/s13,
 *      rho_1 = (r23-r12 r13)/(s12 s13),  s_ij = sqrt(1-r_ij^2),
 *    and cyclic permutations for i = 2,3.
 *  - d/dr_ij = phi2(h_i,h_j;r_ij) Phi((h_k - alpha h_i - beta h_j)/s) with
 *    alpha,beta the regression of X_k on (X_i,X_j) and s the residual
 *    standard deviation in the same correlation matrix (Plackett).
 *  - the full 6x6 Hessian is the derivative of that closed-form gradient
 *    pack, evaluated analytically entry by entry; no integral differentiation
 *    is needed.  kTvnHessIndex(i,j) = i*(i+1)/2+j for i >= j, parameter order
 *    (h1,h2,h3,r12,r13,r23).
 *
 * Accuracy (double, measured): < 1e-14 for well-conditioned inputs; the
 * bivariate kernel's 1.4e-14 near-degenerate ceiling dominates for
 * |conditional rho| >= 0.925.  Graceful degradation for near-singular
 * correlation triples (the path integral stays valid; conditioning limits the
 * last digits).  No allocation, no std::function, deterministic.
 */

using ::quantape::util::kPi;
using ::quantape::util::kTwoPi;

/// Positive abscissae (last is 0) of the QUADPACK 7-15 Gauss-Kronrod rule.
inline constexpr double kTvnGkNodes[8] = {
    0.99145537112081264, 0.94910791234275852, 0.86486442335976907, 0.74153118559939446,
    0.58608723546769113, 0.40584515137739717, 0.20778495500789847, 0.0,
};

/// Kronrod weights (15-point member) matching kTvnGkNodes.
inline constexpr double kTvnGkKronrodWeights[8] = {
    0.022935322010529223, 0.063092092629978553, 0.10479001032225018, 0.14065325971552592,
    0.16900472663926790,  0.19035057806478541,  0.20443294007529889, 0.20948214108472783,
};

/// Gauss 7-point weights, applied at kTvnGkNodes[1], [3], [5] and [7].
inline constexpr double kTvnGkGaussWeights[4] = {
    0.12948496616886969,
    0.27970539148927667,
    0.38183005050511894,
    0.41795918367346939,
};

/// Panel budget and embedded-error target of the path quadrature.
inline constexpr int kTvnMaxPanels = 128;
inline constexpr double kTvnQuadTolerance = 1e-15;

/// |h1|+|h2|+|h3| below this uses the exact orthant form (the closed form has
/// O(h) error above it, so the threshold is kept far below the 1e-12 budget).
inline constexpr double kTvnOrthantEps = 1e-14;

/// Invalid-correlation tolerance: determinants below this are rejected as NaN.
inline constexpr double kTvnDetEps = 1e-12;

/// sin(x) and cos(x)^2 with a series for cos^2 near |x| = pi/2, where the
/// direct 1 - sin^2 loses all significance (the published `sincs` trick).
inline void tvnSinCosSq(double x, double& sinX, double& cosSqX) {
    const double edge = 0.5 * kPi - std::abs(x);
    const double edgeSq = edge * edge;
    if (edgeSq < 5e-5) {
        const double sign = x < 0.0 ? -1.0 : 1.0;
        sinX = sign * (1.0 - edgeSq * (1.0 - edgeSq / 12.0) * 0.5);
        cosSqX = edgeSq * (1.0 - edgeSq * (1.0 - 2.0 * edgeSq / 15.0) / 3.0);
    } else {
        sinX = std::sin(x);
        cosSqX = 1.0 - sinX * sinX;
    }
}

/// Genz's trivariate conditional kernel (the `pntgnd` of the published
/// method) with the published guards: dt <= 0, bt <= -10 or ft >= 100
/// contribute nothing to double precision.
inline double tvnPntgnd(double ba, double bb, double bc, double ra, double rb, double r,
                        double rrd) {
    const double dt = rrd * (rrd - (ra - rb) * (ra - rb) - 2.0 * ra * rb * (1.0 - r));
    if (dt <= 0.0) {
        return 0.0;
    }
    const double bt = (bc * rrd + ba * (r * rb - ra) + bb * (r * ra - rb)) / std::sqrt(dt);
    if (bt <= -10.0) {
        return 0.0;
    }
    const double t = ba - r * bb;
    const double ft = t * t / rrd + bb * bb;
    if (ft >= 100.0) {
        return 0.0;
    }
    double value = std::exp(-0.5 * ft);
    if (bt < 10.0) {
        value *= normalCdfDouble(bt);
    }
    return value;
}

/// Integrand of the path representation before the 1/(2 pi) factor.
inline double tvnPathIntegrand(double x, double rua, double rub, double h1, double h2, double h3,
                               double r23) {
    double r12 = 0.0;
    double rr2 = 0.0;
    double r13 = 0.0;
    double rr3 = 0.0;
    tvnSinCosSq(rua * x, r12, rr2);
    tvnSinCosSq(rub * x, r13, rr3);
    double value = 0.0;
    if (rua != 0.0) {
        value += rua * tvnPntgnd(h1, h2, h3, r13, r23, r12, rr2);
    }
    if (rub != 0.0) {
        value += rub * tvnPntgnd(h1, h3, h2, r12, r23, r13, rr3);
    }
    return value;
}

/// One GK15 panel of the path integrand; `error` receives the embedded
/// |Kronrod - Gauss| estimate.
inline double tvnGk15Panel(double a, double b, double rua, double rub, double h1, double h2,
                           double h3, double r23, double& error) {
    const double half = 0.5 * (b - a);
    const double center = 0.5 * (b + a);
    const double fc = tvnPathIntegrand(center, rua, rub, h1, h2, h3, r23);
    double kronrod = kTvnGkKronrodWeights[7] * fc;
    double gauss = kTvnGkGaussWeights[3] * fc;
    for (int j = 0; j < 7; ++j) {
        const double t = half * kTvnGkNodes[j];
        const double fj = tvnPathIntegrand(center - t, rua, rub, h1, h2, h3, r23) +
                          tvnPathIntegrand(center + t, rua, rub, h1, h2, h3, r23);
        kronrod += kTvnGkKronrodWeights[j] * fj;
        if ((j & 1) != 0) {
            gauss += kTvnGkGaussWeights[(j - 1) / 2] * fj;
        }
    }
    error = std::abs(half * (kronrod - gauss));
    return half * kronrod;
}

/// Deterministic composite GK15 path integral on [0,1]: bisect the panel with
/// the largest embedded error until the summed estimate is at/below the target
/// or the panel budget is exhausted.  Fixed-size local arrays only.
inline double tvnPathCorrection(double rua, double rub, double h1, double h2, double h3,
                                double r23) {
    double lo[kTvnMaxPanels];
    double hi[kTvnMaxPanels];
    double value[kTvnMaxPanels];
    double err[kTvnMaxPanels];
    lo[0] = 0.0;
    hi[0] = 1.0;
    value[0] = tvnGk15Panel(0.0, 1.0, rua, rub, h1, h2, h3, r23, err[0]);
    int panels = 1;
    double total = value[0];
    double totalError = err[0];
    while (totalError > kTvnQuadTolerance && panels < kTvnMaxPanels) {
        int worst = 0;
        for (int i = 1; i < panels; ++i) {
            if (err[i] > err[worst]) {
                worst = i;
            }
        }
        const double a = lo[worst];
        const double b = hi[worst];
        const double mid = 0.5 * (a + b);
        double leftError = 0.0;
        double rightError = 0.0;
        const double left = tvnGk15Panel(a, mid, rua, rub, h1, h2, h3, r23, leftError);
        const double right = tvnGk15Panel(mid, b, rua, rub, h1, h2, h3, r23, rightError);
        lo[worst] = a;
        hi[worst] = mid;
        value[worst] = left;
        err[worst] = leftError;
        lo[panels] = mid;
        hi[panels] = b;
        value[panels] = right;
        err[panels] = rightError;
        ++panels;
        total = 0.0;
        totalError = 0.0;
        for (int i = 0; i < panels; ++i) {
            total += value[i];
            totalError += err[i];
        }
    }
    return total;
}

/// Standard trivariate normal CDF.  Scalar `double` kernel shared by the
/// public type and the derivative packs.
inline double tvnCdf(double h1, double h2, double h3, double r12, double r13, double r23) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    if (std::isnan(h1) || std::isnan(h2) || std::isnan(h3) || std::isnan(r12) || std::isnan(r13) ||
        std::isnan(r23)) {
        return nan;
    }
    if (std::abs(r12) > 1.0 || std::abs(r13) > 1.0 || std::abs(r23) > 1.0) {
        return nan;
    }
    // Exact degenerate limits at infinite arguments.
    if (h1 == -inf || h2 == -inf || h3 == -inf) {
        return 0.0;
    }
    if (h1 == inf) {
        if (h2 == inf) {
            return h3 == inf ? 1.0 : normalCdfDouble(h3);
        }
        return h3 == inf ? normalCdfDouble(h2) : bvnCdf(h2, h3, r23);
    }
    if (h2 == inf) {
        return h3 == inf ? normalCdfDouble(h1) : bvnCdf(h1, h3, r13);
    }
    if (h3 == inf) {
        return bvnCdf(h1, h2, r12);
    }
    const double det = 1.0 - r12 * r12 - r13 * r13 - r23 * r23 + 2.0 * r12 * r13 * r23;
    if (det < -kTvnDetEps) {
        return nan;
    }
    if (std::abs(h1) + std::abs(h2) + std::abs(h3) < kTvnOrthantEps) {
        return 0.125 + (std::asin(r12) + std::asin(r13) + std::asin(r23)) / (4.0 * kPi.hi);
    }
    // Published partial sort: r23 receives the largest |r|, minimizing the
    // range the integration path has to traverse.
    double hs[3] = {h1, h2, h3};
    double rs[3] = {r12, r13, r23};
    if (std::abs(rs[0]) > std::abs(rs[1])) {
        const double th = hs[1];
        hs[1] = hs[2];
        hs[2] = th;
        const double tr = rs[0];
        rs[0] = rs[1];
        rs[1] = tr;
    }
    if (std::abs(rs[1]) > std::abs(rs[2])) {
        const double th = hs[0];
        hs[0] = hs[1];
        hs[1] = th;
        const double tr = rs[1];
        rs[1] = rs[2];
        rs[2] = tr;
    }
    const double eps = std::numeric_limits<double>::epsilon();
    if (std::abs(rs[0]) + std::abs(rs[1]) < eps) {
        return normalCdfDouble(hs[0]) * bvnCdf(hs[1], hs[2], rs[2]);
    }
    if (1.0 - rs[2] < eps) {
        return bvnCdf(hs[0], hs[1] < hs[2] ? hs[1] : hs[2], rs[0]);
    }
    if (rs[2] + 1.0 < eps) {
        if (hs[1] > -hs[2]) {
            const double value = bvnCdf(hs[0], hs[1], rs[0]) - bvnCdf(hs[0], -hs[2], rs[0]);
            return value > 0.0 ? value : 0.0;
        }
        return 0.0;
    }
    const double rua = std::asin(rs[0]);
    const double rub = std::asin(rs[1]);
    double value = normalCdfDouble(hs[0]) * bvnCdf(hs[1], hs[2], rs[2]) +
                   tvnPathCorrection(rua, rub, hs[0], hs[1], hs[2], rs[2]) / kTwoPi.hi;
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

/// Value plus analytic gradient w.r.t. (h1,h2,h3,r12,r13,r23).
struct TvnCdfGrad {
    double value;
    double g[6];
};

/// Value, gradient and full symmetric Hessian.  Parameter order is
/// (h1,h2,h3,r12,r13,r23); `h` stores the lower triangle row-major with
/// index (i,j), i >= j, at i*(i+1)/2 + j.
struct TvnGradHess {
    double value;
    double g[6];
    double h[21];
};

/// Lower-triangle row-major index of the Hessian entry (i,j).
inline constexpr int tvnHessIndex(int i, int j) {
    return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
}

/// g_i = d Phi3/dh_i and its six partial derivatives (the Hessian rows for the
/// h parameters).  Parameter arrays are ordered (h1,h2,h3) and
/// (r12,r13,r23); pair indices are stepped as in h_grad(i,j,k).
inline void tvnHGrad(int i, int j, int k, int pij, int pik, int pjk, const double h[3],
                     const double r[3], double& g, double dg[6]) {
    const double rij = r[pij - 3];
    const double rik = r[pik - 3];
    const double rjk = r[pjk - 3];
    const double sij = std::sqrt(1.0 - rij * rij);
    const double sik = std::sqrt(1.0 - rik * rik);
    const double a = (h[j] - rij * h[i]) / sij;
    const double b = (h[k] - rik * h[i]) / sik;
    const double rho = (rjk - rij * rik) / (sij * sik);
    const double srho = std::sqrt(1.0 - rho * rho);
    const double ca = (b - rho * a) / srho;
    const double cb = (a - rho * b) / srho;
    const double phi2 = bvnCdf(a, b, rho);
    g = normalPdfDouble(h[i]) * phi2;

    double da[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double db[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double drho[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    da[i] = -rij / sij;
    da[j] = 1.0 / sij;
    da[pij] = (rij * h[j] - h[i]) / (sij * sij * sij);
    db[i] = -rik / sik;
    db[k] = 1.0 / sik;
    db[pik] = (rik * h[k] - h[i]) / (sik * sik * sik);
    drho[pij] = (rij * rjk - rik) / (sij * sij * sij * sik);
    drho[pik] = (rik * rjk - rij) / (sij * sik * sik * sik);
    drho[pjk] = 1.0 / (sij * sik);
    const double pa = normalPdfDouble(a) * normalCdfDouble(ca);
    const double pb = normalPdfDouble(b) * normalCdfDouble(cb);
    const double prho = bvnPdf(a, b, rho);
    for (int p = 0; p < 6; ++p) {
        dg[p] = normalPdfDouble(h[i]) * (pa * da[p] + pb * db[p] + prho * drho[p]);
    }
    dg[i] += -h[i] * g;
}

/// g = d Phi3/dr_ij (pair (i,j), third k) and its six partial derivatives.
inline void tvnRGrad(int i, int j, int k, int pij, int pik, int pjk, const double h[3],
                     const double r[3], double& g, double dg[6]) {
    const double rij = r[pij - 3];
    const double rik = r[pik - 3];
    const double rjk = r[pjk - 3];
    const double d = 1.0 - rij * rij;
    const double alpha = (rik - rij * rjk) / d;
    const double beta = (rjk - rij * rik) / d;
    const double det = 1.0 - rij * rij - rik * rik - rjk * rjk + 2.0 * rij * rik * rjk;
    const double s = std::sqrt(det / d);
    const double z = (h[k] - alpha * h[i] - beta * h[j]) / s;
    const double phiz = normalCdfDouble(z);
    const double f2 = bvnPdf(h[i], h[j], rij);
    g = f2 * phiz;

    double dalpha[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dbeta[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double ds[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    dalpha[pij] = (-rjk * d + 2.0 * rij * (rik - rij * rjk)) / (d * d);
    dalpha[pik] = 1.0 / d;
    dalpha[pjk] = -rij / d;
    dbeta[pij] = (-rik * d + 2.0 * rij * (rjk - rij * rik)) / (d * d);
    dbeta[pik] = -rij / d;
    dbeta[pjk] = 1.0 / d;
    const double ddetij = -2.0 * rij + 2.0 * rik * rjk;
    const double ddetik = -2.0 * rik + 2.0 * rij * rjk;
    const double ddetjk = -2.0 * rjk + 2.0 * rij * rik;
    ds[pij] = (ddetij * d - det * (-2.0 * rij)) / (2.0 * s * d * d);
    ds[pik] = ddetik / (2.0 * s * d);
    ds[pjk] = ddetjk / (2.0 * s * d);
    double dz[6];
    for (int p = 0; p < 6; ++p) {
        dz[p] = -(dalpha[p] * h[i] + dbeta[p] * h[j] + z * ds[p]) / s;
    }
    dz[i] += -alpha / s;
    dz[j] += -beta / s;
    dz[k] += 1.0 / s;
    const double df2i = -f2 * (h[i] - rij * h[j]) / d;
    const double df2j = -f2 * (h[j] - rij * h[i]) / d;
    const double df2r =
        f2 *
        (rij / d + (h[i] * h[j] * (1.0 + rij * rij) - rij * (h[i] * h[i] + h[j] * h[j])) / (d * d));
    for (int p = 0; p < 6; ++p) {
        dg[p] = f2 * normalPdfDouble(z) * dz[p];
    }
    dg[i] += df2i * phiz;
    dg[j] += df2j * phiz;
    dg[pij] += df2r * phiz;
}

/// Value, gradient and Hessian of Phi3.  Requires finite h, |r_ij| < 1 and a
/// positive-definite correlation triple (det > 0); otherwise every entry is
/// NaN.  The Hessian is the exact Jacobian of the closed-form gradient pack.
inline TvnGradHess tvnCdfGradHess(double h1, double h2, double h3, double r12, double r13,
                                  double r23) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    TvnGradHess out{};
    const double det = 1.0 - r12 * r12 - r13 * r13 - r23 * r23 + 2.0 * r12 * r13 * r23;
    if (!std::isfinite(h1) || !std::isfinite(h2) || !std::isfinite(h3) || !std::isfinite(r12) ||
        !std::isfinite(r13) || !std::isfinite(r23) || std::abs(r12) >= 1.0 ||
        std::abs(r13) >= 1.0 || std::abs(r23) >= 1.0 || !(det > 0.0)) {
        out.value = nan;
        for (int p = 0; p < 6; ++p) {
            out.g[p] = nan;
        }
        for (int p = 0; p < 21; ++p) {
            out.h[p] = nan;
        }
        return out;
    }
    const double h[3] = {h1, h2, h3};
    const double r[3] = {r12, r13, r23};
    out.value = tvnCdf(h1, h2, h3, r12, r13, r23);
    double dg[6];
    // Rows conditioned on h1, h2, h3: pair (i,j) with third k and the pair
    // parameter slots pij = r_ij, pik = r_ik, pjk = r_jk.
    tvnHGrad(0, 1, 2, 3, 4, 5, h, r, out.g[0], dg);
    for (int p = 0; p < 6; ++p) {
        out.h[tvnHessIndex(0, p)] = dg[p];
    }
    tvnHGrad(1, 0, 2, 3, 5, 4, h, r, out.g[1], dg);
    for (int p = 1; p < 6; ++p) {
        out.h[tvnHessIndex(1, p)] = dg[p];
    }
    tvnHGrad(2, 0, 1, 4, 5, 3, h, r, out.g[2], dg);
    for (int p = 2; p < 6; ++p) {
        out.h[tvnHessIndex(2, p)] = dg[p];
    }
    // Pair rows for r12, r13, r23 (third variable index 2, 1, 0 respectively).
    tvnRGrad(0, 1, 2, 3, 4, 5, h, r, out.g[3], dg);
    for (int p = 3; p < 6; ++p) {
        out.h[tvnHessIndex(3, p)] = dg[p];
    }
    tvnRGrad(0, 2, 1, 4, 3, 5, h, r, out.g[4], dg);
    for (int p = 4; p < 6; ++p) {
        out.h[tvnHessIndex(4, p)] = dg[p];
    }
    tvnRGrad(1, 2, 0, 5, 3, 4, h, r, out.g[5], dg);
    out.h[tvnHessIndex(5, 5)] = dg[5];
    return out;
}

/// Value plus analytic gradient of Phi3.  Requires finite h, |r_ij| < 1 and a
/// positive-definite correlation triple; otherwise all outputs are NaN.
inline TvnCdfGrad tvnCdfGrad(double h1, double h2, double h3, double r12, double r13, double r23) {
    const TvnGradHess pack = tvnCdfGradHess(h1, h2, h3, r12, r13, r23);
    TvnCdfGrad out{};
    out.value = pack.value;
    for (int p = 0; p < 6; ++p) {
        out.g[p] = pack.g[p];
    }
    return out;
}

} // namespace quantape::math::detail

#endif // QUANTAPE_MATH_DISTRIBUTIONS_DETAIL_TRIVARIATE_NORMAL_KERNEL_H
