/**
 * @file test_black76.cpp
 * @brief Validates Black-76 and GBS analytical Greeks against FD and Stan AD
 */

// #include <Eigen/Dense>

#include "quantape/math/StanMath.h"

#include "quantape/log/Log.h"
#include "quantape/pricing/StanPrimitives.h"
#include "quantape/util/Check.h"

#include <cmath>
#include <iomanip>
#include <iostream>

using quantape::pricing::Black76;
using quantape::pricing::black76Analytical;
using quantape::pricing::Black76Result;
using quantape::pricing::GBS;
using quantape::pricing::OptionType;
using stan::math::fvar;
using stan::math::var;

// ============================================================================
// Black-76 tests
// ============================================================================

void checkB76Greeks1_FD(double DF, double F, double K, double vol, double T, OptionType type) {
    auto res = black76Analytical(DF, F, K, vol, T, type);
    const double eps = 1e-6;

    auto p = [&](double DF_, double F_, double K_, double vol_) {
        return black76Analytical(DF_, F_, K_, vol_, T, type).price;
    };

    double fd_dD = (p(DF + eps, F, K, vol) - p(DF - eps, F, K, vol)) / (2 * eps);
    double fd_dF = (p(DF, F + eps, K, vol) - p(DF, F - eps, K, vol)) / (2 * eps);
    double fd_dK = (p(DF, F, K + eps, vol) - p(DF, F, K - eps, vol)) / (2 * eps);
    double fd_ds = (p(DF, F, K, vol + eps) - p(DF, F, K, vol - eps)) / (2 * eps);

    QTA_LOG_INFO("test", "  1st-order Greeks vs FD:");
    auto row = [](const char* name, double anal, double fd) {
        QTA_LOG_INFO("test", "    {}  anal={}  FD={}  err={}", name, quantape::format::num(anal),
                     quantape::format::num(fd), quantape::format::num(std::abs(anal - fd), 2));
    };
    row("dV/dD", res.g1.dV_dDF, fd_dD);
    row("dV/dF", res.g1.dV_dF, fd_dF);
    row("dV/dK", res.g1.dV_dK, fd_dK);
    row("dV/dsigma", res.g1.dV_dvol, fd_ds);
}

void checkB76Greeks2_FD(double DF, double F, double K, double vol, double T, OptionType type) {
    auto res = black76Analytical(DF, F, K, vol, T, type);
    const double eps = 1e-5;

    auto g1 = [&](double DF_, double F_, double K_, double vol_) {
        return black76Analytical(DF_, F_, K_, vol_, T, type).g1;
    };

    auto g1_Dp = g1(DF + eps, F, K, vol), g1_Dm = g1(DF - eps, F, K, vol);
    auto g1_Fp = g1(DF, F + eps, K, vol), g1_Fm = g1(DF, F - eps, K, vol);
    auto g1_Kp = g1(DF, F, K + eps, vol), g1_Km = g1(DF, F, K - eps, vol);
    auto g1_sp = g1(DF, F, K, vol + eps), g1_sm = g1(DF, F, K, vol - eps);

    QTA_LOG_INFO("test", "  2nd-order Greeks vs FD:");
    auto row = [](const char* name, double anal, double fd) {
        QTA_LOG_INFO("test", "    {}  anal={}  FD={}  err={}", name, quantape::util::num(anal),
                     quantape::util::num(fd), quantape::util::num(std::abs(anal - fd), 2));
    };
    row("d2V/dD_dF", res.g2.d2V_dDF_dF, (g1_Dp.dV_dF - g1_Dm.dV_dF) / (2 * eps));
    row("d2V/dD_dK", res.g2.d2V_dDF_dK, (g1_Dp.dV_dK - g1_Dm.dV_dK) / (2 * eps));
    row("d2V/dD_ds", res.g2.d2V_dDF_dvol, (g1_Dp.dV_dvol - g1_Dm.dV_dvol) / (2 * eps));
    row("d2V/dF2", res.g2.d2V_dF2, (g1_Fp.dV_dF - g1_Fm.dV_dF) / (2 * eps));
    row("d2V/dF_dK", res.g2.d2V_dF_dK, (g1_Fp.dV_dK - g1_Fm.dV_dK) / (2 * eps));
    row("d2V/dF_ds", res.g2.d2V_dF_dvol, (g1_Fp.dV_dvol - g1_Fm.dV_dvol) / (2 * eps));
    row("d2V/dK2", res.g2.d2V_dK2, (g1_Kp.dV_dK - g1_Km.dV_dK) / (2 * eps));
    row("d2V/dK_ds", res.g2.d2V_dK_dvol, (g1_Kp.dV_dvol - g1_Km.dV_dvol) / (2 * eps));
    row("d2V/ds2", res.g2.d2V_dvol2, (g1_sp.dV_dvol - g1_sm.dV_dvol) / (2 * eps));
}

void checkB76StanVar(double DF0, double F0, double K0, double vol0, double T, OptionType type) {
    var DF(DF0), F(F0), K(K0), vol(vol0);
    var price = Black76<var>{DF, F, K, vol, T, type}.price();
    stan::math::grad(price.vi_);

    auto res = black76Analytical(DF0, F0, K0, vol0, T, type);

    QTA_LOG_INFO("test", "  Stan var vs analytical:");
    auto row = [](const char* name, double ad, double anal) {
        QTA_LOG_INFO("test", "    {}  AD={}  anal={}  err={}", name, quantape::util::num(ad),
                     quantape::util::num(anal), quantape::util::num(std::abs(ad - anal), 2));
    };
    row("price", price.val(), res.price);
    row("dV/dDF", DF.adj(), res.g1.dV_dDF);
    row("dV/dF", F.adj(), res.g1.dV_dF);
    row("dV/dK", K.adj(), res.g1.dV_dK);
    row("dV/dvol", vol.adj(), res.g1.dV_dvol);
    stan::math::recover_memory();
}

struct B76Functor {
    double T;
    OptionType type;
    template <typename TT>
    TT operator()(const Eigen::Matrix<TT, Eigen::Dynamic, 1>& theta) const {
        return Black76<TT>{theta(0), theta(1), theta(2), theta(3), T, type}.price();
    }
};

void checkB76StanHessian(double DF0, double F0, double K0, double vol0, double T, OptionType type) {
    Eigen::VectorXd x(4);
    x << DF0, F0, K0, vol0;

    double fx;
    Eigen::VectorXd grad(4);
    Eigen::MatrixXd H(4, 4);
    stan::math::hessian(B76Functor{T, type}, x, fx, grad, H);

    auto res = black76Analytical(DF0, F0, K0, vol0, T, type);

    Eigen::Matrix4d H_anal = Eigen::Matrix4d::Zero();
    H_anal(0, 1) = H_anal(1, 0) = res.g2.d2V_dDF_dF;
    H_anal(0, 2) = H_anal(2, 0) = res.g2.d2V_dDF_dK;
    H_anal(0, 3) = H_anal(3, 0) = res.g2.d2V_dDF_dvol;
    H_anal(1, 1) = res.g2.d2V_dF2;
    H_anal(1, 2) = H_anal(2, 1) = res.g2.d2V_dF_dK;
    H_anal(1, 3) = H_anal(3, 1) = res.g2.d2V_dF_dvol;
    H_anal(2, 2) = res.g2.d2V_dK2;
    H_anal(2, 3) = H_anal(3, 2) = res.g2.d2V_dK_dvol;
    H_anal(3, 3) = res.g2.d2V_dvol2;

    double hess_err = (H - Eigen::MatrixXd(H_anal)).cwiseAbs().maxCoeff();
    double grad_err = 0.0;
    double anal_grad[] = {res.g1.dV_dDF, res.g1.dV_dF, res.g1.dV_dK, res.g1.dV_dvol};
    for (int i = 0; i < 4; ++i)
        grad_err = std::max(grad_err, std::abs(grad(i) - anal_grad[i]));

    QTA_LOG_INFO("test", "  Stan hessian: price_err={}  grad_err={}  hess_err={}",
                 quantape::util::num(std::abs(fx - res.price), 2), quantape::util::num(grad_err, 2),
                 quantape::util::num(hess_err, 2));
}

// ============================================================================
// GBS tests
// ============================================================================

void checkGBSEquivalence(double S, double K, double r_disc, double b, double vol, double T,
                         OptionType type) {
    double F = S * std::exp(b * T);
    double DF = std::exp(-r_disc * T);
    double b76_price = Black76<double>{DF, F, K, vol, T, type}.price();
    double gbs_price = GBS<double>{S, K, r_disc, b, vol, T, type}.price();
    QTA_LOG_INFO("test", "  GBS vs B76: b76={}  gbs={}  err={}", quantape::util::num(b76_price),
                 quantape::util::num(gbs_price),
                 quantape::util::num(std::abs(b76_price - gbs_price), 2));
}

void checkGBSStanVar(double S0, double K0, double r0, double b0, double vol0, double T,
                     OptionType type) {
    var S(S0), K(K0), r_disc(r0), b(b0), vol(vol0);
    var price = GBS<var>{S, K, r_disc, b, vol, T, type}.price();
    stan::math::grad(price.vi_);

    double ad_dS = S.adj(), ad_dK = K.adj(), ad_dr = r_disc.adj(), ad_db = b.adj(),
           ad_dvol = vol.adj();
    stan::math::recover_memory();

    const double eps = 1e-7;
    auto p = [&](double S_, double K_, double r_, double b_, double vol_) {
        return GBS<double>{S_, K_, r_, b_, vol_, T, type}.price();
    };
    double fd_dS = (p(S0 + eps, K0, r0, b0, vol0) - p(S0 - eps, K0, r0, b0, vol0)) / (2 * eps);
    double fd_dK = (p(S0, K0 + eps, r0, b0, vol0) - p(S0, K0 - eps, r0, b0, vol0)) / (2 * eps);
    double fd_dr = (p(S0, K0, r0 + eps, b0, vol0) - p(S0, K0, r0 - eps, b0, vol0)) / (2 * eps);
    double fd_db = (p(S0, K0, r0, b0 + eps, vol0) - p(S0, K0, r0, b0 - eps, vol0)) / (2 * eps);
    double fd_dvol = (p(S0, K0, r0, b0, vol0 + eps) - p(S0, K0, r0, b0, vol0 - eps)) / (2 * eps);

    QTA_LOG_INFO("test", "  GBS Stan var vs FD:");
    auto row = [](const char* name, double ad, double fd) {
        QTA_LOG_INFO("test", "    {}  AD={}  FD={}  err={}", name, quantape::util::num(ad),
                     quantape::util::num(fd), quantape::util::num(std::abs(ad - fd), 2));
    };
    row("dV/dS", ad_dS, fd_dS);
    row("dV/dK", ad_dK, fd_dK);
    row("dV/dr_disc", ad_dr, fd_dr);
    row("dV/db", ad_db, fd_db);
    row("dV/dvol", ad_dvol, fd_dvol);
}

struct GBSFunctor {
    double T;
    OptionType type;
    template <typename TT>
    TT operator()(const Eigen::Matrix<TT, Eigen::Dynamic, 1>& theta) const {
        return GBS<TT>{theta(0), theta(1), theta(2), theta(3), theta(4), T, type}.price();
    }
};

void checkGBSStanHessian(double S0, double K0, double r0, double b0, double vol0, double T,
                         OptionType type) {
    Eigen::VectorXd x(5);
    x << S0, K0, r0, b0, vol0;

    double fx;
    Eigen::VectorXd grad(5);
    Eigen::MatrixXd H(5, 5);
    stan::math::hessian(GBSFunctor{T, type}, x, fx, grad, H);

    // Verify gradient against FD
    const double eps = 1e-7;
    auto p = [&](double S_, double K_, double r_, double b_, double vol_) {
        return GBS<double>{S_, K_, r_, b_, vol_, T, type}.price();
    };
    double fd_grad[] = {
        (p(S0 + eps, K0, r0, b0, vol0) - p(S0 - eps, K0, r0, b0, vol0)) / (2 * eps),
        (p(S0, K0 + eps, r0, b0, vol0) - p(S0, K0 - eps, r0, b0, vol0)) / (2 * eps),
        (p(S0, K0, r0 + eps, b0, vol0) - p(S0, K0, r0 - eps, b0, vol0)) / (2 * eps),
        (p(S0, K0, r0, b0 + eps, vol0) - p(S0, K0, r0, b0 - eps, vol0)) / (2 * eps),
        (p(S0, K0, r0, b0, vol0 + eps) - p(S0, K0, r0, b0, vol0 - eps)) / (2 * eps),
    };

    double grad_err = 0.0;
    for (int i = 0; i < 5; ++i)
        grad_err = std::max(grad_err, std::abs(grad(i) - fd_grad[i]));

    // Verify Hessian against FD of gradient
    Eigen::MatrixXd H_fd(5, 5);
    for (int i = 0; i < 5; ++i) {
        Eigen::VectorXd xp = x, xm = x;
        xp(i) += eps;
        xm(i) -= eps;
        double fxp, fxm;
        Eigen::VectorXd gp(5), gm(5);
        Eigen::MatrixXd Hp(5, 5), Hm(5, 5);
        stan::math::hessian(GBSFunctor{T, type}, xp, fxp, gp, Hp);
        stan::math::hessian(GBSFunctor{T, type}, xm, fxm, gm, Hm);
        for (int j = 0; j < 5; ++j)
            H_fd(i, j) = (gp(j) - gm(j)) / (2 * eps);
    }

    double hess_err = (H - H_fd).cwiseAbs().maxCoeff();

    QTA_LOG_INFO("test", "  GBS hessian: price_err={}  grad_err={}  hess_err={}",
                 quantape::util::num(std::abs(fx - p(S0, K0, r0, b0, vol0)), 2),
                 quantape::util::num(grad_err, 2), quantape::util::num(hess_err, 2));

    const char* names[] = {"S", "K", "r_disc", "b", "vol"};
    QTA_LOG_INFO("test", "    Hessian (AD):");
    for (int i = 0; i < 5; ++i) {
        QTA_LOG_INFO("test", "{}  {}", quantape::format::matrix(H.row(i), {.precision = 6}),
                     names[i]);
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    QTA_LOG_INFO("test", "{}", std::string(60, '='));
    QTA_LOG_INFO("test", "  Black-76 Tests");
    QTA_LOG_INFO("test", "{}", std::string(60, '='));

    struct B76Case {
        double DF, F, K, vol, T;
        OptionType type;
        const char* label;
    };

    B76Case b76_cases[] = {
        {0.95, 100.0, 100.0, 0.20, 1.0, OptionType::Call, "ATM Call"},
        {0.95, 100.0, 100.0, 0.20, 1.0, OptionType::Put, "ATM Put"},
        {0.90, 100.0, 120.0, 0.25, 2.0, OptionType::Call, "OTM Call"},
        {0.90, 100.0, 80.0, 0.25, 2.0, OptionType::Put, "OTM Put"},
        {0.98, 100.0, 80.0, 0.15, 0.5, OptionType::Call, "ITM Call"},
        {0.98, 100.0, 120.0, 0.15, 0.5, OptionType::Put, "ITM Put"},
        {0.99, 100.0, 100.0, 0.40, 5.0, OptionType::Call, "High vol long T"},
    };

    for (const auto& tc : b76_cases) {
        QTA_LOG_INFO("test", "-- {} --", tc.label);
        checkB76Greeks1_FD(tc.DF, tc.F, tc.K, tc.vol, tc.T, tc.type);
        checkB76Greeks2_FD(tc.DF, tc.F, tc.K, tc.vol, tc.T, tc.type);
        checkB76StanVar(tc.DF, tc.F, tc.K, tc.vol, tc.T, tc.type);
        checkB76StanHessian(tc.DF, tc.F, tc.K, tc.vol, tc.T, tc.type);
    }

    QTA_LOG_INFO("test", "{}", std::string(60, '='));
    QTA_LOG_INFO("test", "  GBS Tests");
    QTA_LOG_INFO("test", "{}", std::string(60, '='));

    struct GBSCase {
        double S, K, r_disc, b, vol, T;
        OptionType type;
        const char* label;
    };

    // b = r_fund - q.  When r_disc = r_fund and q = div yield, b = r_disc - q (classic GBS).
    // When r_disc != r_fund (OIS discounting), they decouple.
    GBSCase gbs_cases[] = {
        // Classic: r_disc = r_fund, b = r_fund - q
        {100.0, 100.0, 0.05, 0.03, 0.20, 1.0, OptionType::Call, "ATM Call, b=r-q"},
        {100.0, 100.0, 0.05, 0.03, 0.20, 1.0, OptionType::Put, "ATM Put, b=r-q"},
        // OIS discounting: r_disc (OIS) != b (LIBOR - q)
        {100.0, 100.0, 0.02, 0.05, 0.20, 1.0, OptionType::Call, "OIS disc, r_disc < b"},
        {100.0, 120.0, 0.01, 0.04, 0.25, 2.0, OptionType::Call, "OIS OTM Call"},
        // Futures-style: b = 0 (forward = spot)
        {100.0, 90.0, 0.03, 0.00, 0.30, 0.5, OptionType::Put, "Futures put, b=0"},
        // Negative carry (q > r_fund)
        {100.0, 100.0, 0.03, -0.02, 0.15, 3.0, OptionType::Call, "Neg carry b<0"},
    };

    for (const auto& tc : gbs_cases) {
        QTA_LOG_INFO("test", "-- {} (S={} K={} r_disc={} b={} vol={} T={}) --", tc.label,
                     quantape::util::num(tc.S), quantape::util::num(tc.K),
                     quantape::util::num(tc.r_disc), quantape::util::num(tc.b),
                     quantape::util::num(tc.vol), quantape::util::num(tc.T));
        checkGBSEquivalence(tc.S, tc.K, tc.r_disc, tc.b, tc.vol, tc.T, tc.type);
        checkGBSStanVar(tc.S, tc.K, tc.r_disc, tc.b, tc.vol, tc.T, tc.type);
        checkGBSStanHessian(tc.S, tc.K, tc.r_disc, tc.b, tc.vol, tc.T, tc.type);
    }

    // util::checkClose AD-scalar overload: checks value and adjoint together.
    {
        var x = 2.0;
        var y = 2.0;
        x.vi_->adj_ = 0.5;
        y.vi_->adj_ = 0.5;
        quantape::util::checkClose("util var value+adj", x, y, 0.0);
    }

    return 0;
}
