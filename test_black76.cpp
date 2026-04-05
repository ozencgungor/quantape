/**
 * @file test_black76.cpp
 * @brief Validates Black-76 and GBS analytical Greeks against FD and Stan AD
 */

#include <Eigen/Dense>
#include <stan/math.hpp>
#include <stan/math/mix.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "Pricing/StanPrimitives.h"

using Pricing::Black76;
using Pricing::black76Analytical;
using Pricing::Black76Result;
using Pricing::GBS;
using Pricing::OptionType;
using stan::math::fvar;
using stan::math::var;

// ============================================================================
// Black-76 tests
// ============================================================================

void checkB76Greeks1_FD(double D, double F, double K, double sigma, double T, OptionType type) {
    auto res = black76Analytical(D, F, K, sigma, T, type);
    const double eps = 1e-6;

    auto p = [&](double D_, double F_, double K_, double s_) {
        return black76Analytical(D_, F_, K_, s_, T, type).price;
    };

    double fd_dD = (p(D + eps, F, K, sigma) - p(D - eps, F, K, sigma)) / (2 * eps);
    double fd_dF = (p(D, F + eps, K, sigma) - p(D, F - eps, K, sigma)) / (2 * eps);
    double fd_dK = (p(D, F, K + eps, sigma) - p(D, F, K - eps, sigma)) / (2 * eps);
    double fd_ds = (p(D, F, K, sigma + eps) - p(D, F, K, sigma - eps)) / (2 * eps);

    std::cout << std::setprecision(10) << std::fixed;
    std::cout << "  1st-order Greeks vs FD:\n";
    auto row = [](const char* name, double anal, double fd) {
        std::cout << "    " << std::setw(12) << name << "  anal=" << std::setw(16) << anal
                  << "  FD=" << std::setw(16) << fd << "  err=" << std::scientific
                  << std::abs(anal - fd) << std::fixed << "\n";
    };
    row("dV/dD", res.g1.dV_dD, fd_dD);
    row("dV/dF", res.g1.dV_dF, fd_dF);
    row("dV/dK", res.g1.dV_dK, fd_dK);
    row("dV/dsigma", res.g1.dV_dsigma, fd_ds);
}

void checkB76Greeks2_FD(double D, double F, double K, double sigma, double T, OptionType type) {
    auto res = black76Analytical(D, F, K, sigma, T, type);
    const double eps = 1e-5;

    auto g1 = [&](double D_, double F_, double K_, double s_) {
        return black76Analytical(D_, F_, K_, s_, T, type).g1;
    };

    auto g1_Dp = g1(D + eps, F, K, sigma), g1_Dm = g1(D - eps, F, K, sigma);
    auto g1_Fp = g1(D, F + eps, K, sigma), g1_Fm = g1(D, F - eps, K, sigma);
    auto g1_Kp = g1(D, F, K + eps, sigma), g1_Km = g1(D, F, K - eps, sigma);
    auto g1_sp = g1(D, F, K, sigma + eps), g1_sm = g1(D, F, K, sigma - eps);

    std::cout << "  2nd-order Greeks vs FD:\n";
    auto row = [](const char* name, double anal, double fd) {
        std::cout << "    " << std::setw(14) << name << "  anal=" << std::setw(16) << anal
                  << "  FD=" << std::setw(16) << fd << "  err=" << std::scientific
                  << std::abs(anal - fd) << std::fixed << "\n";
    };
    row("d2V/dD_dF", res.g2.d2V_dD_dF, (g1_Dp.dV_dF - g1_Dm.dV_dF) / (2 * eps));
    row("d2V/dD_dK", res.g2.d2V_dD_dK, (g1_Dp.dV_dK - g1_Dm.dV_dK) / (2 * eps));
    row("d2V/dD_ds", res.g2.d2V_dD_dsigma, (g1_Dp.dV_dsigma - g1_Dm.dV_dsigma) / (2 * eps));
    row("d2V/dF2", res.g2.d2V_dF2, (g1_Fp.dV_dF - g1_Fm.dV_dF) / (2 * eps));
    row("d2V/dF_dK", res.g2.d2V_dF_dK, (g1_Fp.dV_dK - g1_Fm.dV_dK) / (2 * eps));
    row("d2V/dF_ds", res.g2.d2V_dF_dsigma, (g1_Fp.dV_dsigma - g1_Fm.dV_dsigma) / (2 * eps));
    row("d2V/dK2", res.g2.d2V_dK2, (g1_Kp.dV_dK - g1_Km.dV_dK) / (2 * eps));
    row("d2V/dK_ds", res.g2.d2V_dK_dsigma, (g1_Kp.dV_dsigma - g1_Km.dV_dsigma) / (2 * eps));
    row("d2V/ds2", res.g2.d2V_dsigma2, (g1_sp.dV_dsigma - g1_sm.dV_dsigma) / (2 * eps));
}

void checkB76StanVar(double D0, double F0, double K0, double sigma0, double T, OptionType type) {
    var D(D0), F(F0), K(K0), sigma(sigma0);
    var price = Black76<var>{D, F, K, sigma, T, type}.price();
    stan::math::grad(price.vi_);

    auto res = black76Analytical(D0, F0, K0, sigma0, T, type);

    std::cout << "  Stan var vs analytical:\n";
    std::cout << std::setprecision(12) << std::fixed;
    auto row = [](const char* name, double ad, double anal) {
        std::cout << "    " << std::setw(12) << name << "  AD=" << std::setw(18) << ad
                  << "  anal=" << std::setw(18) << anal << "  err=" << std::scientific
                  << std::abs(ad - anal) << std::fixed << "\n";
    };
    row("price", price.val(), res.price);
    row("dV/dD", D.adj(), res.g1.dV_dD);
    row("dV/dF", F.adj(), res.g1.dV_dF);
    row("dV/dK", K.adj(), res.g1.dV_dK);
    row("dV/dsigma", sigma.adj(), res.g1.dV_dsigma);
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

void checkB76StanHessian(double D0, double F0, double K0, double sigma0, double T,
                         OptionType type) {
    Eigen::VectorXd x(4);
    x << D0, F0, K0, sigma0;

    double fx;
    Eigen::VectorXd grad(4);
    Eigen::MatrixXd H(4, 4);
    stan::math::hessian(B76Functor{T, type}, x, fx, grad, H);

    auto res = black76Analytical(D0, F0, K0, sigma0, T, type);

    Eigen::Matrix4d H_anal = Eigen::Matrix4d::Zero();
    H_anal(0, 1) = H_anal(1, 0) = res.g2.d2V_dD_dF;
    H_anal(0, 2) = H_anal(2, 0) = res.g2.d2V_dD_dK;
    H_anal(0, 3) = H_anal(3, 0) = res.g2.d2V_dD_dsigma;
    H_anal(1, 1) = res.g2.d2V_dF2;
    H_anal(1, 2) = H_anal(2, 1) = res.g2.d2V_dF_dK;
    H_anal(1, 3) = H_anal(3, 1) = res.g2.d2V_dF_dsigma;
    H_anal(2, 2) = res.g2.d2V_dK2;
    H_anal(2, 3) = H_anal(3, 2) = res.g2.d2V_dK_dsigma;
    H_anal(3, 3) = res.g2.d2V_dsigma2;

    double hess_err = (H - Eigen::MatrixXd(H_anal)).cwiseAbs().maxCoeff();
    double grad_err = 0.0;
    double anal_grad[] = {res.g1.dV_dD, res.g1.dV_dF, res.g1.dV_dK, res.g1.dV_dsigma};
    for (int i = 0; i < 4; ++i)
        grad_err = std::max(grad_err, std::abs(grad(i) - anal_grad[i]));

    std::cout << "  Stan hessian: price_err=" << std::scientific << std::abs(fx - res.price)
              << "  grad_err=" << grad_err << "  hess_err=" << hess_err << std::fixed << "\n";
}

// ============================================================================
// GBS tests
// ============================================================================

void checkGBSEquivalence(double S, double K, double r_disc, double b, double sigma, double T,
                         OptionType type) {
    double F = S * std::exp(b * T);
    double D = std::exp(-r_disc * T);
    double b76_price = Black76<double>{D, F, K, sigma, T, type}.price();
    double gbs_price = GBS<double>{S, K, r_disc, b, sigma, T, type}.price();
    std::cout << "  GBS vs B76: b76=" << std::setprecision(12) << b76_price << "  gbs=" << gbs_price
              << "  err=" << std::scientific << std::abs(b76_price - gbs_price) << std::fixed
              << "\n";
}

void checkGBSStanVar(double S0, double K0, double r0, double b0, double sigma0, double T,
                     OptionType type) {
    var S(S0), K(K0), r_disc(r0), b(b0), sigma(sigma0);
    var price = GBS<var>{S, K, r_disc, b, sigma, T, type}.price();
    stan::math::grad(price.vi_);

    double ad_dS = S.adj(), ad_dK = K.adj(), ad_dr = r_disc.adj(), ad_db = b.adj(),
           ad_ds = sigma.adj();
    stan::math::recover_memory();

    const double eps = 1e-7;
    auto p = [&](double S_, double K_, double r_, double b_, double s_) {
        return GBS<double>{S_, K_, r_, b_, s_, T, type}.price();
    };
    double fd_dS = (p(S0 + eps, K0, r0, b0, sigma0) - p(S0 - eps, K0, r0, b0, sigma0)) / (2 * eps);
    double fd_dK = (p(S0, K0 + eps, r0, b0, sigma0) - p(S0, K0 - eps, r0, b0, sigma0)) / (2 * eps);
    double fd_dr = (p(S0, K0, r0 + eps, b0, sigma0) - p(S0, K0, r0 - eps, b0, sigma0)) / (2 * eps);
    double fd_db = (p(S0, K0, r0, b0 + eps, sigma0) - p(S0, K0, r0, b0 - eps, sigma0)) / (2 * eps);
    double fd_ds = (p(S0, K0, r0, b0, sigma0 + eps) - p(S0, K0, r0, b0, sigma0 - eps)) / (2 * eps);

    std::cout << "  GBS Stan var vs FD:\n";
    auto row = [](const char* name, double ad, double fd) {
        std::cout << "    " << std::setw(12) << name << "  AD=" << std::setw(18)
                  << std::setprecision(12) << ad << "  FD=" << std::setw(18) << fd
                  << "  err=" << std::scientific << std::abs(ad - fd) << std::fixed << "\n";
    };
    row("dV/dS", ad_dS, fd_dS);
    row("dV/dK", ad_dK, fd_dK);
    row("dV/dr_disc", ad_dr, fd_dr);
    row("dV/db", ad_db, fd_db);
    row("dV/dsigma", ad_ds, fd_ds);
}

struct GBSFunctor {
    double T;
    OptionType type;
    template <typename TT>
    TT operator()(const Eigen::Matrix<TT, Eigen::Dynamic, 1>& theta) const {
        return GBS<TT>{theta(0), theta(1), theta(2), theta(3), theta(4), T, type}.price();
    }
};

void checkGBSStanHessian(double S0, double K0, double r0, double b0, double sigma0, double T,
                         OptionType type) {
    Eigen::VectorXd x(5);
    x << S0, K0, r0, b0, sigma0;

    double fx;
    Eigen::VectorXd grad(5);
    Eigen::MatrixXd H(5, 5);
    stan::math::hessian(GBSFunctor{T, type}, x, fx, grad, H);

    // Verify gradient against FD
    const double eps = 1e-7;
    auto p = [&](double S_, double K_, double r_, double b_, double s_) {
        return GBS<double>{S_, K_, r_, b_, s_, T, type}.price();
    };
    double fd_grad[] = {
        (p(S0 + eps, K0, r0, b0, sigma0) - p(S0 - eps, K0, r0, b0, sigma0)) / (2 * eps),
        (p(S0, K0 + eps, r0, b0, sigma0) - p(S0, K0 - eps, r0, b0, sigma0)) / (2 * eps),
        (p(S0, K0, r0 + eps, b0, sigma0) - p(S0, K0, r0 - eps, b0, sigma0)) / (2 * eps),
        (p(S0, K0, r0, b0 + eps, sigma0) - p(S0, K0, r0, b0 - eps, sigma0)) / (2 * eps),
        (p(S0, K0, r0, b0, sigma0 + eps) - p(S0, K0, r0, b0, sigma0 - eps)) / (2 * eps),
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

    std::cout << "  GBS hessian: price_err=" << std::scientific
              << std::abs(fx - p(S0, K0, r0, b0, sigma0)) << "  grad_err=" << grad_err
              << "  hess_err=" << hess_err << std::fixed << "\n";

    const char* names[] = {"S", "K", "r_disc", "b", "sigma"};
    std::cout << "    Hessian (AD):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "      [";
        for (int j = 0; j < 5; ++j)
            std::cout << std::setw(14) << std::setprecision(6) << H(i, j);
        std::cout << " ]  " << names[i] << "\n";
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    std::cout << std::string(60, '=') << "\n";
    std::cout << "  Black-76 Tests\n";
    std::cout << std::string(60, '=') << "\n\n";

    struct B76Case {
        double D, F, K, sigma, T;
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
        std::cout << "-- " << tc.label << " --\n";
        checkB76Greeks1_FD(tc.D, tc.F, tc.K, tc.sigma, tc.T, tc.type);
        checkB76Greeks2_FD(tc.D, tc.F, tc.K, tc.sigma, tc.T, tc.type);
        checkB76StanVar(tc.D, tc.F, tc.K, tc.sigma, tc.T, tc.type);
        checkB76StanHessian(tc.D, tc.F, tc.K, tc.sigma, tc.T, tc.type);
        std::cout << "\n";
    }

    std::cout << std::string(60, '=') << "\n";
    std::cout << "  GBS Tests\n";
    std::cout << std::string(60, '=') << "\n\n";

    struct GBSCase {
        double S, K, r_disc, b, sigma, T;
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
        std::cout << "-- " << tc.label << " (S=" << tc.S << " K=" << tc.K << " r_disc=" << tc.r_disc
                  << " b=" << tc.b << " sigma=" << tc.sigma << " T=" << tc.T << ") --\n";
        checkGBSEquivalence(tc.S, tc.K, tc.r_disc, tc.b, tc.sigma, tc.T, tc.type);
        checkGBSStanVar(tc.S, tc.K, tc.r_disc, tc.b, tc.sigma, tc.T, tc.type);
        checkGBSStanHessian(tc.S, tc.K, tc.r_disc, tc.b, tc.sigma, tc.T, tc.type);
        std::cout << "\n";
    }

    return 0;
}
