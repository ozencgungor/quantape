// test_state_curve.cpp — analytic state-to-curve evaluation gates
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/sde/StateCurveMap.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <span>
#include <utility>
#include <vector>

#include "support/GtestSupport.h"

using namespace quantape;

namespace {

std::atomic<std::uint64_t> g_allocations{0};
std::atomic<bool> g_countAllocations{false};

void noteAllocation(std::size_t bytes) {
    (void)bytes;
    if (g_countAllocations.load(std::memory_order_relaxed)) {
        g_allocations.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace

// Global allocation-function replacements: the zero-allocation case flips
// `g_countAllocations` around the evaluation loop, so a heap touch anywhere in
// the hot path lands in the counter. The scalar, array and aligned forms are
// replaced as a family (the benchmark alloc twins use the same pattern).
void* operator new(std::size_t size) {
    noteAllocation(size);
    if (void* p = std::malloc(size == 0 ? 1 : size)) {
        return p;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void* operator new(std::size_t size, std::align_val_t alignment) {
    noteAllocation(size);
    void* p = nullptr;
    if (posix_memalign(&p, static_cast<std::size_t>(alignment), size == 0 ? 1 : size) == 0) {
        return p;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
    return ::operator new(size, alignment);
}

void operator delete(void* p) noexcept {
    std::free(p);
}
void operator delete[](void* p) noexcept {
    ::operator delete(p);
}
void operator delete(void* p, std::size_t) noexcept {
    ::operator delete(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    ::operator delete(p);
}
void operator delete(void* p, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
    ::operator delete(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    ::operator delete(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    ::operator delete(p);
}

namespace {

using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using markets::SpreadCurve;

constexpr std::size_t kNObs = 4;
constexpr std::size_t kNPay = 8;

DiscountCurve<double> makeOisCurve() {
    return DiscountCurve<double>({0.0, 1.0, 2.0, 3.0, 5.0}, {0.0, 0.02, 0.025, 0.03, 0.035},
                                 InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
}

SpreadCurve<double> makeSpreadCurve(std::shared_ptr<const DiscountCurve<double>> parent) {
    return SpreadCurve<double>(std::move(parent), {0.0, 1.0, 2.0, 3.0, 5.0},
                               {0.0, 0.001, 0.0012, 0.0015, 0.002});
}

std::vector<double> obsTimes() {
    return {0.0, 0.5, 1.0, 2.0};
}
std::vector<double> payTimes() {
    return {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0};
}

/// Two-factor loading block on the observation/payment grid:
/// B_0 = 0.04 * (T_j - t_k), B_1 = 0.015 * (T_j - t_k)^2, rho = 0.35.
sde::GaussianStateCurve::Loadings makeLoadings() {
    sde::GaussianStateCurve::Loadings loadings;
    loadings.factor.assign(2, std::vector<double>(kNObs * kNPay, 0.0));
    loadings.correlation = {1.0, 0.35, 0.35, 1.0};
    const std::vector<double> obs = obsTimes();
    const std::vector<double> pay = payTimes();
    for (std::size_t k = 0; k < kNObs; ++k) {
        for (std::size_t j = 0; j < kNPay; ++j) {
            const double tau = pay[j] - obs[k];
            loadings.factor[0][k * kNPay + j] = 0.04 * tau;
            loadings.factor[1][k * kNPay + j] = 0.015 * tau * tau;
        }
    }
    return loadings;
}

sde::GaussianStateCurve makeDeterministicMap(const DiscountCurve<double>& ois) {
    return sde::GaussianStateCurve::fromCurve(ois, obsTimes(), payTimes(), {});
}

} // namespace

static_assert(sde::StateCurve<sde::GaussianStateCurve, double>);

TEST(GaussianStateCurve, zeroStateReproducesOisBaseCurve) {
    const DiscountCurve<double> ois = makeOisCurve();
    const sde::GaussianStateCurve map = makeDeterministicMap(ois);
    EXPECT_TRUE(map.stateSize() == 0);
    const std::span<const double> zero;
    for (std::size_t k = 0; k < map.obsCount(); ++k) {
        for (std::size_t j = 0; j < map.payCount(); ++j) {
            const double expected =
                ois.discount(map.payTimes()[j]) / ois.discount(map.obsTimes()[k]);
            EXPECT_EQ(map.bond(k, j, zero), expected);
        }
    }
}

TEST(GaussianStateCurve, zeroStateReproducesSpreadCurve) {
    const auto parent = std::make_shared<const DiscountCurve<double>>(makeOisCurve());
    const SpreadCurve<double> spread = makeSpreadCurve(parent);
    const sde::GaussianStateCurve map =
        sde::GaussianStateCurve::fromCurve(spread, obsTimes(), payTimes(), {});
    const std::span<const double> zero;
    for (std::size_t k = 0; k < map.obsCount(); ++k) {
        for (std::size_t j = 0; j < map.payCount(); ++j) {
            const double expected =
                spread.discount(map.payTimes()[j]) / spread.discount(map.obsTimes()[k]);
            EXPECT_EQ(map.bond(k, j, zero), expected);
        }
    }
}

TEST(GaussianStateCurve, annuitySumsCouponBonds) {
    const DiscountCurve<double> ois = makeOisCurve();
    const sde::GaussianStateCurve map = makeDeterministicMap(ois);
    const std::vector<double> accruals(map.payCount(), 0.25);
    const std::span<const double> zero;

    double expected = 0.0;
    for (std::size_t j = 0; j < map.payCount(); ++j) {
        expected += accruals[j] * ois.discount(map.payTimes()[j]) / ois.discount(map.obsTimes()[1]);
    }
    CHECK_CLOSE_REL("full annuity", map.annuity(1, 0, zero, accruals), expected, 1e-15);

    // Windowed annuity: coupons [2, 4) only.
    const std::vector<double> window{accruals[2], accruals[3]};
    const double windowExpected =
        accruals[2] * ois.discount(map.payTimes()[2]) / ois.discount(map.obsTimes()[2]) +
        accruals[3] * ois.discount(map.payTimes()[3]) / ois.discount(map.obsTimes()[2]);
    CHECK_CLOSE_REL("windowed annuity", map.annuity(2, 2, zero, window), windowExpected, 1e-15);
}

TEST(GaussianStateCurve, loadingsShiftSymmetricallyAndCarryQuadraticForm) {
    const DiscountCurve<double> ois = makeOisCurve();
    const sde::GaussianStateCurve map =
        sde::GaussianStateCurve::fromCurve(ois, obsTimes(), payTimes(), makeLoadings());
    EXPECT_TRUE(map.stateSize() == 2);

    const std::vector<double> state{0.4, -0.25};
    const std::vector<double> mirrored{-0.4, 0.25};
    const std::span<const double> stateView(state);
    const std::span<const double> mirroredView(mirrored);
    for (std::size_t k = 0; k < map.obsCount(); ++k) {
        for (std::size_t j = 0; j < map.payCount(); ++j) {
            const double base = map.baseRatio(k, j);
            const double omega = map.omega(k, j);
            // bond(x) * bond(-x) = base^2 * exp(-Omega).
            CHECK_CLOSE_REL("mirrored shift cancels the linear term",
                            map.bond(k, j, stateView) * map.bond(k, j, mirroredView),
                            base * base * std::exp(-omega), 1e-14);

            const double b0 = map.loading(0, k, j);
            const double b1 = map.loading(1, k, j);
            CHECK_CLOSE_REL("quadratic loading form", omega,
                            b0 * b0 + b1 * b1 + 2.0 * 0.35 * b0 * b1, 1e-14);
        }
    }
}

TEST(GaussianStateCurve, bondsIntoMatchesBond) {
    const DiscountCurve<double> ois = makeOisCurve();
    const sde::GaussianStateCurve map =
        sde::GaussianStateCurve::fromCurve(ois, obsTimes(), payTimes(), makeLoadings());
    const std::vector<double> state{0.15, 0.05};
    const std::span<const double> stateView(state);
    std::vector<double> out(map.payCount(), 0.0);
    map.bondsInto(1, stateView, std::span<double>(out));
    for (std::size_t j = 0; j < map.payCount(); ++j) {
        EXPECT_EQ(out[j], map.bond(1, j, stateView));
    }
}

TEST(GaussianStateCurve, evaluationAllocatesNothing) {
    const DiscountCurve<double> ois = makeOisCurve();
    const sde::GaussianStateCurve map =
        sde::GaussianStateCurve::fromCurve(ois, obsTimes(), payTimes(), makeLoadings());
    const std::vector<double> state{0.3, -0.2};
    const std::vector<double> accruals(map.payCount(), 0.25);
    const std::span<const double> stateView(state);
    const std::span<const double> accrualView(accruals);
    std::vector<double> out(map.payCount(), 0.0);
    const std::span<double> outView(out);

    g_allocations.store(0, std::memory_order_relaxed);
    g_countAllocations.store(true, std::memory_order_relaxed);
    for (std::size_t rep = 0; rep < 1000; ++rep) {
        const std::size_t k = rep % map.obsCount();
        outView[0] = map.bond(k, rep % map.payCount(), stateView);
        outView[1] = map.annuity(k, 0, stateView, accrualView);
        map.bondsInto(k, stateView, outView);
    }
    g_countAllocations.store(false, std::memory_order_relaxed);
    EXPECT_EQ(g_allocations.load(std::memory_order_relaxed), 0u);
    for (const double value : out) {
        EXPECT_TRUE(std::isfinite(value));
    }
}

TEST(GaussianStateCurve, rejectsMalformedInput) {
    const DiscountCurve<double> ois = makeOisCurve();
    sde::GaussianStateCurve::Loadings bad = makeLoadings();
    bad.factor[0].pop_back();
    EXPECT_THROW(sde::GaussianStateCurve::fromCurve(ois, obsTimes(), payTimes(), std::move(bad)),
                 std::invalid_argument);
    EXPECT_THROW(sde::GaussianStateCurve::fromCurve(ois, {}, payTimes(), {}),
                 std::invalid_argument);
    EXPECT_THROW(sde::GaussianStateCurve::fromCurve(ois, obsTimes(), payTimes(), makeLoadings())
                     .bond(1, 0, std::span<const double>{}),
                 std::invalid_argument);
}

TEST(GaussianStateCurve, analyticBondEvaluatesFromLoadings) {
    const std::vector<double> loading{0.31, -0.12};
    const std::vector<double> state{0.8, -0.5};
    const double variance = 0.047;
    const double dot = 0.31 * 0.8 + (-0.12) * (-0.5);
    const double got = sde::GaussianStateCurve::analyticBond(
        0.97, std::span<const double>(loading), std::span<const double>(state), variance);
    CHECK_CLOSE_REL("analytic bond value", got, 0.97 * std::exp(-dot - 0.5 * variance), 1e-15);
    EXPECT_THROW(sde::GaussianStateCurve::analyticBond(0.97, std::span<const double>(loading),
                                                       std::span<const double>(state).first(1),
                                                       variance),
                 std::invalid_argument);
}
