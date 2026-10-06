// bench_fx.cpp — timing and allocation harness for FX pricing and xccy bootstraps
//
// Measures the FX forward-point and cross-currency basis bootstraps, the
// implied forward points, the reset-aware cross-currency Jacobian rows, the
// cross-currency stack factor assembly and the Fx.h pricers and greeks in
// double and AD modes. Each case runs `reps` times after a warmup and reports
// p50/p99 wall time; the allocation-tracking build (bench_fx_alloc)
// additionally reports the allocation count and bytes of one run:
//   ./build/release/benchmarks/bench_fx [reps] [name-filter]
//   ./build/release/benchmarks/bench_fx_alloc [reps] [name-filter]
#include "quantape/math/StanMath.h"

#include "quantape/format/Number.h"
#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/FxSwapBuilder.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/StackRisk.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"
#include "quantape/markets/Curves/XccyRisk.h"
#include "quantape/pricing/Fx.h"

#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

using namespace quantape;

namespace {

// ── allocation counters (compile-time opt-in, see bench_fx_alloc) ──
//
// The bootstraps rebuild trial curves, the row builders allocate scratch and
// the AD pricers allocate tape temporaries; counting every allocation needs
// the malloc/free symbols replaced. The counters and overrides therefore live
// behind QUANTAPE_FX_ALLOC_COUNTS and are built into the separate
// bench_fx_alloc target, whose allocation column is the metric and whose
// timing column is not comparable to bench_fx.
#if defined(QUANTAPE_FX_ALLOC_COUNTS)
std::atomic<std::uint64_t> g_allocations{0};
std::atomic<std::uint64_t> g_allocatedBytes{0};

void countAllocation(std::size_t bytes) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    g_allocatedBytes.fetch_add(bytes, std::memory_order_relaxed);
}
#endif

volatile double g_sink = 0.0;

} // namespace

#if defined(QUANTAPE_FX_ALLOC_COUNTS) && defined(__APPLE__)
#include <malloc/malloc.h>

extern "C" void* malloc(std::size_t size) {
    countAllocation(size);
    return malloc_zone_malloc(malloc_default_zone(), size);
}
extern "C" void free(void* pointer) {
    if (pointer != nullptr) {
        malloc_zone_free(malloc_default_zone(), pointer);
    }
}
#endif

#if defined(QUANTAPE_FX_ALLOC_COUNTS)
void* operator new(std::size_t size) {
    countAllocation(size);
#if defined(__APPLE__)
    void* p = malloc_zone_malloc(malloc_default_zone(), size);
#else
    void* p = std::malloc(size);
#endif
    if (p != nullptr) {
        return p;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
    countAllocation(size);
#if defined(__APPLE__)
    void* p = malloc_zone_malloc(malloc_default_zone(), size);
#else
    void* p = std::malloc(size);
#endif
    if (p != nullptr) {
        return p;
    }
    throw std::bad_alloc();
}
void* operator new(std::size_t size, std::align_val_t alignment) {
    countAllocation(size);
    void* p = nullptr;
    if (posix_memalign(&p, static_cast<std::size_t>(alignment), size) == 0) {
        return p;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return ::operator new(size, alignment);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    countAllocation(size);
#if defined(__APPLE__)
    return malloc_zone_malloc(malloc_default_zone(), size);
#else
    return std::malloc(size);
#endif
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return ::operator new(size, std::nothrow);
}
void operator delete(void* p) noexcept {
#if defined(__APPLE__)
    if (p != nullptr) {
        malloc_zone_free(malloc_default_zone(), p);
    }
#else
    std::free(p);
#endif
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
#if defined(__APPLE__)
    if (p != nullptr) {
        malloc_zone_free(malloc_default_zone(), p);
    }
#else
    std::free(p);
#endif
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
void operator delete(void* p, const std::nothrow_t&) noexcept {
    ::operator delete(p);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
    ::operator delete(p);
}
#endif // QUANTAPE_FX_ALLOC_COUNTS

namespace {

using markets::DiscountCurve;
using markets::FxSwapPillar;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using markets::SpreadCurve;
using markets::XccyMixedPillar;
using markets::XccyNotionalMode;
using markets::XccyPillar;
using stan::math::fvar;
using stan::math::var;

std::string g_filter;

bool selected(std::string_view name) {
    return g_filter.empty() || name.find(g_filter) != std::string_view::npos;
}

struct Timing {
    double p50Us = 0.0;
    double p99Us = 0.0;
    std::uint64_t allocations = 0;
    std::uint64_t allocatedBytes = 0;
};

/// Warm up once, then time `reps` runs and reduce to p50/p99. On the
/// allocation build the reported count is the smallest seen, which drops the
/// one-off allocations from returning the result into the caller's sink.
void bench(const std::string& name, int reps, const std::function<double()>& runOnce) {
    if (!selected(name)) {
        return;
    }
    g_sink = runOnce();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(reps));
    Timing timing;
#if defined(QUANTAPE_FX_ALLOC_COUNTS)
    timing.allocations = UINT64_MAX;
#endif
    for (int r = 0; r < reps; ++r) {
#if defined(QUANTAPE_FX_ALLOC_COUNTS)
        g_allocations.store(0, std::memory_order_relaxed);
        g_allocatedBytes.store(0, std::memory_order_relaxed);
#endif
        const auto start = std::chrono::steady_clock::now();
        g_sink = runOnce();
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
#if defined(QUANTAPE_FX_ALLOC_COUNTS)
        const std::uint64_t allocations = g_allocations.load(std::memory_order_relaxed);
        if (allocations < timing.allocations) {
            timing.allocations = allocations;
            timing.allocatedBytes = g_allocatedBytes.load(std::memory_order_relaxed);
        }
#endif
    }
    std::sort(samples.begin(), samples.end());
    const auto quantile = [&](double q) {
        const std::size_t index =
            static_cast<std::size_t>(std::ceil(q * static_cast<double>(samples.size()))) - 1;
        return samples[std::min(index, samples.size() - 1)];
    };
    timing.p50Us = quantile(0.5);
    timing.p99Us = quantile(0.99);
    QTA_LOG_INFO("bench", "{}  p50 {} us  p99 {} us  allocs {}  {} KB", name,
                 format::num(timing.p50Us, 3), format::num(timing.p99Us, 3), timing.allocations,
                 format::num(static_cast<double>(timing.allocatedBytes) / 1024.0, 1));
}

// ── curve fixtures ───────────────────────────────────────────────────────────

datetime::Calendar noHolidays() {
    return datetime::Calendar::noHolidays();
}

DiscountCurve<double> buildDiscountCurve(const datetime::Date& reference,
                                         const datetime::DayCounter& zeroDc,
                                         const std::vector<datetime::Date>& dates, double base,
                                         double slope) {
    std::vector<double> zeros;
    zeros.reserve(dates.size());
    for (const datetime::Date& date : dates) {
        zeros.push_back(base + slope * datetime::yearFraction(reference, date, zeroDc));
    }
    return DiscountCurve<double>(reference, dates, zeroDc, zeros, InterpolationSpace::LogDiscount,
                                 InterpolationScheme::Linear);
}

/// One xccy pillar of the bootstrap benchmarks, quoted at par against the
/// fixture curves.
XccyPillar makeXccyPillar(const datetime::Date& reference, const datetime::Calendar& calendar,
                          const datetime::DayCounter& zeroDc, int year,
                          const DiscountCurve<double>& foreignTarget,
                          const DiscountCurve<double>& foreignForecast,
                          const DiscountCurve<double>& domesticDiscount,
                          const DiscountCurve<double>& domesticForecast) {
    XccyPillar pillar;
    pillar.maturity = reference.plusYears(year);
    pillar.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
    pillar.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
    pillar.foreignCalendar = calendar;
    pillar.domesticCalendar = calendar;
    pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
    pillar.foreignPaymentLag = 2;
    pillar.domesticPaymentLag = 1;
    pillar.foreignBusinessDayConvention = datetime::BusinessDayConvention::Following;
    pillar.spread =
        markets::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                        domesticForecast, pillar, reference, zeroDc);
    return pillar;
}

struct FxFixture {
    datetime::Date reference{2026, 9, 29};
    datetime::DayCounter zeroDc{datetime::DayCount::Actual365Fixed};
    datetime::Calendar calendar = noHolidays();

    DiscountCurve<double> domesticDiscount;
    DiscountCurve<double> domesticForecast;
    DiscountCurve<double> foreignTarget;
    DiscountCurve<double> foreignForecast;
    double spot = 1.10;

    std::vector<FxSwapPillar> fxPillars;
    std::vector<XccyPillar> xccyPillars;
    std::vector<XccyPillar> mtmPillars;
    std::vector<XccyMixedPillar> mixedExplicit;
    std::vector<XccyMixedPillar> mixedSelf;
    std::vector<XccyMixedPillar> mixedMtM;

    FxFixture()
        : domesticDiscount(buildGrid(0.040, 0.0005)), domesticForecast(buildGrid(0.043, 0.0004)),
          foreignTarget(buildGrid(0.028, 0.0006)), foreignForecast(buildGrid(0.025, 0.0008)) {
        for (int months = 1; months <= 12; ++months) {
            FxSwapPillar pillar;
            pillar.start = reference.plusDays(2);
            pillar.maturity = reference.plusMonths(months);
            pillar.spot = spot;
            pillar.points = markets::impliedFxForwardPoints(foreignTarget, domesticDiscount, spot,
                                                            pillar, reference, zeroDc);
            fxPillars.push_back(pillar);
        }
        for (int year = 1; year <= 5; ++year) {
            xccyPillars.push_back(makeXccyPillar(reference, calendar, zeroDc, year, foreignTarget,
                                                 foreignForecast, domesticDiscount,
                                                 domesticForecast));
            XccyPillar mtm = xccyPillars.back();
            mtm.notional = XccyNotionalMode::MtM;
            mtm.resetForeignLeg = (year % 2 == 0);
            mtm.spread =
                markets::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                                domesticForecast, mtm, reference, zeroDc);
            mtmPillars.push_back(mtm);
        }
        for (int months : {3, 6, 9}) {
            FxSwapPillar pillar;
            pillar.start = reference.plusDays(2);
            pillar.maturity = reference.plusMonths(months);
            pillar.spot = spot;
            pillar.points = markets::impliedFxForwardPoints(foreignTarget, domesticDiscount, spot,
                                                            pillar, reference, zeroDc);
            mixedExplicit.emplace_back(pillar);
            mixedSelf.emplace_back(pillar);
            mixedMtM.emplace_back(pillar);
        }
        for (int year = 1; year <= 4; ++year) {
            const XccyPillar& explicitPillar = xccyPillars[static_cast<std::size_t>(year - 1)];
            mixedExplicit.emplace_back(explicitPillar);
            XccyPillar selfPillar = explicitPillar;
            selfPillar.spread =
                markets::impliedXccyBasisSpread(foreignTarget, foreignTarget, domesticDiscount,
                                                domesticForecast, selfPillar, reference, zeroDc);
            mixedSelf.emplace_back(selfPillar);
            XccyPillar mtm = explicitPillar;
            mtm.notional = XccyNotionalMode::MtM;
            mtm.resetForeignLeg = false;
            mtm.spread =
                markets::impliedXccyBasisSpread(foreignTarget, foreignForecast, domesticDiscount,
                                                domesticForecast, mtm, reference, zeroDc);
            mixedMtM.emplace_back(mtm);
        }
    }

    std::vector<datetime::Date> gridDates() const {
        return {reference.plusMonths(6), reference.plusYears(1), reference.plusYears(2),
                reference.plusYears(3),  reference.plusYears(4), reference.plusYears(5)};
    }

    DiscountCurve<double> buildGrid(double base, double slope) const {
        return buildDiscountCurve(reference, zeroDc, gridDates(), base, slope);
    }
};

void benchFxBootstraps(const FxFixture& fixture, int reps) {
    const std::string fxName = std::to_string(fixture.fxPillars.size()) + " pillars";
    const std::string mixedName = std::to_string(fixture.mixedExplicit.size()) + " pillars";
    bench("impliedFxForwardPoints " + fxName, reps, [&] {
        double sink = 0.0;
        for (const FxSwapPillar& pillar : fixture.fxPillars) {
            sink += markets::impliedFxForwardPoints(fixture.foreignTarget, fixture.domesticDiscount,
                                                    fixture.spot, pillar, fixture.reference,
                                                    fixture.zeroDc);
        }
        return sink;
    });
    bench("bootstrapFxDiscountCurve " + fxName, reps, [&] {
        const DiscountCurve<double> curve = markets::bootstrapFxDiscountCurve(
            fixture.domesticDiscount, fixture.reference, fixture.zeroDc,
            InterpolationSpace::LogDiscount, InterpolationScheme::Linear, fixture.fxPillars);
        return curve.discount(0.5);
    });
    bench("bootstrapMixedXccyDiscountCurve explicit " + mixedName, reps, [&] {
        const DiscountCurve<double> curve = markets::bootstrapMixedXccyDiscountCurve(
            fixture.domesticDiscount, fixture.domesticForecast, fixture.foreignForecast,
            fixture.reference, fixture.zeroDc, InterpolationSpace::LogDiscount,
            InterpolationScheme::Linear, fixture.mixedExplicit);
        return curve.discount(0.5);
    });
    bench("bootstrapMixedXccyDiscountCurve self " + mixedName, reps, [&] {
        const DiscountCurve<double> curve = markets::bootstrapMixedXccyDiscountCurve(
            fixture.domesticDiscount, fixture.domesticForecast, fixture.reference, fixture.zeroDc,
            InterpolationSpace::LogDiscount, InterpolationScheme::Linear, fixture.mixedSelf);
        return curve.discount(0.5);
    });
    bench("bootstrapMixedXccyDiscountCurve mtm " + mixedName, reps, [&] {
        const DiscountCurve<double> curve = markets::bootstrapMixedXccyDiscountCurve(
            fixture.domesticDiscount, fixture.domesticForecast, fixture.foreignForecast,
            fixture.reference, fixture.zeroDc, InterpolationSpace::LogDiscount,
            InterpolationScheme::Linear, fixture.mixedMtM);
        return curve.discount(0.5);
    });
}

// ── reset-aware rows fixture ─────────────────────────────────────────────────

struct MtMRowsFixture {
    datetime::Date reference{2026, 9, 29};
    datetime::DayCounter zeroDc{datetime::DayCount::Actual365Fixed};
    datetime::Calendar calendar = noHolidays();

    std::shared_ptr<DiscountCurve<double>> foreignDiscount;
    std::shared_ptr<DiscountCurve<double>> foreignBase;
    std::shared_ptr<SpreadCurve<double>> foreignParent;
    std::shared_ptr<SpreadCurve<double, SpreadCurve<double>>> foreignForecast;
    std::shared_ptr<DiscountCurve<double>> domesticDiscount;
    std::shared_ptr<DiscountCurve<double>> domesticParent;
    std::shared_ptr<SpreadCurve<double>> domesticForecast;
    std::vector<XccyPillar> constPillars;
    std::vector<XccyPillar> mtmPillars;

    MtMRowsFixture() {
        const std::vector<datetime::Date> dates{reference.plusYears(1), reference.plusYears(2),
                                                reference.plusYears(3), reference.plusYears(4)};
        foreignDiscount = std::make_shared<DiscountCurve<double>>(
            buildDiscountCurve(reference, zeroDc, dates, 0.030, 0.0006));
        foreignBase = std::make_shared<DiscountCurve<double>>(
            buildDiscountCurve(reference, zeroDc, dates, 0.025, 0.0007));
        domesticDiscount = std::make_shared<DiscountCurve<double>>(
            buildDiscountCurve(reference, zeroDc, dates, 0.040, 0.0005));
        domesticParent = std::make_shared<DiscountCurve<double>>(
            buildDiscountCurve(reference, zeroDc, dates, 0.040, 0.0005));
        const std::vector<double> nodeTimes{0.0, 1.0, 2.0, 3.0, 4.0};
        foreignParent = std::make_shared<SpreadCurve<double>>(
            foreignBase, nodeTimes, std::vector<double>{0.0, 0.0004, 0.0005, 0.0006, 0.0007},
            InterpolationScheme::Linear);
        foreignForecast = std::make_shared<SpreadCurve<double, SpreadCurve<double>>>(
            foreignParent, nodeTimes, std::vector<double>{0.0, 0.0002, 0.0003, 0.0004, 0.0005},
            InterpolationScheme::Linear);
        domesticForecast = std::make_shared<SpreadCurve<double>>(
            domesticParent, nodeTimes, std::vector<double>{0.0, 0.0003, 0.0004, 0.0005, 0.0006},
            InterpolationScheme::Linear);
        for (int year = 1; year <= 4; ++year) {
            XccyPillar pillar;
            pillar.maturity = reference.plusYears(year);
            pillar.foreignTenor = datetime::Period(7, datetime::TimeUnit::Months);
            pillar.domesticTenor = datetime::Period(4, datetime::TimeUnit::Months);
            pillar.foreignPaymentLag = 2;
            pillar.domesticPaymentLag = 1;
            pillar.foreignBusinessDayConvention = datetime::BusinessDayConvention::Following;
            pillar.domesticBusinessDayConvention =
                datetime::BusinessDayConvention::ModifiedFollowing;
            pillar.foreignCalendar = calendar;
            pillar.domesticCalendar = calendar;
            pillar.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            pillar.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            constPillars.push_back(pillar);
            XccyPillar mtm = pillar;
            mtm.notional = XccyNotionalMode::MtM;
            mtm.resetForeignLeg = (year % 2 == 0);
            mtmPillars.push_back(mtm);
        }
    }
};

void benchXccyRows(const MtMRowsFixture& fixture, int reps) {
    const markets::StackCurveView::Ptr foreignDiscountView =
        markets::StackCurveView::make(*fixture.foreignDiscount);
    const markets::StackCurveView::Ptr foreignForecastView =
        markets::StackCurveView::make(*fixture.foreignForecast);
    const markets::StackCurveView::Ptr domesticDiscountView =
        markets::StackCurveView::make(*fixture.domesticDiscount);
    const markets::StackCurveView::Ptr domesticForecastView =
        markets::StackCurveView::make(*fixture.domesticForecast);
    const std::string suffix = std::to_string(fixture.constPillars.size()) + " pillars";
    bench("xccy rows view const " + suffix, reps, [&] {
        double sink = 0.0;
        for (const XccyPillar& pillar : fixture.constPillars) {
            const std::vector<markets::XccyRowBlock> blocks = markets::xccySwapJacobianRowsView(
                *foreignDiscountView, *foreignForecastView, *domesticDiscountView,
                *domesticForecastView, pillar, fixture.reference, fixture.zeroDc);
            sink += blocks.back().row.back();
        }
        return sink;
    });
    bench("xccy rows view mtm " + suffix, reps, [&] {
        double sink = 0.0;
        for (const XccyPillar& pillar : fixture.mtmPillars) {
            const std::vector<markets::XccyRowBlock> blocks = markets::xccySwapJacobianRowsView(
                *foreignDiscountView, *foreignForecastView, *domesticDiscountView,
                *domesticForecastView, pillar, fixture.reference, fixture.zeroDc);
            sink += blocks.back().row.back();
        }
        return sink;
    });
}

// ── cross-currency stack fixture ─────────────────────────────────────────────

struct RootFixture {
    datetime::Date reference{2026, 9, 29};
    datetime::DayCounter zeroDc{datetime::DayCount::Actual365Fixed};
    datetime::Calendar calendar = noHolidays();
    std::vector<markets::CurvePillar> pillars;
    DiscountCurve<double> root;

    explicit RootFixture(int years)
        : pillars(makePillars(reference, zeroDc, calendar, years)),
          root(markets::bootstrapDiscountCurve(reference, zeroDc, InterpolationSpace::LogDiscount,
                                               InterpolationScheme::Linear, pillars)) {}

private:
    static std::vector<markets::CurvePillar> makePillars(const datetime::Date& reference,
                                                         const datetime::DayCounter& zeroDc,
                                                         const datetime::Calendar& calendar,
                                                         int years) {
        std::vector<datetime::Date> dates;
        for (int year = 1; year <= years; ++year) {
            dates.push_back(reference.plusYears(year));
        }
        const DiscountCurve<double> target =
            buildDiscountCurve(reference, zeroDc, dates, 0.03, 0.0004);
        std::vector<markets::CurvePillar> pillars;
        for (const datetime::Date& date : dates) {
            markets::CurvePillar swap;
            swap.maturity = date;
            swap.kind = markets::PillarKind::OisSwap;
            swap.quoteDayCounter = zeroDc;
            swap.calendar = calendar;
            swap.quote = markets::impliedQuote(swap, reference, target);
            pillars.push_back(swap);
        }
        return pillars;
    }
};

void benchXccyStack(int reps) {
    constexpr int kRootYears = 4;
    constexpr int kPillarCount = 4;
    RootFixture rootFixture(kRootYears);
    FxFixture fxFixture;
    const datetime::Date reference(2026, 9, 29);
    const datetime::DayCounter zeroDc(datetime::DayCount::Actual365Fixed);
    std::vector<datetime::Date> foreignDates;
    for (int year = 1; year <= kPillarCount; ++year) {
        foreignDates.push_back(reference.plusYears(year));
    }
    const DiscountCurve<double> foreignDiscount =
        buildDiscountCurve(reference, zeroDc, foreignDates, 0.028, 0.0006);
    const std::vector<double>& rootTimes = rootFixture.root.times();
    std::vector<double> foreignSpreads{0.0};
    std::vector<double> domesticSpreads{0.0};
    for (std::size_t i = 1; i < rootTimes.size(); ++i) {
        foreignSpreads.push_back(0.0002 + 0.0001 * static_cast<double>(i));
        domesticSpreads.push_back(0.0005 + 0.0001 * static_cast<double>(i));
    }
    const std::shared_ptr<DiscountCurve<double>> rootParent =
        std::make_shared<DiscountCurve<double>>(rootFixture.root);
    const SpreadCurve<double> foreignForecast(rootParent, rootTimes, foreignSpreads,
                                              InterpolationScheme::Linear);
    const SpreadCurve<double> domesticForecast(rootParent, rootTimes, domesticSpreads,
                                               InterpolationScheme::Linear);
    std::vector<XccyPillar> pillars;
    for (int year = 1; year <= kPillarCount; ++year) {
        pillars.push_back(fxFixture.xccyPillars[static_cast<std::size_t>(year - 1)]);
    }
    std::vector<double> dVdForeignZeros(foreignDiscount.size(), 0.0);
    const double t = 4.5;
    std::vector<double> weights;
    foreignDiscount.zeroNodeWeights(t, weights);
    const double df = foreignDiscount.discount(t);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        dVdForeignZeros[i] = -t * df * weights[i];
    }
    markets::XccyChildInput<SpreadCurve<double>, SpreadCurve<double>> child;
    child.foreignDiscount = &foreignDiscount;
    child.pillars = pillars;
    child.dVdForeignZeros = dVdForeignZeros;
    child.foreignForecast = &foreignForecast;
    child.domesticForecast = &domesticForecast;
    const std::vector<std::variant<decltype(child)>> children{child};
    const std::string suffix = std::to_string(kRootYears + kPillarCount) + " dim";
    bench("stackQuoteRiskXccy " + suffix, reps, [&] {
        const std::vector<markets::StackRiskEntry> entries = markets::stackQuoteRiskXccy(
            rootFixture.root, rootFixture.pillars,
            std::vector<double>(rootFixture.root.size(), 0.0), children, rootFixture.reference);
        return entries.back().points.back().delta;
    });
    bench("assembleXccyJacobianFull " + std::to_string(pillars.size()) + " pillars", reps, [&] {
        std::vector<double> f;
        std::vector<double> c;
        std::vector<double> g;
        std::vector<double> h;
        markets::assembleXccyJacobianFull(foreignDiscount, foreignForecast,
                                          fxFixture.domesticDiscount, domesticForecast, pillars,
                                          rootFixture.reference, f, c, g, h);
        return f.back() + c.back() + g.back() + h.back();
    });
}

// ── Fx.h pricers and greeks ──────────────────────────────────────────────────

struct FxPricingFixture {
    double spot = 1.10;
    double strike = 1.12;
    double notional = 2.5e6;
    double nearTime = 0.5;
    double farTime = 1.0;
    double nearBase = 0.975;
    double nearQuote = 0.973;
    double farBase = 0.951;
    double farQuote = 0.947;
    double baseDiscount = 0.955;
    double quoteDiscount = 0.945;
    double farStrike = 1.10 * 0.951 / 0.947;
    std::vector<double> baseNotionals;
    std::vector<double> baseDiscounts;
    std::vector<double> quoteCashflows;
    std::vector<double> quoteDiscounts;

    FxPricingFixture() {
        for (int k = 1; k <= 16; ++k) {
            baseNotionals.push_back(1.0e5 * static_cast<double>(k));
            baseDiscounts.push_back(std::exp(-0.03 * static_cast<double>(k) / 4.0));
            quoteCashflows.push_back(-2.0e4 * static_cast<double>(k));
            quoteDiscounts.push_back(std::exp(-0.04 * static_cast<double>(k) / 4.0));
        }
    }
};

/// Nested-AD functor of the forward value over the spot and the base discount.
struct ForwardHessian {
    double strike = 0.0;
    double quoteDiscount = 0.0;
    double notional = 0.0;
    template <typename ScalarT>
    ScalarT operator()(const Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>& x) const {
        return markets::fxForwardPv(x[0], ScalarT(strike), x[1], ScalarT(quoteDiscount),
                                    ScalarT(notional));
    }
};

void benchFxPricing(const FxPricingFixture& fixture, int reps) {
    bench("fxForwardPv double 256", reps, [&] {
        double sink = 0.0;
        for (int i = 0; i < 256; ++i) {
            sink += markets::fxForwardPv(fixture.spot, fixture.strike, fixture.baseDiscount,
                                         fixture.quoteDiscount, fixture.notional);
        }
        return sink;
    });
    bench("fxSwapPv double 256", reps, [&] {
        double sink = 0.0;
        for (int i = 0; i < 256; ++i) {
            sink += markets::fxSwapPv(fixture.spot, fixture.spot, fixture.nearBase,
                                      fixture.nearQuote, fixture.farStrike, fixture.farBase,
                                      fixture.farQuote, fixture.notional);
        }
        return sink;
    });
    bench("fxCashflowPv double 256", reps, [&] {
        double sink = 0.0;
        for (int i = 0; i < 256; ++i) {
            sink +=
                markets::fxCashflowPv(fixture.spot, fixture.baseNotionals, fixture.baseDiscounts,
                                      fixture.quoteCashflows, fixture.quoteDiscounts);
        }
        return sink;
    });
    bench("fxCashflowGreeks double 256", reps, [&] {
        double sink = 0.0;
        for (int i = 0; i < 256; ++i) {
            const markets::FxGreeks<double> greeks =
                markets::fxCashflowGreeks(fixture.baseNotionals, fixture.baseDiscounts);
            sink += greeks.delta + greeks.gamma;
        }
        return sink;
    });
    bench("fxGreeksFiniteDifference 256", reps, [&] {
        const auto value = [&](double s) {
            return markets::fxCashflowPv(s, fixture.baseNotionals, fixture.baseDiscounts,
                                         fixture.quoteCashflows, fixture.quoteDiscounts);
        };
        double sink = 0.0;
        for (int i = 0; i < 256; ++i) {
            const markets::FxGreeks<double> greeks =
                markets::fxGreeksFiniteDifference(value, fixture.spot);
            sink += greeks.delta + greeks.gamma;
        }
        return sink;
    });
}

void benchFxAd(const FxPricingFixture& fixture, int reps) {
    bench("fxForwardPv var grad 64", reps, [&] {
        stan::math::recover_memory();
        var spotVar(fixture.spot);
        var strikeVar(fixture.strike);
        var baseVar(fixture.baseDiscount);
        var quoteVar(fixture.quoteDiscount);
        var notionalVar(fixture.notional);
        var value = markets::fxForwardPv(spotVar, strikeVar, baseVar, quoteVar, notionalVar);
        for (int i = 0; i < 64; ++i) {
            value += markets::fxForwardPv(spotVar, strikeVar, baseVar, quoteVar, notionalVar);
        }
        value.grad();
        const double out = spotVar.adj();
        stan::math::recover_memory();
        return out;
    });
    bench("fxSwapPv var grad 64", reps, [&] {
        stan::math::recover_memory();
        var spotVar(fixture.spot);
        var nearVar(fixture.nearBase);
        var farVar(fixture.farBase);
        var notionalVar(fixture.notional);
        var farStrikeVar(fixture.farStrike);
        var value = markets::fxSwapPv(spotVar, var(fixture.spot), nearVar, var(fixture.nearQuote),
                                      farStrikeVar, farVar, var(fixture.farQuote), notionalVar);
        for (int i = 0; i < 64; ++i) {
            value += markets::fxSwapPv(spotVar, var(fixture.spot), nearVar, var(fixture.nearQuote),
                                       farStrikeVar, farVar, var(fixture.farQuote), notionalVar);
        }
        value.grad();
        const double out = spotVar.adj();
        stan::math::recover_memory();
        return out;
    });
    bench("fxCashflowPv var grad 64", reps, [&] {
        stan::math::recover_memory();
        var spotVar(fixture.spot);
        std::vector<var> baseNotionals;
        std::vector<var> baseDiscounts;
        std::vector<var> quoteCashflows;
        std::vector<var> quoteDiscounts;
        for (std::size_t i = 0; i < fixture.baseNotionals.size(); ++i) {
            baseNotionals.push_back(var(fixture.baseNotionals[i]));
            baseDiscounts.push_back(var(fixture.baseDiscounts[i]));
        }
        for (std::size_t i = 0; i < fixture.quoteCashflows.size(); ++i) {
            quoteCashflows.push_back(var(fixture.quoteCashflows[i]));
            quoteDiscounts.push_back(var(fixture.quoteDiscounts[i]));
        }
        var value = markets::fxCashflowPv(spotVar, baseNotionals, baseDiscounts, quoteCashflows,
                                          quoteDiscounts);
        for (int i = 0; i < 64; ++i) {
            value += markets::fxCashflowPv(spotVar, baseNotionals, baseDiscounts, quoteCashflows,
                                           quoteDiscounts);
        }
        value.grad();
        const double out = spotVar.adj();
        stan::math::recover_memory();
        return out;
    });
    bench("fx forward hessian fvar 16", reps, [&] {
        stan::math::recover_memory();
        Eigen::VectorXd point(2);
        point << fixture.spot, fixture.baseDiscount;
        double value = 0.0;
        for (int i = 0; i < 16; ++i) {
            Eigen::MatrixXd hessian;
            Eigen::VectorXd gradient;
            stan::math::hessian(
                ForwardHessian{fixture.strike, fixture.quoteDiscount, fixture.notional}, point,
                value, gradient, hessian);
        }
        stan::math::recover_memory();
        return value;
    });
}

} // namespace

int main(int argc, char** argv) {
    int reps = 50;
    if (argc > 1) {
        reps = std::max(1, std::atoi(argv[1]));
    }
    if (argc > 2) {
        g_filter = argv[2];
    }
    const FxFixture fxFixture;
    benchFxBootstraps(fxFixture, reps);
    const MtMRowsFixture rowsFixture;
    benchXccyRows(rowsFixture, reps);
    benchXccyStack(reps);
    const FxPricingFixture pricingFixture;
    benchFxPricing(pricingFixture, reps);
    benchFxAd(pricingFixture, reps);
    QTA_LOG_INFO("bench", "done");
    return 0;
}
