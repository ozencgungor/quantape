// bench_instruments.cpp — timing and allocation harness for the instruments layer
//
// Measures the config-side materialization of the pillar families (factories,
// `Ladder`/variant construction, prepared quote-time building), the model
// quotes of every family on prepared instruments, and the config curve
// builders that drive them. Each case runs `reps` times after a warmup and
// reports p50/p99 wall time; the allocation-tracking build
// (bench_instruments_alloc) additionally reports the allocation count and
// bytes of one run:
//   ./build/release/benchmarks/bench_instruments [reps] [name-filter]
//   ./build/release/benchmarks/bench_instruments_alloc [reps] [name-filter]
#include "quantape/format/Number.h"
#include "quantape/instruments/PillarVariants.h"
#include "quantape/log/Log.h"
#include "quantape/markets/Curves/CurveBuilder.h"
#include "quantape/markets/Curves/CurveConfig.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/markets/Curves/SpreadCurve.h"
#include "quantape/markets/Curves/XccyBasisBuilder.h"

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

// ── allocation counters (compile-time opt-in, see bench_instruments_alloc) ──
//
// The factories allocate the prepared quote-time grids of every alternative
// and the bootstraps rebuild trial curves; counting every allocation needs the
// malloc/free symbols replaced. The counters and overrides live behind
// QUANTAPE_INSTRUMENTS_ALLOC_COUNTS and are built into the separate
// bench_instruments_alloc target, whose allocation column is the metric and
// whose timing column is not comparable to bench_instruments.
#if defined(QUANTAPE_INSTRUMENTS_ALLOC_COUNTS)
std::atomic<std::uint64_t> g_allocations{0};
std::atomic<std::uint64_t> g_allocatedBytes{0};

void countAllocation(std::size_t bytes) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    g_allocatedBytes.fetch_add(bytes, std::memory_order_relaxed);
}
#endif

volatile double g_sink = 0.0;

} // namespace

#if defined(QUANTAPE_INSTRUMENTS_ALLOC_COUNTS) && defined(__APPLE__)
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

#if defined(QUANTAPE_INSTRUMENTS_ALLOC_COUNTS)
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
#endif // QUANTAPE_INSTRUMENTS_ALLOC_COUNTS

namespace {

using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
using markets::SpreadCurve;

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
#if defined(QUANTAPE_INSTRUMENTS_ALLOC_COUNTS)
    timing.allocations = UINT64_MAX;
#endif
    for (int r = 0; r < reps; ++r) {
#if defined(QUANTAPE_INSTRUMENTS_ALLOC_COUNTS)
        g_allocations.store(0, std::memory_order_relaxed);
        g_allocatedBytes.store(0, std::memory_order_relaxed);
#endif
        const auto start = std::chrono::steady_clock::now();
        g_sink = runOnce();
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
#if defined(QUANTAPE_INSTRUMENTS_ALLOC_COUNTS)
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

// ── fixtures ─────────────────────────────────────────────────────────────────

struct InstrumentFixture {
    datetime::Date reference{2026, 9, 29};
    datetime::DayCounter zeroDc{datetime::DayCount::Actual365Fixed};
    datetime::Calendar calendar = datetime::Calendar::noHolidays();

    std::vector<double> gridTimes;
    std::shared_ptr<const DiscountCurve<double>> parent;
    SpreadCurve<double> forecastTarget;

    std::vector<markets::CurvePillar> discountTagged;
    std::vector<markets::ForecastPillar> forecastTagged;

    markets::CurveStackSpec stack;
    markets::CurveSpec discountSpec;
    markets::CurveSpec forecastSpec;

    std::vector<markets::PillarSpec> discountSpecs;
    std::vector<markets::PillarSpec> forecastSpecs;
    std::vector<markets::PillarSpec> xccySpecs;
    std::vector<markets::PillarSpec> fxSpecs;
    markets::XccyCurveSpec xccy;

    InstrumentFixture()
        : parent(std::make_shared<DiscountCurve<double>>(buildTarget())),
          forecastTarget(parent, gridTimes, std::vector<double>(gridTimes.size(), 0.0004),
                         InterpolationScheme::Linear) {
        buildDiscountSpecs();
        buildForecastSpecs();
        buildXccySpecs();
        stack.asOf = reference;
        stack.spotLag["EUR"] = 2;
        stack.spotLag["USD"] = 2;
        markets::FxSpotSpec spot;
        spot.descriptor = markets::FXDescriptor("EUR", "USD", reference.toIso());
        spot.spot = 1.10;
        stack.fxSpots.push_back(spot);
        discountSpec.key = markets::CurveKey{"EUR", markets::CurveRole::Discount, {}, "USD"};
        discountSpec.zeroDayCounter = zeroDc;
        discountSpec.space = InterpolationSpace::LogDiscount;
        discountSpec.scheme = InterpolationScheme::Linear;
        discountSpec.pillars = discountSpecs;
        stack.curves.push_back(discountSpec);
        forecastSpec.key = markets::CurveKey{"EUR", markets::CurveRole::Forecast, {}, "USD"};
        forecastSpec.parent = discountSpec.key;
        forecastSpec.hasParent = true;
        forecastSpec.zeroDayCounter = zeroDc;
        forecastSpec.scheme = InterpolationScheme::Linear;
        forecastSpec.forecastPillars = forecastSpecs;
        xccy.pair = "EURUSD";
    }

    DiscountCurve<double> buildTarget() {
        std::vector<datetime::Date> dates;
        dates.push_back(reference.plusMonths(6));
        for (int years = 1; years <= 8; ++years) {
            dates.push_back(reference.plusYears(years));
        }
        std::vector<double> zeros;
        for (const datetime::Date& date : dates) {
            const double t = datetime::yearFraction(reference, date, zeroDc);
            gridTimes.push_back(t);
            zeros.push_back(0.04 - 0.002 * std::exp(-0.5 * t));
        }
        return DiscountCurve<double>(reference, dates, zeroDc, zeros,
                                     InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
    }

    void buildDiscountSpecs() {
        markets::CurvePillar depositTagged;
        depositTagged.maturity = reference.plusMonths(6);
        depositTagged.kind = markets::PillarKind::Deposit;
        depositTagged.quoteDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
        depositTagged.calendar = calendar;
        depositTagged.quote = markets::impliedQuote(depositTagged, reference, *parent);
        discountTagged.push_back(depositTagged);
        markets::PillarSpec deposit;
        deposit.kind = markets::PillarSpec::Kind::Deposit;
        deposit.maturity = depositTagged.maturity;
        deposit.quote = depositTagged.quote;
        deposit.quoteDayCounter = depositTagged.quoteDayCounter;
        deposit.calendar = calendar;
        discountSpecs.push_back(deposit);
        for (int years = 1; years <= 8; ++years) {
            markets::CurvePillar oisTagged;
            oisTagged.maturity = reference.plusYears(years);
            oisTagged.kind = markets::PillarKind::OisSwap;
            oisTagged.quoteDayCounter = zeroDc;
            oisTagged.calendar = calendar;
            oisTagged.quote = markets::impliedQuote(oisTagged, reference, *parent);
            discountTagged.push_back(oisTagged);
            markets::PillarSpec ois;
            ois.kind = markets::PillarSpec::Kind::OisSwap;
            ois.maturity = oisTagged.maturity;
            ois.quote = oisTagged.quote;
            ois.quoteDayCounter = zeroDc;
            ois.calendar = calendar;
            ois.fixedTenor = datetime::Period(1, datetime::TimeUnit::Years);
            discountSpecs.push_back(ois);
        }
    }

    void buildForecastSpecs() {
        for (int years = 1; years <= 8; ++years) {
            markets::IrsPillar irs;
            irs.maturity = reference.plusYears(years);
            irs.floatTenor = datetime::Period(3, datetime::TimeUnit::Months);
            irs.floatCalendar = calendar;
            irs.floatDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            irs.fixedTenor = datetime::Period(1, datetime::TimeUnit::Years);
            irs.fixedCalendar = calendar;
            irs.fixedDayCounter = datetime::DayCounter(datetime::DayCount::Thirty360BondBasis);
            irs.quote = markets::impliedIrsRate(forecastTarget, *parent, irs, reference, zeroDc);
            markets::ForecastPillar tagged;
            tagged.kind = markets::ForecastPillar::Kind::Irs;
            tagged.irs = irs;
            forecastTagged.push_back(tagged);
            markets::PillarSpec spec;
            spec.kind = markets::PillarSpec::Kind::Irs;
            spec.maturity = irs.maturity;
            spec.quote = irs.quote;
            spec.floatTenor = irs.floatTenor;
            spec.floatCalendar = irs.floatCalendar;
            spec.floatDayCounter = irs.floatDayCounter;
            spec.fixedTenor = irs.fixedTenor;
            spec.fixedCalendar = irs.fixedCalendar;
            spec.fixedDayCounter = irs.fixedDayCounter;
            forecastSpecs.push_back(spec);
        }
    }

    void buildXccySpecs() {
        for (int years = 1; years <= 5; ++years) {
            markets::PillarSpec spec;
            spec.kind = markets::PillarSpec::Kind::XccySwap;
            spec.maturity = reference.plusYears(years);
            spec.quote = 0.0002 + 0.0001 * static_cast<double>(years);
            spec.foreignTenor = datetime::Period(3, datetime::TimeUnit::Months);
            spec.domesticTenor = datetime::Period(3, datetime::TimeUnit::Months);
            spec.foreignCalendar = calendar;
            spec.domesticCalendar = calendar;
            spec.foreignDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            spec.domesticDayCounter = datetime::DayCounter(datetime::DayCount::Actual360);
            spec.foreignPaymentLag = 2;
            spec.domesticPaymentLag = 2;
            xccySpecs.push_back(spec);
        }
        for (int months = 1; months <= 3; ++months) {
            markets::PillarSpec spec;
            spec.kind = markets::PillarSpec::Kind::FxSwap;
            spec.maturity = reference.plusMonths(months);
            spec.fxConvention = markets::QuoteConvention::Points;
            spec.fxPoints = 0.001 * static_cast<double>(months);
            spec.quoteDayCounter = zeroDc;
            fxSpecs.push_back(spec);
        }
    }
};

// ── materialization ──────────────────────────────────────────────────────────

void benchMaterialization(const InstrumentFixture& fixture, int reps) {
    const std::string suffix = std::to_string(fixture.discountSpecs.size()) + " pillars";
    bench("toDiscountInstrument " + suffix, reps, [&] {
        double sink = 0.0;
        for (const markets::PillarSpec& spec : fixture.discountSpecs) {
            const markets::DiscountInstrument instrument =
                markets::toDiscountInstrument(spec, fixture.reference, fixture.zeroDc);
            sink += static_cast<double>(instrument.date().serial());
        }
        return sink;
    });
    const std::string forecastSuffix = std::to_string(fixture.forecastSpecs.size()) + " pillars";
    bench("toForecastInstrument " + forecastSuffix, reps, [&] {
        double sink = 0.0;
        for (const markets::PillarSpec& spec : fixture.forecastSpecs) {
            const markets::ForecastInstrument instrument =
                markets::toForecastInstrument(spec, fixture.reference, fixture.zeroDc);
            sink += static_cast<double>(instrument.date().serial());
        }
        return sink;
    });
    bench("toXccyInstrument 5 pillars", reps, [&] {
        double sink = 0.0;
        for (const markets::PillarSpec& spec : fixture.xccySpecs) {
            const markets::XccyInstrument instrument =
                markets::toXccyInstrument(spec, fixture.xccy);
            sink += instrument.target();
        }
        return sink;
    });
    bench("toFxInstrument 3 pillars", reps, [&] {
        double sink = 0.0;
        for (const markets::PillarSpec& spec : fixture.fxSpecs) {
            const markets::FxInstrument instrument =
                markets::toFxInstrument(fixture.stack, spec, fixture.xccy, "bench");
            sink += instrument.target();
        }
        return sink;
    });
}

// ── prepared times and model quotes ──────────────────────────────────────────

void benchPreparedQuotes(const InstrumentFixture& fixture, int reps) {
    const instruments::XccyBasisSwap xccyPillar = std::get<instruments::XccyBasisSwap>(
        markets::toXccyInstrument(fixture.xccySpecs.back(), fixture.xccy).instrument());
    const markets::ForecastInstrument forecast = markets::toForecastInstrument(
        fixture.forecastSpecs.back(), fixture.reference, fixture.zeroDc);
    const DiscountCurve<double>& parent = *fixture.parent;
    const markets::SpreadCurve<double>& forecastCurve = fixture.forecastTarget;

    bench("prepareOisCouponTimes 8y annual", reps, [&] {
        const std::vector<pricing::OisCouponTimes> coupons = pricing::prepareOisCouponTimes(
            fixture.reference, fixture.reference.plusYears(8),
            datetime::Period(1, datetime::TimeUnit::Years), fixture.calendar,
            datetime::BusinessDayConvention::ModifiedFollowing, 0, fixture.zeroDc, false, 0.0,
            fixture.reference, fixture.zeroDc);
        return static_cast<double>(coupons.size());
    });
    bench("impliedOisParRate 8y annual", reps, [&] {
        const std::vector<pricing::OisCouponTimes> coupons = pricing::prepareOisCouponTimes(
            fixture.reference, fixture.reference.plusYears(8),
            datetime::Period(1, datetime::TimeUnit::Years), fixture.calendar,
            datetime::BusinessDayConvention::ModifiedFollowing, 0, fixture.zeroDc, false, 0.0,
            fixture.reference, fixture.zeroDc);
        return pricing::impliedOisParRate<double>(coupons, parent);
    });
    bench("prepareIrsQuoteTimes 8y 3M/1Y", reps, [&] {
        const markets::IrsPillar& irs = std::get<instruments::Irs>(forecast.instrument());
        return static_cast<double>(
            pricing::makeIrsQuoteTimes(irs, fixture.reference, fixture.zeroDc)
                .annuityCoupons.size());
    });
    bench("impliedQuote forecast Irs 8 pillars", reps, [&] {
        double sink = 0.0;
        for (const markets::PillarSpec& spec : fixture.forecastSpecs) {
            const markets::ForecastInstrument instrument =
                markets::toForecastInstrument(spec, fixture.reference, fixture.zeroDc);
            sink += instrument.template impliedQuote<double>(
                markets::ForecastSet<markets::SpreadCurve<double>, DiscountCurve<double>>{
                    forecastCurve, parent});
        }
        return sink;
    });
    bench("impliedQuote xccy 5 pillars", reps, [&] {
        double sink = 0.0;
        for (const markets::PillarSpec& spec : fixture.xccySpecs) {
            instruments::XccyBasisSwap instrument = std::get<instruments::XccyBasisSwap>(
                markets::toXccyInstrument(spec, fixture.xccy).instrument());
            instrument.prepare(fixture.reference, fixture.zeroDc);
            sink += instrument.template impliedQuote<double>(
                markets::XccySet<DiscountCurve<double>, DiscountCurve<double>,
                                 DiscountCurve<double>, DiscountCurve<double>>{parent, parent,
                                                                               parent, parent});
        }
        return sink;
    });
    bench("impliedQuote fx 3 pillars", reps, [&] {
        double sink = 0.0;
        for (const markets::PillarSpec& spec : fixture.fxSpecs) {
            const instruments::FxSwap instrument =
                markets::toFxInstrument(fixture.stack, spec, fixture.xccy, "bench");
            sink += instrument.template impliedQuote<double>(
                markets::FxSet<DiscountCurve<double>, DiscountCurve<double>>{parent, parent});
        }
        return sink;
    });
    bench("prepareXccyPillarTimes 8y", reps, [&] {
        const pricing::XccyPillarTimes times =
            pricing::makeXccyPillarTimes(xccyPillar, fixture.reference, fixture.zeroDc);
        return static_cast<double>(times.foreign.coupons.size() + times.domestic.coupons.size());
    });
}

// ── config builders ──────────────────────────────────────────────────────────

void benchConfigBuilders(const InstrumentFixture& fixture, int reps) {
    bench("buildCurve config 9 pillars", reps, [&] {
        const DiscountCurve<double> curve =
            markets::buildCurve(fixture.stack, fixture.discountSpec);
        return curve.zeros().back();
    });
    bench("buildForecastCurve config 8 pillars", reps, [&] {
        const SpreadCurve<double> curve =
            markets::buildForecastCurve(fixture.stack, fixture.forecastSpec, fixture.parent);
        return curve.spreadNodes().zeros().back();
    });
    bench("bootstrapDiscountCurve tagged 9 pillars", reps, [&] {
        const DiscountCurve<double> curve = markets::bootstrapDiscountCurve(
            fixture.reference, fixture.zeroDc, InterpolationSpace::LogDiscount,
            InterpolationScheme::Linear, fixture.discountTagged);
        return curve.zeros().back();
    });
    bench("bootstrapForecastCurve tagged 8 pillars", reps, [&] {
        const SpreadCurve<double> curve = markets::bootstrapForecastCurve(
            fixture.parent, nullptr, fixture.reference, fixture.zeroDc, InterpolationScheme::Linear,
            fixture.forecastTagged);
        return curve.spreadNodes().zeros().back();
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
    const InstrumentFixture fixture;
    benchMaterialization(fixture, reps);
    benchPreparedQuotes(fixture, reps);
    benchConfigBuilders(fixture, reps);
    QTA_LOG_INFO("bench", "done");
    return 0;
}
