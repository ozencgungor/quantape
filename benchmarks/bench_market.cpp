// bench_market.cpp — timing and allocation harness for the SDE market layer
//
// Measures the numeraire views (numeraireValue / deflated / rollRatio), the
// analytic state-to-curve evaluation (bond / annuity / bondsInto) across
// schedule sizes and factor counts, and the frozen-market lookups and freeze
// across date counts. Each case runs `reps` runs after a warmup and reports
// p50/p99 wall time; the allocation-tracking build (bench_market_alloc)
// additionally reports the smallest allocation count/bytes of one run:
//   ./build/release/benchmarks/bench_market [reps] [name-filter]
//   ./build/release/benchmarks/bench_market_alloc [reps] [name-filter]
//
// The var / fvar<var> cases serve the allocation shape only and are compiled
// into the alloc twin (their timing is distorted by the shim).
#if defined(QUANTAPE_MARKET_ALLOC_COUNTS)
#include "quantape/math/StanMath.h" // must be first when AD scalars are enabled
#endif

#include "quantape/datetime/Period.h"
#include "quantape/format/Number.h"
#include "quantape/log/Log.h"
#include "quantape/markets/Curves/Curve.h"
#include "quantape/markets/Curves/CurveConfig.h"
#include "quantape/markets/Curves/CurveOnGrid.h"
#include "quantape/markets/Curves/DiscountCurve.h"
#include "quantape/sde/FrozenMarket.h"
#include "quantape/sde/Measure.h"
#include "quantape/sde/MeasureAdjustment.h"
#include "quantape/sde/Numeraire.h"
#include "quantape/sde/StateCurveMap.h"

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
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// ── allocation counters (compile-time opt-in, see bench_market_alloc) ──
//
// Stan's arena and the evaluation helpers allocate through malloc, not operator
// new, so counting every allocation needs the malloc/free symbols replaced; the
// overrides therefore live behind QUANTAPE_MARKET_ALLOC_COUNTS and are built
// into the separate bench_market_alloc target, whose allocation column is the
// metric and whose timing column is not comparable to bench_market.
#if defined(QUANTAPE_MARKET_ALLOC_COUNTS)
std::atomic<std::uint64_t> g_allocations{0};
std::atomic<std::uint64_t> g_allocatedBytes{0};

void countAllocation(std::size_t bytes) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    g_allocatedBytes.fetch_add(bytes, std::memory_order_relaxed);
}
#endif

volatile double g_sink = 0.0;

} // namespace

#if defined(QUANTAPE_MARKET_ALLOC_COUNTS) && defined(__APPLE__)
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

#if defined(QUANTAPE_MARKET_ALLOC_COUNTS)
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
#endif // QUANTAPE_MARKET_ALLOC_COUNTS

using namespace quantape;

namespace {

using markets::CurveHandle;
using markets::CurveKey;
using markets::CurveRole;
using markets::DiscountCurve;
using markets::InterpolationScheme;
using markets::InterpolationSpace;
std::string g_filter;

bool selected(std::string_view name) {
    return g_filter.empty() || name.find(g_filter) != std::string_view::npos;
}

struct Timing {
    double p50Us = 0.0;
    double p99Us = 0.0;
    double meanUs = 0.0;
    std::uint64_t allocations = 0;
    std::uint64_t allocatedBytes = 0;
};

/// Warm up once, then time `reps` runs and reduce to p50/p99. On the
/// allocation build the reported count is the smallest seen, which drops the
/// one-off allocations from returning the result into the caller's sink.
void bench_us(const std::string& name, double per_unit, int reps,
              const std::function<double()>& fn) {
    if (!selected(name)) {
        return;
    }
    g_sink = fn();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(reps));
    Timing timing;
#if defined(QUANTAPE_MARKET_ALLOC_COUNTS)
    timing.allocations = UINT64_MAX;
#endif
    for (int r = 0; r < reps; ++r) {
#if defined(QUANTAPE_MARKET_ALLOC_COUNTS)
        g_allocations.store(0, std::memory_order_relaxed);
        g_allocatedBytes.store(0, std::memory_order_relaxed);
#endif
        const auto start = std::chrono::steady_clock::now();
        g_sink = fn();
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
#if defined(QUANTAPE_MARKET_ALLOC_COUNTS)
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
    double sum = 0.0;
    for (double sample : samples) {
        sum += sample;
    }
    timing.meanUs = sum / static_cast<double>(samples.size());
    const double nsPerUnit = per_unit > 0.0 ? timing.meanUs * 1000.0 / per_unit : 0.0;
    const double mUnitsPerSecond = per_unit > 0.0 ? per_unit / timing.meanUs : 0.0;
    QTA_LOG_INFO("bench", "{}  p50 {} us  p99 {} us  {} ns/unit  {} Munit/s  allocs {}  {} KB",
                 name, quantape::format::num(timing.p50Us, 3),
                 quantape::format::num(timing.p99Us, 3), quantape::format::num(nsPerUnit, 3),
                 quantape::format::num(mUnitsPerSecond, 3), timing.allocations,
                 quantape::format::num(static_cast<double>(timing.allocatedBytes) / 1024.0, 1));
}

// ── curve and state-curve fixtures ──────────────────────────────────────────

DiscountCurve<double> makeCurve() {
    return DiscountCurve<double>(
        {0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0},
        {0.0, 0.005, 0.009, 0.013, 0.02, 0.024, 0.028, 0.032, 0.034, 0.036, 0.038},
        InterpolationSpace::LogDiscount, InterpolationScheme::Linear);
}

struct ScheduleSize {
    std::size_t nObs = 0;
    std::size_t nPay = 0;
};

const std::vector<ScheduleSize>& scheduleSizes() {
    static const std::vector<ScheduleSize> sizes = {{4, 16}, {16, 64}, {64, 256}};
    return sizes;
}

std::string tag(const ScheduleSize& size, std::size_t factors) {
    return std::to_string(size.nObs) + "x" + std::to_string(size.nPay) +
           " q=" + std::to_string(factors);
}

std::vector<double> obsTimes(std::size_t n) {
    std::vector<double> times(n);
    for (std::size_t k = 0; k < n; ++k) {
        times[k] = 0.25 * static_cast<double>(k);
    }
    return times;
}

std::vector<double> payTimes(std::size_t n) {
    std::vector<double> times(n);
    for (std::size_t j = 0; j < n; ++j) {
        times[j] = 0.25 * static_cast<double>(j + 1);
    }
    return times;
}

sde::GaussianStateCurve::Loadings makeLoadings(std::size_t nObs, std::size_t nPay,
                                               std::size_t factors) {
    sde::GaussianStateCurve::Loadings loadings;
    if (factors == 0) {
        return loadings;
    }
    constexpr double kCoefficients[4] = {0.03, 0.012, 0.005, 0.002};
    loadings.factor.assign(factors, std::vector<double>(nObs * nPay, 0.0));
    loadings.correlation.assign(factors * factors, 0.0);
    for (std::size_t i = 0; i < factors; ++i) {
        for (std::size_t l = 0; l < factors; ++l) {
            loadings.correlation[i * factors + l] = (i == l) ? 1.0 : 0.35;
        }
    }
    const std::vector<double> obs = obsTimes(nObs);
    const std::vector<double> pay = payTimes(nPay);
    for (std::size_t k = 0; k < nObs; ++k) {
        for (std::size_t j = 0; j < nPay; ++j) {
            const double tau = pay[j] - obs[k];
            // tau == -1 sits exactly on the grid (pay - obs = -1) and makes the
            // rational weight singular; pin it to zero so the loading blocks
            // stay finite. Every other node keeps the original weight.
            const double denominator = 1.0 + tau;
            const double bounded = denominator == 0.0 ? 0.0 : tau / denominator;
            double weight = bounded;
            for (std::size_t i = 0; i < factors; ++i) {
                loadings.factor[i][k * nPay + j] = kCoefficients[i] * weight;
                weight *= bounded;
            }
        }
    }
    return loadings;
}

sde::GaussianStateCurve makeMap(const DiscountCurve<double>& curve, const ScheduleSize& size,
                                std::size_t factors) {
    return sde::GaussianStateCurve::fromCurve(curve, obsTimes(size.nObs), payTimes(size.nPay),
                                              makeLoadings(size.nObs, size.nPay, factors));
}

std::vector<double> makeState(std::size_t factors) {
    std::vector<double> state(factors);
    for (std::size_t i = 0; i < factors; ++i) {
        state[i] = 0.004 * static_cast<double>(i + 1) * ((i % 2 == 0) ? 1.0 : -1.0);
    }
    return state;
}

// ── Bermudan-shaped worst-case fixture ──────────────────────────────────────
//
// One engine block of states through a 30Y quarterly Bermudan schedule: 60
// exercise dates crossed with 60 cashflow dates, evaluated per state (the
// SoA state block of a path batch is stored state-major so every state is a
// contiguous `stateSize()` span). The per-state values are offset slightly so
// the evaluation cannot be hoisted across the block.
constexpr std::size_t kBermudanObs = 60;
constexpr std::size_t kBermudanPay = 60;
constexpr std::size_t kBermudanStates = 64;
constexpr std::size_t kBermudanAdStates = 16;

std::vector<double> makeStateBlock(std::size_t nStates, std::size_t factors) {
    std::vector<double> states(nStates * factors);
    for (std::size_t s = 0; s < nStates; ++s) {
        for (std::size_t i = 0; i < factors; ++i) {
            const double base = 0.004 * static_cast<double>(i + 1) * ((i % 2 == 0) ? 1.0 : -1.0);
            states[s * factors + i] = base + 0.0007 * static_cast<double>(s % 3);
        }
    }
    return states;
}

// ── measure-selection fixture ───────────────────────────────────────────────
//
// Deterministic stand-in for a canonical model (mirrors the unit-test model):
// every measure except collateral-currency is representable and a
// deterministic state carries no drift correction.
class MeasureSwitchModel {
public:
    bool supports(const sde::MeasureSpec& measure) const noexcept {
        return measure.kind != sde::MeasureKind::CollateralCurrency;
    }

    template <typename Scalar>
    Scalar measureAdjustment(const sde::MeasureSpec&, double, std::span<const Scalar>) const {
        return Scalar(0.0);
    }
};

// ── frozen-market fixture ───────────────────────────────────────────────────

struct FrozenFixture {
    sde::FrozenMarket market;
    CurveKey lastDiscount;
    CurveKey lastForecast;
    std::string lastCurrency;
    std::string ambiguousCurrency = "AMB";
    std::string fxPair = "CCY0CCY1";
    datetime::Period forecastTenor{3, datetime::TimeUnit::Months};
};

FrozenFixture makeFrozen(std::size_t nCurves) {
    const CurveHandle::Ptr ois = CurveHandle::make(makeCurve());
    const std::size_t nDiscount = (nCurves + 1) / 2;
    std::vector<sde::FrozenMarket::CurveEntry> entries;
    entries.reserve(nCurves + 2);
    FrozenFixture out;
    for (std::size_t i = 0; i < nDiscount; ++i) {
        CurveKey key;
        key.currency = "CCY" + std::to_string(i);
        key.role = CurveRole::Discount;
        key.collateral = key.currency;
        entries.push_back({key, CurveRole::Discount, ois});
        out.lastDiscount = key;
    }
    for (std::size_t i = nDiscount; i < nCurves; ++i) {
        CurveKey key;
        key.currency = "CCY" + std::to_string(i - nDiscount);
        key.role = CurveRole::Forecast;
        key.indexTenor = out.forecastTenor;
        key.collateral = key.currency;
        entries.push_back({key, CurveRole::Forecast, ois});
        out.lastForecast = key;
    }
    for (const char* collateral : {"A", "B"}) {
        CurveKey key;
        key.currency = out.ambiguousCurrency;
        key.role = CurveRole::Discount;
        key.collateral = collateral;
        entries.push_back({key, CurveRole::Discount, ois});
    }
    std::vector<sde::FrozenMarket::FxSpotView> fx;
    sde::FrozenMarket::FxSpotView view;
    view.pair = out.fxPair;
    view.spot = 1.08;
    view.base = ois;
    view.quote = ois;
    fx.push_back(std::move(view));
    out.market = sde::FrozenMarket(std::move(entries), std::move(fx));
    out.lastCurrency = out.lastDiscount.currency;
    return out;
}

// ── AD allocation-shape cases (alloc twin only) ─────────────────────────────

#if defined(QUANTAPE_MARKET_ALLOC_COUNTS)

template <typename Scalar>
double primal(const Scalar& value) {
    if constexpr (requires { value.val_.val(); }) {
        return value.val_.val();
    } else if constexpr (requires { value.val(); }) {
        return value.val();
    } else {
        return static_cast<double>(value);
    }
}

void benchAllocationShape(int reps) {
    using stan::math::fvar;
    using stan::math::var;

    const DiscountCurve<double> ois = makeCurve();
    for (const ScheduleSize& size : scheduleSizes()) {
        const sde::GaussianStateCurve map = makeMap(ois, size, 4);
        const std::string suffix = tag(size, 4);

        std::vector<var> varState(4);
        for (std::size_t i = 0; i < varState.size(); ++i) {
            varState[i] = 0.004 * static_cast<double>(i + 1);
        }
        const std::span<const var> varView(varState);
        std::vector<var> varOut(map.payCount());

        bench_us("alloc bond var " + suffix, double(size.nObs * size.nPay), reps, [&] {
            var sum(0.0);
            for (std::size_t k = 0; k < map.obsCount(); ++k) {
                for (std::size_t j = 0; j < map.payCount(); ++j) {
                    sum += map.bond(k, j, varView);
                }
            }
            return primal(sum);
        });
        const std::vector<double> accruals(map.payCount(), 0.25);
        bench_us("alloc annuity var " + suffix, double(map.payCount()), reps, [&] {
            const var sum = map.annuity(1, 0, varView, std::span<const double>(accruals));
            return primal(sum);
        });
        bench_us("alloc bondsInto var " + suffix, double(map.payCount()), reps, [&] {
            map.bondsInto(1, varView, std::span<var>(varOut));
            return primal(varOut[0] + varOut[map.payCount() - 1]);
        });

        std::vector<fvar<var>> fvarState(4);
        for (std::size_t i = 0; i < fvarState.size(); ++i) {
            fvarState[i] = fvar<var>(var(0.004 * static_cast<double>(i + 1)), 0.7);
        }
        const std::span<const fvar<var>> fvarView(fvarState);
        bench_us("alloc bond fvar " + suffix, double(size.nObs * size.nPay), reps, [&] {
            fvar<var> sum(0.0);
            for (std::size_t k = 0; k < map.obsCount(); ++k) {
                for (std::size_t j = 0; j < map.payCount(); ++j) {
                    sum += map.bond(k, j, fvarView);
                }
            }
            return primal(sum);
        });
    }

    // ── Bermudan-shaped AD tracks: 60 exercise x 60 cashflow x 16 states per
    //    block ──
    //
    // Per state: build a tape over the grid, pull the var adjoints once, then
    // move to the next state; the fvar<var> case propagates directional
    // tangents through the same schedule. State and output buffers are
    // allocated once outside the timed region, so the inner loops build no
    // per-date vectors and allocate nothing per block. Memory is recovered per
    // block, so the tape stays bounded across states and reps.
    {
        const ScheduleSize size{kBermudanObs, kBermudanPay};
        const sde::GaussianStateCurve map = makeMap(ois, size, 4);
        const std::string suffix = " " + tag(size, 4) + " b=" + std::to_string(kBermudanAdStates);
        const std::vector<double> plainStates = makeStateBlock(kBermudanAdStates, map.stateSize());
        const std::vector<double> accruals(map.payCount(), 0.25);
        const std::span<const double> accrualView(accruals);
        const sde::Numeraire<sde::GaussianStateCurve> numeraire(map);
        const std::span<const double> pay = map.payTimes();
        const double units =
            double(kBermudanAdStates) * double(kBermudanObs) * double(kBermudanPay);

        std::vector<var> varState(map.stateSize());
        const std::span<const var> varView(varState);
        std::vector<var> varOut(map.payCount());
        const std::span<var> varOutView(varOut);
        const auto loadVarState = [&](std::size_t s) {
            for (std::size_t i = 0; i < varState.size(); ++i) {
                varState[i] = plainStates[s * map.stateSize() + i];
            }
        };

        bench_us("alloc bermudan bond var" + suffix, units, reps, [&] {
            double sink = 0.0;
            for (std::size_t s = 0; s < kBermudanAdStates; ++s) {
                loadVarState(s);
                var total(0.0);
                for (std::size_t k = 0; k < map.obsCount(); ++k) {
                    for (std::size_t j = 0; j < map.payCount(); ++j) {
                        total += map.bond(k, j, varView);
                    }
                }
                total.grad();
                sink += primal(total) + varState.front().adj() + varState.back().adj();
                stan::math::recover_memory();
            }
            return sink;
        });
        bench_us("alloc bermudan bondsInto var" + suffix, units, reps, [&] {
            double sink = 0.0;
            for (std::size_t s = 0; s < kBermudanAdStates; ++s) {
                loadVarState(s);
                var total(0.0);
                for (std::size_t k = 0; k < map.obsCount(); ++k) {
                    map.bondsInto(k, varView, varOutView);
                    total += varOut.front() + varOut.back();
                }
                total.grad();
                sink += primal(total) + varState.front().adj();
                stan::math::recover_memory();
            }
            return sink;
        });
        bench_us("alloc bermudan annuity var" + suffix, units, reps, [&] {
            double sink = 0.0;
            for (std::size_t s = 0; s < kBermudanAdStates; ++s) {
                loadVarState(s);
                var total(0.0);
                for (std::size_t k = 0; k < map.obsCount(); ++k) {
                    total += map.annuity(k, 0, varView, accrualView);
                }
                total.grad();
                sink += primal(total) + varState.front().adj();
                stan::math::recover_memory();
            }
            return sink;
        });
        bench_us("alloc bermudan deflated var" + suffix, double(kBermudanAdStates * kBermudanPay),
                 reps, [&] {
                     double sink = 0.0;
                     const var payoff(125.0);
                     for (std::size_t s = 0; s < kBermudanAdStates; ++s) {
                         loadVarState(s);
                         var total(0.0);
                         for (const double t : pay) {
                             total += numeraire.deflated(payoff, t, varView);
                         }
                         total.grad();
                         sink += primal(total) + varState.front().adj();
                         stan::math::recover_memory();
                     }
                     return sink;
                 });

        std::vector<fvar<var>> fvarState(map.stateSize());
        const std::span<const fvar<var>> fvarView(fvarState);
        bench_us("alloc bermudan bond fvar" + suffix, units, reps, [&] {
            double sink = 0.0;
            for (std::size_t s = 0; s < kBermudanAdStates; ++s) {
                for (std::size_t i = 0; i < fvarState.size(); ++i) {
                    fvarState[i] = fvar<var>(var(plainStates[s * map.stateSize() + i]),
                                             0.7 - 0.1 * static_cast<double>(i));
                }
                fvar<var> total(0.0);
                for (std::size_t k = 0; k < map.obsCount(); ++k) {
                    for (std::size_t j = 0; j < map.payCount(); ++j) {
                        total += map.bond(k, j, fvarView);
                    }
                }
                sink += primal(total) + total.d_.val();
                stan::math::recover_memory();
            }
            return sink;
        });
    }
}

#endif // QUANTAPE_MARKET_ALLOC_COUNTS

} // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    if (argc > 2) {
        g_filter = argv[2];
    }
    QTA_LOG_INFO("bench", "SDE market benchmark, reps={}, filter='{}'", reps, g_filter);

    const DiscountCurve<double> ois = makeCurve();
    const std::vector<double> discountDates = payTimes(std::size_t(256));

    // ── 1. state-to-curve evaluation across schedule tensors ──
    for (const ScheduleSize& size : scheduleSizes()) {
        const sde::GaussianStateCurve map = makeMap(ois, size, 4);
        const std::string suffix = tag(size, 4);
        const std::vector<double> state = makeState(4);
        const std::span<const double> stateView(state);
        const std::vector<double> accruals(map.payCount(), 0.25);
        const std::span<const double> accrualView(accruals);
        std::vector<double> out(map.payCount(), 0.0);
        const std::span<double> outView(out);
        const std::size_t midObs = size.nObs / 2;

        bench_us("bond grid " + suffix, double(size.nObs * size.nPay), reps, [&] {
            double sum = 0.0;
            for (std::size_t k = 0; k < map.obsCount(); ++k) {
                for (std::size_t j = 0; j < map.payCount(); ++j) {
                    sum += map.bond(k, j, stateView);
                }
            }
            return sum;
        });
        bench_us("annuity " + suffix, double(map.payCount()), reps,
                 [&] { return map.annuity(midObs, 0, stateView, accrualView); });
        bench_us("bondsInto " + suffix, double(map.payCount()), reps, [&] {
            map.bondsInto(midObs, stateView, outView);
            return out.front() + out.back();
        });

        const sde::Numeraire<sde::GaussianStateCurve> numeraire(map);
        bench_us("numeraire value " + suffix, double(map.payCount()), reps, [&] {
            double sum = 0.0;
            for (const double t : map.payTimes()) {
                sum += numeraire.numeraireValue(t, stateView);
            }
            return sum;
        });
        bench_us("numeraire deflated " + suffix, double(map.payCount()), reps, [&] {
            double sum = 0.0;
            for (const double t : map.payTimes()) {
                sum += numeraire.deflated(125.0, t, stateView);
            }
            return sum;
        });
        bench_us("numeraire rollRatio " + suffix, double(map.payCount() - 1), reps, [&] {
            double sum = 0.0;
            const std::span<const double> pay = map.payTimes();
            for (std::size_t j = 0; j + 1 < pay.size(); ++j) {
                sum += numeraire.rollRatio(pay[j], pay[j + 1], stateView);
            }
            return sum;
        });
    }

    // ── 1b. Bermudan-shaped worst case: one block of states through the
    //        exercise x cashflow schedule ──
    //
    // 60 exercise dates x 60 cashflow dates x 64 states (30Y quarterly block).
    // The state block is state-major, so each state is a contiguous
    // stateSize() span; every case reuses caller-owned output/accrual buffers,
    // so the timed region allocates nothing on the double track.
    {
        const ScheduleSize size{kBermudanObs, kBermudanPay};
        const sde::GaussianStateCurve map = makeMap(ois, size, 4);
        const std::string suffix = " " + tag(size, 4) + " b=" + std::to_string(kBermudanStates);
        const std::vector<double> states = makeStateBlock(kBermudanStates, map.stateSize());
        const std::vector<double> accruals(map.payCount(), 0.25);
        const std::span<const double> accrualView(accruals);
        std::vector<double> out(map.payCount(), 0.0);
        const std::span<double> outView(out);
        const sde::Numeraire<sde::GaussianStateCurve> numeraire(map);
        const std::span<const double> pay = map.payTimes();
        const double units = double(kBermudanStates) * double(kBermudanObs) * double(kBermudanPay);

        bench_us("bermudan bond grid" + suffix, units, reps, [&] {
            double sum = 0.0;
            for (std::size_t s = 0; s < kBermudanStates; ++s) {
                const std::span<const double> state(states.data() + s * map.stateSize(),
                                                    map.stateSize());
                for (std::size_t k = 0; k < map.obsCount(); ++k) {
                    for (std::size_t j = 0; j < map.payCount(); ++j) {
                        sum += map.bond(k, j, state);
                    }
                }
            }
            return sum;
        });
        bench_us("bermudan bondsInto" + suffix, units, reps, [&] {
            double sum = 0.0;
            for (std::size_t s = 0; s < kBermudanStates; ++s) {
                const std::span<const double> state(states.data() + s * map.stateSize(),
                                                    map.stateSize());
                for (std::size_t k = 0; k < map.obsCount(); ++k) {
                    map.bondsInto(k, state, outView);
                    sum += outView.front() + outView.back();
                }
            }
            return sum;
        });
        bench_us("bermudan annuity" + suffix, units, reps, [&] {
            double sum = 0.0;
            for (std::size_t s = 0; s < kBermudanStates; ++s) {
                const std::span<const double> state(states.data() + s * map.stateSize(),
                                                    map.stateSize());
                for (std::size_t k = 0; k < map.obsCount(); ++k) {
                    sum += map.annuity(k, 0, state, accrualView);
                }
            }
            return sum;
        });
        bench_us("bermudan numeraire deflated" + suffix, double(kBermudanStates * kBermudanPay),
                 reps, [&] {
                     double sum = 0.0;
                     for (std::size_t s = 0; s < kBermudanStates; ++s) {
                         const std::span<const double> state(states.data() + s * map.stateSize(),
                                                             map.stateSize());
                         for (const double t : pay) {
                             sum += numeraire.deflated(125.0, t, state);
                         }
                     }
                     return sum;
                 });
        // Rolling numeraire: one direct value at the back of the schedule, then
        // a shared ratio product N(T_j) = N(T_{j+1}) * N(T_j)/N(T_{j+1})
        // backward across the cashflow dates. This re-associates the direct
        // deflations (a documented accumulation-order change) and is the
        // LSMC-style pattern; it is measured, not gated.
        bench_us("bermudan numeraire rolling" + suffix, double(kBermudanStates * kBermudanPay),
                 reps, [&] {
                     double sum = 0.0;
                     for (std::size_t s = 0; s < kBermudanStates; ++s) {
                         const std::span<const double> state(states.data() + s * map.stateSize(),
                                                             map.stateSize());
                         double level = numeraire.numeraireValue(pay.back(), state);
                         for (std::size_t j = pay.size() - 1; j-- > 0;) {
                             level *= numeraire.rollRatio(pay[j], pay[j + 1], state);
                             sum += 125.0 / level;
                         }
                     }
                     return sum;
                 });
    }

    // ── 1c. construction: loadings -> record precompute (base ratio, Omega,
    //        loading copy) for increasing date tensors ──
    //
    // The loading blocks are built once and moved in, so the timed region is
    // the constructor's own precompute (Omega is O(nObs*nPay*q^2)) plus the
    // record fill; the per-case pool is sized for the warmup and all reps.
    {
        const std::vector<ScheduleSize> constructSizes = {{4, 16}, {16, 64}, {64, 256}, {128, 512}};
        for (const ScheduleSize& size : constructSizes) {
            const std::size_t records = size.nObs * size.nPay;
            const int constructReps = records > 32768 ? std::min(reps, 8) : reps;
            std::vector<sde::GaussianStateCurve::Loadings> pool;
            pool.reserve(static_cast<std::size_t>(constructReps) + 1);
            for (int r = 0; r <= constructReps; ++r) {
                pool.push_back(makeLoadings(size.nObs, size.nPay, 4));
            }
            std::vector<double> baseObs;
            std::vector<double> basePay;
            const std::vector<double> obs = obsTimes(size.nObs);
            const std::vector<double> pay = payTimes(size.nPay);
            baseObs.reserve(obs.size());
            basePay.reserve(pay.size());
            for (const double t : obs) {
                baseObs.push_back(ois.discount(t));
            }
            for (const double t : pay) {
                basePay.push_back(ois.discount(t));
            }
            std::size_t next = 0;
            bench_us("construct " + tag(size, 4), double(records), constructReps, [&] {
                const sde::GaussianStateCurve map(obs, pay, baseObs, basePay,
                                                  std::move(pool[next++]));
                return map.baseRatio(0, 0) + map.omega(0, 0);
            });
        }
        const ScheduleSize fromCurveSize{64, 256};
        const std::size_t records = fromCurveSize.nObs * fromCurveSize.nPay;
        std::vector<sde::GaussianStateCurve::Loadings> pool;
        pool.reserve(static_cast<std::size_t>(reps) + 1);
        for (int r = 0; r <= reps; ++r) {
            pool.push_back(makeLoadings(fromCurveSize.nObs, fromCurveSize.nPay, 4));
        }
        std::size_t next = 0;
        bench_us("construct fromCurve " + tag(fromCurveSize, 4), double(records), reps, [&] {
            const sde::GaussianStateCurve map = sde::GaussianStateCurve::fromCurve(
                ois, obsTimes(fromCurveSize.nObs), payTimes(fromCurveSize.nPay),
                std::move(pool[next++]));
            return map.baseRatio(0, 0) + map.omega(0, 0);
        });
    }

    // Deterministic (zero-factor) limit: the bond is the base ratio exactly.
    for (const ScheduleSize& size : scheduleSizes()) {
        const sde::GaussianStateCurve map = makeMap(ois, size, 0);
        const std::string suffix = tag(size, 0);
        const std::span<const double> zero;
        std::vector<double> out(map.payCount(), 0.0);
        bench_us("bond det " + suffix, double(size.nObs * size.nPay), reps, [&] {
            double sum = 0.0;
            for (std::size_t k = 0; k < map.obsCount(); ++k) {
                for (std::size_t j = 0; j < map.payCount(); ++j) {
                    sum += map.bond(k, j, zero);
                }
            }
            return sum;
        });
        bench_us("bondsInto det " + suffix, double(map.payCount()), reps, [&] {
            map.bondsInto(0, zero, std::span<double>(out));
            return out.front() + out.back();
        });
    }

    // ── 2. numeraire over a deterministic provider ──
    {
        const sde::Numeraire<DiscountCurve<double>> numeraire(ois);
        const std::span<const double> zero;
        bench_us("numeraire value discount", double(discountDates.size()), reps, [&] {
            double sum = 0.0;
            for (const double t : discountDates) {
                sum += numeraire.numeraireValue(t, zero);
            }
            return sum;
        });
        bench_us("numeraire deflated discount", double(discountDates.size()), reps, [&] {
            double sum = 0.0;
            for (const double t : discountDates) {
                sum += numeraire.deflated(125.0, t, zero);
            }
            return sum;
        });
        bench_us("numeraire rollRatio discount", double(discountDates.size() - 1), reps, [&] {
            double sum = 0.0;
            for (std::size_t j = 0; j + 1 < discountDates.size(); ++j) {
                sum += numeraire.rollRatio(discountDates[j], discountDates[j + 1], zero);
            }
            return sum;
        });
    }

    // ── 2b. measure selection: spec copies, support gate and adjustment ──
    {
        constexpr std::size_t kIterations = 20000;
        const MeasureSwitchModel model;
        const std::span<const double> zero;
        sde::MeasureSpec annuity;
        annuity.kind = sde::MeasureKind::Annuity;
        annuity.annuityWeights.assign(64, 0.25);
        bench_us("measure spec copy 64", double(kIterations), reps, [&] {
            double sum = 0.0;
            for (std::size_t i = 0; i < kIterations; ++i) {
                const sde::MeasureSpec copy = annuity;
                sum += static_cast<double>(copy.annuityWeights.size()) + copy.forwardTime;
            }
            return sum;
        });
        bench_us("measure support", double(kIterations), reps, [&] {
            double sum = 0.0;
            for (std::size_t i = 0; i < kIterations; ++i) {
                sde::requireMeasureSupport(model, annuity);
                sum += model.supports(annuity) ? 1.0 : 0.0;
            }
            return sum;
        });
        bench_us("measure adjustment", double(kIterations), reps, [&] {
            double sum = 0.0;
            for (std::size_t i = 0; i < kIterations; ++i) {
                sum += model.measureAdjustment(annuity, 1.5, zero);
            }
            return sum;
        });
    }

    // ── 3. frozen-market lookups and freeze ──
    {
        constexpr std::size_t kIterations = 20000;
        const FrozenFixture frozen16 = makeFrozen(16);
        const FrozenFixture frozen64 = makeFrozen(64);
        const FrozenFixture frozen256 = makeFrozen(256);
        bench_us("frozen construct 64", 64.0, reps, [&] {
            const FrozenFixture fixture = makeFrozen(64);
            return static_cast<double>(fixture.market.size());
        });
        for (const FrozenFixture* fixturePtr : {&frozen16, &frozen64, &frozen256}) {
            const FrozenFixture& fixture = *fixturePtr;
            const std::size_t n = fixture.market.size();
            const std::string suffix = " n=" + std::to_string(n);
            bench_us("frozen curve lookup" + suffix, double(kIterations), reps, [&] {
                double sum = 0.0;
                for (std::size_t i = 0; i < kIterations; ++i) {
                    const CurveHandle* handle = fixture.market.curve(fixture.lastDiscount);
                    sum += static_cast<double>(reinterpret_cast<std::uintptr_t>(handle));
                }
                return sum;
            });
            bench_us("frozen discount lookup" + suffix, double(kIterations), reps, [&] {
                double sum = 0.0;
                for (std::size_t i = 0; i < kIterations; ++i) {
                    const CurveHandle* handle = fixture.market.discountCurve(fixture.lastCurrency);
                    sum += static_cast<double>(reinterpret_cast<std::uintptr_t>(handle));
                }
                return sum;
            });
            bench_us("frozen discount2 lookup" + suffix, double(kIterations), reps, [&] {
                double sum = 0.0;
                for (std::size_t i = 0; i < kIterations; ++i) {
                    const CurveHandle* handle =
                        fixture.market.discountCurve(fixture.lastCurrency, fixture.lastCurrency);
                    sum += static_cast<double>(reinterpret_cast<std::uintptr_t>(handle));
                }
                return sum;
            });
            bench_us("frozen forecast lookup" + suffix, double(kIterations), reps, [&] {
                double sum = 0.0;
                for (std::size_t i = 0; i < kIterations; ++i) {
                    const CurveHandle* handle =
                        fixture.market.forecastCurve(fixture.lastCurrency, fixture.forecastTenor);
                    sum += static_cast<double>(reinterpret_cast<std::uintptr_t>(handle));
                }
                return sum;
            });
            bench_us("frozen fxSpot lookup" + suffix, double(kIterations), reps, [&] {
                double sum = 0.0;
                for (std::size_t i = 0; i < kIterations; ++i) {
                    const sde::FrozenMarket::FxSpotView* view =
                        fixture.market.fxSpot(fixture.fxPair);
                    sum += static_cast<double>(reinterpret_cast<std::uintptr_t>(view));
                }
                return sum;
            });
            bench_us("frozen ambiguous discount" + suffix, double(kIterations), reps, [&] {
                double sum = 0.0;
                for (std::size_t i = 0; i < kIterations; ++i) {
                    const CurveHandle* handle =
                        fixture.market.discountCurve(fixture.ambiguousCurrency);
                    sum += handle == nullptr ? 1.0 : 0.0;
                }
                return sum;
            });
        }
        for (const std::size_t nDates :
             {std::size_t(8), std::size_t(64), std::size_t(256), std::size_t(1024)}) {
            const std::vector<double> times = payTimes(nDates);
            const std::string suffix = " " + std::to_string(nDates);
            bench_us("freeze" + suffix, double(nDates), reps, [&] {
                const markets::CurveOnGrid<double> grid =
                    frozen64.market.freeze(frozen64.lastDiscount, times);
                return grid.discount.front() + grid.forward.back();
            });
        }
    }

#if defined(QUANTAPE_MARKET_ALLOC_COUNTS)
    benchAllocationShape(reps);
#endif

    return 0;
}
