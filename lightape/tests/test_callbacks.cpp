#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "lightape/ops/Callbacks.h"
#include "lightape/ops/DerivativeCache.h"
#include <gtest/gtest.h>

namespace {

using lightape::DerivativeCache;
using lightape::ExtArg;
using lightape::RevScalar;
using lightape::RevTape;

class CallbacksTest : public ::testing::Test {
protected:
    void SetUp() override { RevTape::active().clear(); }
    void TearDown() override { RevTape::active().clear(); }
};

TEST_F(CallbacksTest, MakeVjpVarValueAndGradients) {
    RevTape& tape = RevTape::active();
    const double xv = 1.25;
    const double yv = -0.4;
    RevScalar x(xv, tape.input(xv));
    RevScalar y(yv, tape.input(yv));
    const RevScalar args[2] = {x, y};
    auto vjp = [](const void* /*payload*/, double seed, const ExtArg* a, std::uint32_t /*n*/,
                  RevTape& t) noexcept {
        t.addAdjoint(a[0].node, seed * a[1].value);
        t.addAdjoint(a[1].node, seed * a[0].value);
    };

    const RevScalar f = lightape::makeVjpVar(xv * yv, args, 2, vjp);
    EXPECT_DOUBLE_EQ(f.value, xv * yv);
    EXPECT_EQ(tape.size(), 3u);

    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), yv);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), xv);
}

TEST_F(CallbacksTest, MakeVjpVarSingleScalarOverloadAndCollapse) {
    RevTape& tape = RevTape::active();
    const double xv = 0.75;
    RevScalar x(xv, tape.input(xv));
    auto vjp = [](const void* /*payload*/, double seed, const ExtArg* a, std::uint32_t /*n*/,
                  RevTape& t) noexcept { t.addAdjoint(a[0].node, seed * 2.0 * a[0].value); };

    const RevScalar f = lightape::makeVjpVar(xv * xv, x, vjp);
    EXPECT_DOUBLE_EQ(f.value, xv * xv);
    EXPECT_EQ(tape.size(), 2u);
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 2.0 * xv);

    const RevScalar folded = lightape::makeVjpVar(4.0, RevScalar(2.0), vjp);
    EXPECT_EQ(folded.node, RevTape::kNone);
    EXPECT_DOUBLE_EQ(folded.value, 4.0);
    EXPECT_EQ(tape.size(), 2u);
}

TEST_F(CallbacksTest, MakeVjpVarCollapsesConstants) {
    auto vjp = [](const void* /*payload*/, double seed, const ExtArg* a, std::uint32_t n,
                  RevTape& t) noexcept {
        for (std::uint32_t i = 0; i < n; ++i) {
            t.addAdjoint(a[i].node, seed);
        }
    };
    const RevScalar args[2] = {RevScalar(2.0), RevScalar(3.0)};
    const RevScalar f = lightape::makeVjpVar(6.0, args, 2, vjp);
    EXPECT_EQ(f.node, RevTape::kNone);
    EXPECT_DOUBLE_EQ(f.value, 6.0);
    EXPECT_EQ(RevTape::active().size(), 0u);
}

struct AffinePayload {
    double a = 0.0;
    double b = 0.0;
};

TEST_F(CallbacksTest, MakeVjpVarPayloadCopyOutlivesSource) {
    RevTape& tape = RevTape::active();
    const double xv = 2.0;
    const double yv = -3.0;
    RevScalar x(xv, tape.input(xv));
    RevScalar y(yv, tape.input(yv));
    const RevScalar f = [&] {
        auto vjp = [](const void* p, double seed, const ExtArg* a, std::uint32_t /*n*/,
                      RevTape& t) noexcept {
            const auto* payload = static_cast<const AffinePayload*>(p);
            t.addAdjoint(a[0].node, seed * payload->a);
            t.addAdjoint(a[1].node, seed * payload->b);
        };
        const AffinePayload payload{4.0, -1.5};
        const RevScalar args[2] = {x, y};
        return lightape::makeVjpVar(payload.a * xv + payload.b * yv, args, 2, vjp, payload);
    }();

    EXPECT_DOUBLE_EQ(f.value, 4.0 * xv - 1.5 * yv);
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), 4.0);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), -1.5);
}

TEST_F(CallbacksTest, MakeVjpVarHessianFailsLoudly) {
    RevTape& tape = RevTape::active();
    RevScalar x(1.0, tape.input(1.0));
    RevScalar y(2.0, tape.input(2.0));
    const RevScalar args[2] = {x, y};
    auto vjp = [](const void* /*payload*/, double seed, const ExtArg* a, std::uint32_t n,
                  RevTape& t) noexcept {
        for (std::uint32_t i = 0; i < n; ++i) {
            t.addAdjoint(a[i].node, seed);
        }
    };
    const RevScalar f = lightape::makeVjpVar(3.0, args, 2, vjp);

    const std::uint32_t inputs[2] = {x.node, y.node};
    const double v[2] = {1.0, 1.0};
    double out[2] = {-99.0, -99.0};
    EXPECT_FALSE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_DOUBLE_EQ(out[0], -99.0);
    EXPECT_DOUBLE_EQ(out[1], -99.0);
}

TEST_F(CallbacksTest, MakeVhpFvarFmaLikeBlock) {
    RevTape& tape = RevTape::active();
    const double xv = 0.4;
    const double yv = 1.7;
    const double zv = 0.6;
    RevScalar x(xv, tape.input(xv));
    RevScalar y(yv, tape.input(yv));
    RevScalar z(zv, tape.input(zv));
    const RevScalar args[3] = {x, y, z};

    auto vjp = [](const void* /*payload*/, double seed, const ExtArg* a, std::uint32_t /*n*/,
                  RevTape& t) noexcept {
        t.addAdjoint(a[0].node, seed * a[1].value);
        t.addAdjoint(a[1].node, seed * a[0].value);
        t.addAdjoint(a[2].node, seed);
    };
    auto svp = [](const void* /*payload*/, const ExtArg* a, const double* s,
                  std::uint32_t /*n*/) noexcept {
        return a[1].value * s[0] + a[0].value * s[1] + s[2];
    };
    auto vhp = [](const void* /*payload*/, double adjoint, const ExtArg* /*a*/, const double* s,
                  std::uint32_t /*n*/, double* b) noexcept {
        b[0] += adjoint * s[1];
        b[1] += adjoint * s[0];
    };

    const RevScalar f = lightape::makeVhpFvar(std::fma(xv, yv, zv), args, 3, vjp, svp, vhp);
    EXPECT_DOUBLE_EQ(f.value, std::fma(xv, yv, zv));
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), yv);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), xv);
    EXPECT_DOUBLE_EQ(tape.adjoint(z.node), 1.0);

    const std::uint32_t inputs[3] = {x.node, y.node, z.node};
    const double v[3] = {0.5, 0.25, -0.3};
    double out[3] = {0.0, 0.0, 0.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 3, v, out));
    EXPECT_DOUBLE_EQ(out[0], v[1]);
    EXPECT_DOUBLE_EQ(out[1], v[0]);
    EXPECT_DOUBLE_EQ(out[2], 0.0);
}

TEST_F(CallbacksTest, MakeVhpFvarSingleScalarExp) {
    RevTape& tape = RevTape::active();
    const double xv = 0.7;
    RevScalar x(xv, tape.input(xv));

    auto vjp = [](const void* /*payload*/, double seed, const ExtArg* a, std::uint32_t /*n*/,
                  RevTape& t) noexcept { t.addAdjoint(a[0].node, seed * std::exp(a[0].value)); };
    auto svp = [](const void* /*payload*/, const ExtArg* a, const double* s,
                  std::uint32_t /*n*/) noexcept { return std::exp(a[0].value) * s[0]; };
    auto vhp = [](const void* /*payload*/, double adjoint, const ExtArg* a, const double* s,
                  std::uint32_t /*n*/,
                  double* b) noexcept { b[0] += adjoint * std::exp(a[0].value) * s[0]; };

    const RevScalar f = lightape::makeVhpFvar(std::exp(xv), x, vjp, svp, vhp);
    EXPECT_DOUBLE_EQ(f.value, std::exp(xv));
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), std::exp(xv));

    const std::uint32_t inputs[1] = {x.node};
    const double v[1] = {1.3};
    double out[1] = {0.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 1, v, out));
    EXPECT_NEAR(out[0], std::exp(xv) * v[0], 1e-15);
}

TEST_F(CallbacksTest, MakeVhpFvarPayloadScale) {
    RevTape& tape = RevTape::active();
    const double av = 0.8;
    const double bv = -1.25;
    RevScalar a(av, tape.input(av));
    RevScalar b(bv, tape.input(bv));
    const RevScalar f = [&] {
        struct ScalePayload {
            double s = 0.0;
        };
        auto vjp = [](const void* p, double seed, const ExtArg* x, std::uint32_t /*n*/,
                      RevTape& t) noexcept {
            const auto* payload = static_cast<const ScalePayload*>(p);
            t.addAdjoint(x[0].node, seed * payload->s * x[1].value);
            t.addAdjoint(x[1].node, seed * payload->s * x[0].value);
        };
        auto svp = [](const void* p, const ExtArg* x, const double* s,
                      std::uint32_t /*n*/) noexcept {
            const auto* payload = static_cast<const ScalePayload*>(p);
            return payload->s * (x[1].value * s[0] + x[0].value * s[1]);
        };
        auto vhp = [](const void* p, double adjoint, const ExtArg* /*x*/, const double* s,
                      std::uint32_t /*n*/, double* out) noexcept {
            const auto* payload = static_cast<const ScalePayload*>(p);
            out[0] += adjoint * payload->s * s[1];
            out[1] += adjoint * payload->s * s[0];
        };
        const ScalePayload payload{2.5};
        const RevScalar args[2] = {a, b};
        return lightape::makeVhpFvar(2.5 * av * bv, args, 2, vjp, svp, vhp, payload);
    }();

    EXPECT_DOUBLE_EQ(f.value, 2.5 * av * bv);
    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(a.node), 2.5 * bv);
    EXPECT_DOUBLE_EQ(tape.adjoint(b.node), 2.5 * av);

    const std::uint32_t inputs[2] = {a.node, b.node};
    const double v[2] = {0.6, -0.4};
    double out[2] = {0.0, 0.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    EXPECT_DOUBLE_EQ(out[0], 2.5 * v[1]);
    EXPECT_DOUBLE_EQ(out[1], 2.5 * v[0]);
}

TEST_F(CallbacksTest, DerivativeCacheMissHitEpochAndClear) {
    DerivativeCache<int, double, 4> cache;
    EXPECT_EQ(cache.find(3), nullptr);
    EXPECT_EQ(cache.size(), 0u);

    cache.insert(3, 1.5);
    const double* hit = cache.find(3);
    ASSERT_NE(hit, nullptr);
    EXPECT_DOUBLE_EQ(*hit, 1.5);
    EXPECT_EQ(cache.find(4), nullptr);
    EXPECT_EQ(cache.size(), 1u);

    const std::uint64_t e0 = cache.epoch();
    cache.bumpEpoch();
    EXPECT_EQ(cache.epoch(), e0 + 1u);
    EXPECT_EQ(cache.find(3), nullptr);
    EXPECT_EQ(cache.size(), 0u);

    cache.insert(3, 2.5);
    hit = cache.find(3);
    ASSERT_NE(hit, nullptr);
    EXPECT_DOUBLE_EQ(*hit, 2.5);

    cache.clear();
    EXPECT_EQ(cache.find(3), nullptr);
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_EQ(cache.epoch(), e0 + 1u);
}

TEST_F(CallbacksTest, DerivativeCacheRoundRobinEviction) {
    DerivativeCache<int, int, 3> cache;
    for (int i = 0; i < 3; ++i) {
        cache.insert(i, 10 * i);
    }
    EXPECT_EQ(cache.size(), 3u);

    cache.insert(3, 30); // evicts key 0
    EXPECT_EQ(cache.size(), 3u);
    EXPECT_EQ(cache.find(0), nullptr);
    for (int i = 1; i <= 3; ++i) {
        const int* hit = cache.find(i);
        ASSERT_NE(hit, nullptr) << "i=" << i;
        EXPECT_EQ(*hit, 10 * i);
    }

    cache.insert(4, 40); // evicts key 1
    EXPECT_EQ(cache.find(1), nullptr);
    const int* hit = cache.find(4);
    ASSERT_NE(hit, nullptr);
    EXPECT_EQ(*hit, 40);
}

TEST_F(CallbacksTest, DerivativeCacheRefreshesDuplicateKey) {
    DerivativeCache<int, double, 2> cache;
    cache.insert(5, 1.0);
    cache.insert(5, 2.0);
    EXPECT_EQ(cache.size(), 1u);
    const double* hit = cache.find(5);
    ASSERT_NE(hit, nullptr);
    EXPECT_DOUBLE_EQ(*hit, 2.0);
}

TEST_F(CallbacksTest, DerivativeCacheThreadLocalAccessor) {
    // Single-threaded semantics check: repeated calls hand back the same
    // instance. True cross-thread isolation is a static property of the
    // `static thread_local` backing store and is deliberately not asserted by
    // spawning threads in this suite.
    DerivativeCache<int, double, 4>& first = DerivativeCache<int, double, 4>::threadLocal();
    DerivativeCache<int, double, 4>& second = DerivativeCache<int, double, 4>::threadLocal();
    EXPECT_EQ(&first, &second);

    first.clear();
    first.insert(7, 2.5);
    const double* hit = second.find(7);
    ASSERT_NE(hit, nullptr);
    EXPECT_DOUBLE_EQ(*hit, 2.5);
    first.clear();
}

struct PointKey {
    double x = 0.0;
    double y = 0.0;
    bool operator==(const PointKey&) const = default;
};

struct BlockDerivs {
    double gx = 0.0;
    double gy = 0.0;
    double hxx = 0.0;
    double hxy = 0.0;
    double hyy = 0.0;
};

struct CachePayload {
    const DerivativeCache<PointKey, BlockDerivs, 4>* cache = nullptr;
    PointKey key{};
    bool* observed = nullptr;
};

TEST_F(CallbacksTest, CacheBackedCallbacksReadValuesOnly) {
    RevTape& tape = RevTape::active();
    const double xv = -0.5;
    const double yv = 1.25;
    RevScalar x(xv, tape.input(xv));
    RevScalar y(yv, tape.input(yv));

    // Block: z = log(exp(x) + exp(y)) at (xv, yv); cached analytic derivatives.
    const double m = std::max(xv, yv);
    const double ex = std::exp(xv - m);
    const double ey = std::exp(yv - m);
    const double z = ex + ey;
    const double ox = ex / z;
    const double oy = ey / z;
    DerivativeCache<PointKey, BlockDerivs, 4> cache;
    cache.insert(PointKey{xv, yv}, BlockDerivs{ox, oy, ox * (1.0 - ox), -ox * oy, oy * (1.0 - oy)});

    bool observed = false;
    const CachePayload payload{&cache, PointKey{xv, yv}, &observed};

    auto vjp = [](const void* p, double seed, const ExtArg* a, std::uint32_t /*n*/,
                  RevTape& t) noexcept {
        const auto* cp = static_cast<const CachePayload*>(p);
        const BlockDerivs* d = cp->cache->find(cp->key);
        if (d == nullptr) {
            return;
        }
        *cp->observed = true;
        t.addAdjoint(a[0].node, seed * d->gx);
        t.addAdjoint(a[1].node, seed * d->gy);
    };
    auto svp = [](const void* p, const ExtArg* /*a*/, const double* s,
                  std::uint32_t /*n*/) noexcept {
        const auto* cp = static_cast<const CachePayload*>(p);
        const BlockDerivs* d = cp->cache->find(cp->key);
        if (d == nullptr) {
            return 0.0;
        }
        *cp->observed = true;
        return d->gx * s[0] + d->gy * s[1];
    };
    auto vhp = [](const void* p, double adjoint, const ExtArg* /*a*/, const double* s,
                  std::uint32_t /*n*/, double* b) noexcept {
        const auto* cp = static_cast<const CachePayload*>(p);
        const BlockDerivs* d = cp->cache->find(cp->key);
        if (d == nullptr) {
            return;
        }
        *cp->observed = true;
        b[0] += adjoint * (d->hxx * s[0] + d->hxy * s[1]);
        b[1] += adjoint * (d->hxy * s[0] + d->hyy * s[1]);
    };

    const RevScalar args[2] = {x, y};
    const RevScalar f = lightape::makeVhpFvar(m + std::log(z), args, 2, vjp, svp, vhp, payload);

    tape.reverse(f.node);
    EXPECT_DOUBLE_EQ(tape.adjoint(x.node), ox);
    EXPECT_DOUBLE_EQ(tape.adjoint(y.node), oy);
    EXPECT_TRUE(observed);

    const std::uint32_t inputs[2] = {x.node, y.node};
    const double v[2] = {0.3, -0.8};
    double out[2] = {0.0, 0.0};
    ASSERT_TRUE(tape.hessianVector(f.node, inputs, 2, v, out));
    const double dot = ox * v[0] + oy * v[1];
    EXPECT_NEAR(out[0], ox * (v[0] - dot), 1e-15);
    EXPECT_NEAR(out[1], oy * (v[1] - dot), 1e-15);
}

} // namespace
