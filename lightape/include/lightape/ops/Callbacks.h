#ifndef LIGHTAPE_OPS_CALLBACKS_H
#define LIGHTAPE_OPS_CALLBACKS_H

#include <cstdint>
#include <type_traits>

#include "lightape/rev/LeanReverse.h"

namespace lightape {
/**
 * @file Callbacks.h
 * @brief Ergonomic factories for `External` tape blocks (quantape-style callbacks)
 *
 * The raw ABI in LeanReverse.h (`ExtVjpFn`/`ExtSvpFn`/`ExtVhpFn`) is a set of
 * `noexcept` function pointers taking a `const void*` payload; recording a
 * block additionally needs the primal value and the cached parent scalars.
 * These factories wrap a single `detail::makeExternal` call:
 *
 *   RevScalar z = makeVjpVar(value, args, nArgs, vjp);
 *   RevScalar z = makeVhpFvar(value, args, nArgs, vjp, svp, vhp);
 *
 * Mapping onto the tape's differentiability contract:
 * - `makeVjpVar` records **first order only** (no `svp`/`vhp`): a Hessian
 *   replay stops at this block and `hessianVector` returns false — fail loud,
 *   never approximate.
 * - `makeVhpFvar` records **full first + second order** (`vjp` + `svp` +
 *   `vhp`), so `hessianVector` replays the block exactly.
 *
 * `VjpFn`/`SvpFn`/`VhpFn` must be convertible to the matching ABI pointer:
 * captureless lambdas whose call operator is `noexcept`, or plain `noexcept`
 * function pointers. A throwing lambda does not convert to a `noexcept`
 * function pointer and trips the static_assert:
 *
 *   auto vjp = [](const void*, double seed, const ExtArg* a, std::uint32_t n,
 *                 RevTape& t) noexcept { ... };
 *
 * The `payload` overloads copy a trivially copyable payload into the tape at
 * record time, so the callback observes the tape's copy and the source object
 * may go out of scope before the sweep.
 */

/// First-order-only External block over `nArgs` parents (`args` = value + slot).
template <typename VjpFn>
RevScalar makeVjpVar(double value, const RevScalar* args, std::uint32_t nArgs, VjpFn vjp) {
    static_assert(std::is_convertible_v<VjpFn, ExtVjpFn>,
                  "makeVjpVar: VjpFn must be convertible to ExtVjpFn (a captureless lambda "
                  "marked noexcept, or a noexcept function pointer)");
    return detail::makeExternal(value, args, nArgs, static_cast<ExtVjpFn>(vjp));
}

/// First-order-only External block with a payload copied into the tape.
template <typename VjpFn, typename P>
RevScalar makeVjpVar(double value, const RevScalar* args, std::uint32_t nArgs, VjpFn vjp,
                     const P& payload) {
    static_assert(std::is_convertible_v<VjpFn, ExtVjpFn>,
                  "makeVjpVar: VjpFn must be convertible to ExtVjpFn (a captureless lambda "
                  "marked noexcept, or a noexcept function pointer)");
    return detail::makeExternal(value, args, nArgs, static_cast<ExtVjpFn>(vjp), payload);
}

/// Full first+second order External block over `nArgs` parents.
template <typename VjpFn, typename SvpFn, typename VhpFn>
RevScalar makeVhpFvar(double value, const RevScalar* args, std::uint32_t nArgs, VjpFn vjp,
                      SvpFn svp, VhpFn vhp) {
    static_assert(std::is_convertible_v<VjpFn, ExtVjpFn>,
                  "makeVhpFvar: VjpFn must be convertible to ExtVjpFn (a captureless lambda "
                  "marked noexcept, or a noexcept function pointer)");
    static_assert(std::is_convertible_v<SvpFn, ExtSvpFn>,
                  "makeVhpFvar: SvpFn must be convertible to ExtSvpFn (a captureless lambda "
                  "marked noexcept, or a noexcept function pointer)");
    static_assert(std::is_convertible_v<VhpFn, ExtVhpFn>,
                  "makeVhpFvar: VhpFn must be convertible to ExtVhpFn (a captureless lambda "
                  "marked noexcept, or a noexcept function pointer)");
    return detail::makeExternal(value, args, nArgs, static_cast<ExtVjpFn>(vjp), nullptr, 0,
                                static_cast<ExtSvpFn>(svp), static_cast<ExtVhpFn>(vhp));
}

/// Full first+second order External block with a payload copied into the tape.
template <typename VjpFn, typename SvpFn, typename VhpFn, typename P>
RevScalar makeVhpFvar(double value, const RevScalar* args, std::uint32_t nArgs, VjpFn vjp,
                      SvpFn svp, VhpFn vhp, const P& payload) {
    static_assert(std::is_convertible_v<VjpFn, ExtVjpFn>,
                  "makeVhpFvar: VjpFn must be convertible to ExtVjpFn (a captureless lambda "
                  "marked noexcept, or a noexcept function pointer)");
    static_assert(std::is_convertible_v<SvpFn, ExtSvpFn>,
                  "makeVhpFvar: SvpFn must be convertible to ExtSvpFn (a captureless lambda "
                  "marked noexcept, or a noexcept function pointer)");
    static_assert(std::is_convertible_v<VhpFn, ExtVhpFn>,
                  "makeVhpFvar: VhpFn must be convertible to ExtVhpFn (a captureless lambda "
                  "marked noexcept, or a noexcept function pointer)");
    return detail::makeExternal(value, args, nArgs, static_cast<ExtVjpFn>(vjp), payload,
                                static_cast<ExtSvpFn>(svp), static_cast<ExtVhpFn>(vhp));
}

// ── single-argument conveniences (nArgs = 1) ──

template <typename VjpFn>
RevScalar makeVjpVar(double value, const RevScalar& x, VjpFn vjp) {
    return makeVjpVar(value, &x, 1u, vjp);
}

template <typename VjpFn, typename P>
RevScalar makeVjpVar(double value, const RevScalar& x, VjpFn vjp, const P& payload) {
    return makeVjpVar(value, &x, 1u, vjp, payload);
}

template <typename VjpFn, typename SvpFn, typename VhpFn>
RevScalar makeVhpFvar(double value, const RevScalar& x, VjpFn vjp, SvpFn svp, VhpFn vhp) {
    return makeVhpFvar(value, &x, 1u, vjp, svp, vhp);
}

template <typename VjpFn, typename SvpFn, typename VhpFn, typename P>
RevScalar makeVhpFvar(double value, const RevScalar& x, VjpFn vjp, SvpFn svp, VhpFn vhp,
                      const P& payload) {
    return makeVhpFvar(value, &x, 1u, vjp, svp, vhp, payload);
}

} // namespace lightape

#endif // LIGHTAPE_OPS_CALLBACKS_H
